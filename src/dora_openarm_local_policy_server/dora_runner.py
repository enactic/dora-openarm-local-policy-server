"""Direct Dora transport: one latest input and one model worker, no socket/ring bridge."""

import contextlib
import math
import os
import threading
import time

import pyarrow as pa

from openarm_policy_runtime import ModelSession
from .session import Session, launch_options
from .execution import accept_execution_plan, fresh_observation, reset_execution, wait_for_execution


class LatestInput:
    """Track Start/Stop, task changes and observation-ID generation boundaries."""

    def __init__(self):
        self.condition = threading.Condition()
        self.ready = self.active = self.done = False
        self.generation = 0
        self.previous_id = self.latest = self.error = None
        self.execution_plan = None
        self.attempt_id = None
        self.waiting_chunk = None
        self.stop_and_go = False
        self.observation_after_ns = None

    def accept(self, event):
        received_ns = time.time_ns()
        with self.condition:
            if event["id"] == "command":
                command = event["value"][0].as_py()
                if command == "start" and self.ready:
                    self.generation += 1
                    self.previous_id = None
                    self.active = True
                    self.latest = None
                elif command in {"stop", "intervene", "quit"}:
                    self.active = False
                    self.latest = None
                else:
                    return
                reset_execution(vars(self))
                self.attempt_id = event["metadata"].get("episode_attempt_id")
                self.condition.notify_all()
                return
            if event["id"] == "execution_plan":
                accept_execution_plan(vars(self), event)
                self.condition.notify_all()
                return
            if event["id"] != "observation" or not self.ready or not self.active:
                return
            attempt = event["metadata"].get("episode_attempt_id")
            # Start fences the first observation; later IDs can mark an in-place task change.
            if (
                self.previous_id is None
                and self.attempt_id is not None
                and attempt != self.attempt_id
            ):
                return
            observation_id = max(event["value"].field("id").to_pylist())
            task_changed = (
                attempt is not None and self.attempt_id is not None and attempt != self.attempt_id
            )
            if task_changed or (self.previous_id is not None and observation_id < self.previous_id):
                self.generation += 1
                reset_execution(vars(self))
            if attempt is not None:
                self.attempt_id = attempt
            self.previous_id = observation_id
            if not fresh_observation(vars(self), event["metadata"]):
                return
            # Keeping the Arrow event retains Dora's allocation, including its drop notification.
            self.latest = (self.generation, event, received_ns)
            self.condition.notify_all()


def serve_dora(
    backend_factory,
    *,
    infer_hz=None,
    borrow_inputs=False,
    node=None,
    startup_timeout=None,
    **session_options,
):
    """Load/warm the backend inside the Dora process, then publish readiness and actions."""
    options = launch_options()
    configured_hz = options.pop("infer_hz", 10.0)
    infer_hz = float(configured_hz if infer_hz is None else infer_hz)
    options.update(session_options)
    if not math.isfinite(infer_hz) or infer_hz <= 0:
        raise ValueError("infer_hz must be positive and finite")
    if node is None:
        try:
            from dora import Node
        except ImportError as exc:
            raise RuntimeError(
                "Install openarm-policy-runtime[dora] in a Python >=3.11 model environment"
            ) from exc
        node = Node()
    state = LatestInput()
    stop_reader = threading.Event()
    backend = model = session = None

    def status(value, message=""):
        node.send_output(
            "status",
            pa.array([value]),
            {
                "timestamp": time.time_ns(),
                "message": message,
                "borrow_inputs": borrow_inputs,
            },
        )

    def report_error(exc):
        with state.condition:
            if state.error is not None:
                return
            state.error = exc
            state.active = False
            state.condition.notify_all()
            with contextlib.suppress(Exception):
                status("error", str(exc))

    def receive():
        try:
            while not stop_reader.is_set():
                event = node.next(timeout=0.1)
                if event is None or event["type"] == "STOP":
                    break
                if event["type"] == "ERROR":
                    if str(event["error"]).startswith("Timeout"):
                        continue
                    raise RuntimeError(event["error"])
                if event["type"] == "INPUT_CLOSED" and event["id"] == "observation":
                    break
                if event["type"] == "NODE_FAILED":
                    raise RuntimeError(str(event))
                if event["type"] == "INPUT":
                    state.accept(event)
        except BaseException as exc:
            report_error(exc)
        finally:
            with state.condition:
                state.done = True
                state.active = False
                state.latest = None
                state.condition.notify_all()

    if startup_timeout is None:
        startup_timeout = float(os.getenv("POLICY_START_TIMEOUT_SEC", "600"))

    def startup_expired():
        with state.condition:
            if state.ready or state.done:
                return
            report_error(TimeoutError(f"Policy startup timed out after {startup_timeout:g}s"))

    timer = threading.Timer(startup_timeout, startup_expired)
    timer.daemon = True
    status("loading")
    timer.start()
    reader = threading.Thread(target=receive, name="policy-dora-input", daemon=True)
    reader.start()
    try:
        session = Session(**options)
        state.stop_and_go = session.stop_and_go
        backend = backend_factory()
        if borrow_inputs and not backend.supports_borrowed_observations:
            raise ValueError(f"Backend {backend.name} does not support --borrow-inputs")
        backend.warmup()
        model = ModelSession(backend)
        with state.condition:
            if state.error is not None:
                raise state.error
            if state.done:
                return
            timer.cancel()
            state.ready = True
            status("ready")
        last_submit = None
        last_submit_generation = None
        submit_interval = 1.0 / infer_hz
        while True:
            with state.condition:
                while not state.done:
                    delay = 0.0
                    if last_submit is not None and state.generation == last_submit_generation:
                        delay = last_submit + submit_interval - time.monotonic()
                    if (
                        state.latest is not None
                        and state.waiting_chunk is None
                        and delay <= 0
                        and not session.requires_plan(state.generation, state.execution_plan)
                    ):
                        break
                    state.condition.wait(timeout=delay if delay > 0 else None)
                if state.error is not None:
                    raise state.error
                if state.done:
                    break
                generation, event, received_ns = state.latest
                state.latest = None
                execution_plan = state.execution_plan
                last_submit = time.monotonic()
                last_submit_generation = generation
            started = time.perf_counter()
            observation = model.reader.from_arrow(
                event["value"],
                event["metadata"],
                session.prompt,
                backend.camera_fields,
                copy=not borrow_inputs,
            )
            event = None
            observation.metadata["policy_node_received_timestamp_ns"] = received_ns
            request = session.prepare(
                generation, observation.metadata, observation.prompt, execution_plan
            )
            observation.execution_plan = request["execution_plan"]
            result = model.handle_observation(
                observation,
                control=request,
                input_read_ms=(time.perf_counter() - started) * 1000,
            )
            with state.condition:
                current = state.active and not state.done and generation == state.generation
                response = session.complete(request, result) if current else None
                if response is not None and response["positions"]:
                    metadata = dict(response["metadata"])
                    metadata.update(
                        interval=response["interval"],
                        policy_node_sent_timestamp_ns=time.time_ns(),
                    )
                    if "cutoff_hz" in response:
                        metadata["cutoff_hz"] = response["cutoff_hz"]
                    wait_for_execution(vars(state), metadata)
                    node.send_output(
                        "actions",
                        pa.array(response["positions"], type=pa.list_(pa.float32())),
                        metadata,
                    )
                    session.sent(response)
            # Match socket flush semantics even for prefill or a discarded old generation.
            backend.after_response_sent()
    except BaseException as exc:
        report_error(exc)
        raise
    finally:
        timer.cancel()
        stop_reader.set()
        reader.join()
        with state.condition:
            state.latest = None
        try:
            if session is not None:
                session.close()
        finally:
            try:
                if model is not None:
                    model.close()
            finally:
                if backend is not None:
                    backend.close()
