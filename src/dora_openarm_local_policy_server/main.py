# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to communicate with a local policy server."""

import argparse
import contextlib
from dataclasses import dataclass
import dora
import json
import logging
import math
import os
import pyarrow as pa
import socket
import tempfile
import threading
import time

from dora_openarm_local_policy_server.shm_ring import (
    SHM_RING_TRANSPORT,
    SharedMemoryRingWriter,
)
from dora_openarm_local_policy_server.session import Session, launch_options
from dora_openarm_local_policy_server.execution import (
    accept_execution_plan, fresh_observation, reset_execution, wait_for_execution,
)


START_COMMANDS = {"start"}
STOP_COMMANDS = {"stop", "intervene", "quit"}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass
class _PreparedRequest:
    request: dict
    request_json: str
    event_ns: int
    ready_ns: int
    event_period_ms: float | None
    prepare_input_ms: float
    json_request_ms: float
    reset: bool
    generation: int
    should_log_timing: bool
    transport: str = SHM_RING_TRANSPORT
    resource: tuple | None = None
    payload_bytes: int = 0






def _format_ms(value):
    return "NA" if value is None else f"{value:.2f}ms"


def _new_prepare_state():
    return {
        "previous_observation_id": None,
        "last_event_ns": None,
        "timing_index": 0,
        "generation": 0,
    }


def _start_episode(state):
    state["generation"] += 1
    state["previous_observation_id"] = None
    return state["generation"]


def _update_observation_generation(observation, state):
    observation_id = max(observation.field("id").to_pylist())
    previous_id = state["previous_observation_id"]
    if state["generation"] == 0 or (
        previous_id is not None and observation_id < previous_id
    ):
        _start_episode(state)
    state["previous_observation_id"] = observation_id
    return state["generation"]


def _prepare_request(
    event,
    state,
    shm_writer,
    generation,
    protected_resources=(),
):
    event_ns = time.perf_counter_ns()
    event_period_ms = (
        None
        if state["last_event_ns"] is None
        else (event_ns - state["last_event_ns"]) / 1e6
    )
    state["last_event_ns"] = event_ns
    state["timing_index"] += 1

    prepare_start_ns = time.perf_counter_ns()
    result = shm_writer.write(event["value"], event["metadata"], protected_resources)
    resource, payload_bytes = result.resource, result.bytes_written
    request = {"transport": SHM_RING_TRANSPORT, "shm": result.descriptor,
               "metadata": event["metadata"]}
    ready_ns = time.perf_counter_ns()

    return _PreparedRequest(
        request=request,
        request_json="",
        event_ns=event_ns,
        ready_ns=ready_ns,
        event_period_ms=event_period_ms,
        prepare_input_ms=(ready_ns - prepare_start_ns) / 1e6,
        json_request_ms=0.0,
        reset=False,
        generation=generation,
        should_log_timing=(
            state["timing_every"] > 0
            and state["timing_index"] % state["timing_every"] == 0
        ),
        resource=resource,
        payload_bytes=payload_bytes,
    )


def _serialize_request(prepared, control):
    start_ns = time.perf_counter_ns()
    prepared.reset = control.get("reset_reason") is not None
    prepared.request.update(control)
    prepared.request_json = json.dumps(prepared.request)
    prepared.json_request_ms = (time.perf_counter_ns() - start_ns) / 1e6


def _send_actions(node, actions):
    if not actions["positions"]:
        return False
    metadata = dict(actions.get("metadata", {}))
    metadata["interval"] = actions["interval"]
    if "cutoff_hz" in actions:
        metadata["cutoff_hz"] = actions["cutoff_hz"]
    node.send_output(
        "actions",
        pa.array(actions["positions"], type=pa.list_(pa.float32())),
        metadata,
    )
    return True




def _validate_input_ack(prepared, actions):
    """Match this response to its request; input release is a separate notification."""
    expected = prepared.request["shm"]["sequence"]
    actual = actions.get("input_sequence")
    if actual != expected:
        raise RuntimeError(
            f"Policy server acknowledged input sequence {actual!r}, expected {expected}"
        )


def _print_local_timing(
    *,
    mode,
    prepared,
    t_send_start_ns,
    t_request_sent_ns,
    t_response_done_ns,
    t_response_parsed_ns,
    t_loop_done_ns,
    actions,
    request_period_ms=None,
    next_request_wait_ms=None,
    dropped_stale=False,
):
    timing = actions.get("timing") or {}
    print(
        "[local-policy timing] "
        f"mode={mode} "
        f"transport={prepared.transport} "
        f"loop={(t_loop_done_ns - prepared.event_ns) / 1e6:.2f}ms "
        f"event_period={_format_ms(prepared.event_period_ms)} "
        f"request_period={_format_ms(request_period_ms)} "
        f"next_request_wait={_format_ms(next_request_wait_ms)} "
        f"event_to_send={(t_send_start_ns - prepared.event_ns) / 1e6:.2f}ms "
        f"queued={(t_send_start_ns - prepared.ready_ns) / 1e6:.2f}ms "
        f"prepare_input={prepared.prepare_input_ms:.2f}ms "
        f"payload={prepared.payload_bytes / (1024 * 1024):.2f}MiB "
        f"json_request={prepared.json_request_ms:.2f}ms "
        f"write_flush={(t_request_sent_ns - t_send_start_ns) / 1e6:.2f}ms "
        f"response_wait={(t_response_done_ns - t_request_sent_ns) / 1e6:.2f}ms "
        f"socket_rtt={(t_response_done_ns - t_send_start_ns) / 1e6:.2f}ms "
        f"json_response={(t_response_parsed_ns - t_response_done_ns) / 1e6:.2f}ms "
        f"send_output={(t_loop_done_ns - t_response_parsed_ns) / 1e6:.2f}ms "
        f"positions={len(actions.get('positions') or [])} "
        f"reset={prepared.reset} "
        f"dropped_stale={int(dropped_stale)} "
        f"server_policy={_format_ms(timing.get('policy_ms'))} "
        f"server_ready={_format_ms(timing.get('response_ready_ms'))} "
        f"server_input={_format_ms(timing.get('input_read_ms', timing.get('arrow_read_ms')))} "
        f"server_parse={_format_ms(timing.get('input_parse_ms', timing.get('parse_observations_ms')))} "
        f"server_transport={SHM_RING_TRANSPORT}",
        flush=True,
    )


def _send_status(node, value, message=""):
    node.send_output(
        "status", pa.array([value]), {"timestamp": time.time_ns(), "message": message}
    )


@contextlib.contextmanager
def _connect_ready(sock, path, timeout, stopped):
    """Wait for warmup and handshake on the connection used for inference."""
    deadline = time.monotonic() + timeout
    while not stopped.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Policy startup timed out after {timeout}s: {path}")
        sock.settimeout(remaining)
        try:
            sock.connect(path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            stopped.wait(min(0.1, remaining))
    else:
        raise InterruptedError("Policy startup cancelled")
    with sock.makefile("rw") as io:
        io.write('{"ping": true}\n')
        io.flush()
        response = io.readline()
        if not response:
            raise ConnectionError("Policy server closed the readiness handshake")
        reply = json.loads(response)
        if reply.get("ready") is not True:
            raise RuntimeError(
                reply.get("error") or "Policy server did not acknowledge ready"
            )
        sock.settimeout(None)
        yield io, reply


def _main_dora(
    sock, *, socket_path, shm_writer, infer_hz=10.0, **session_options
):
    timing_every = max(0, _env_int("LOCAL_POLICY_TIMING_EVERY", 20))
    session = Session(**session_options)
    state = _new_prepare_state()
    state["timing_every"] = timing_every

    node = dora.Node()
    cond = threading.Condition()
    stopped = threading.Event()
    shared = {
        "latest": None,
        "error": None,
        "in_flight_resource": None,
        "retained_resources": {},
        "active": False,
        "ready": False,
        "generation": 0,
        "execution_plan": None,
        "attempt_id": None,
        "waiting_chunk": None,
        "stop_and_go": session.stop_and_go,
        "observation_after_ns": None,
    }

    def reader_loop():
        try:
            while not stopped.is_set():
                event = node.next(timeout=0.1)
                if event is None or event["type"] == "STOP":
                    break
                if event["type"] == "ERROR":
                    if str(event["error"]).startswith("Timeout"):
                        continue
                    raise RuntimeError(event["error"])
                if event["type"] == "INPUT_CLOSED" and event["id"] == "observation":
                    break
                if event["type"] != "INPUT":
                    continue

                event_id = event["id"]
                if event_id == "command":
                    command = event["value"][0].as_py()
                    with cond:
                        if command in START_COMMANDS:
                            if not shared["ready"]:
                                continue
                            shared["generation"] = _start_episode(state)
                            shared["active"] = True
                            shared["latest"] = None
                        elif command in STOP_COMMANDS:
                            shared["active"] = False
                            shared["latest"] = None
                        else:
                            continue
                        reset_execution(shared)
                        shared["attempt_id"] = event["metadata"].get("episode_attempt_id")
                        cond.notify_all()
                    continue

                if event_id == "execution_plan":
                    with cond:
                        accept_execution_plan(shared, event)
                        cond.notify_all()
                    continue

                if event_id != "observation":
                    continue
                with cond:
                    if not shared["active"]:
                        continue
                    attempt = event["metadata"].get("episode_attempt_id")
                    # Start fences the first observation, not all later task attempts.
                    if (
                        state["previous_observation_id"] is None
                        and shared["attempt_id"] is not None and attempt != shared["attempt_id"]
                    ):
                        continue
                    if (
                        attempt is not None and shared["attempt_id"] is not None
                        and attempt != shared["attempt_id"]
                    ):
                        _start_episode(state)
                    if attempt is not None:
                        shared["attempt_id"] = attempt
                    generation = _update_observation_generation(event["value"], state)
                    if shared["generation"] != generation:
                        shared["latest"] = None
                        reset_execution(shared)
                    shared["generation"] = generation
                    if not fresh_observation(shared, event["metadata"]):
                        continue
                    latest = shared["latest"]
                    protected = {
                        shared["in_flight_resource"],
                        latest.resource if latest is not None else None,
                        *shared["retained_resources"].values(),
                    }

                prepared = _prepare_request(
                    event,
                    state,
                    shm_writer,
                    generation,
                    protected,
                )
                with cond:
                    if (shared["active"] and shared["generation"] == generation
                            and fresh_observation(shared, prepared.request["metadata"])):
                        shared["latest"] = prepared
                        cond.notify()
        except BaseException as exc:
            with cond:
                shared["error"] = exc
        finally:
            with cond:
                stopped.set()
                shared["active"] = False
                shared["latest"] = None
                cond.notify_all()
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)

    reader = threading.Thread(
        target=reader_loop,
        name="local-policy-observation-reader",
        daemon=True,
    )
    _send_status(node, "loading")
    reader.start()
    try:
        timeout = _env_int("POLICY_START_TIMEOUT_SEC", 600)
        with _connect_ready(sock, socket_path, timeout, stopped) as (io, ready):
            with cond:
                if shared["error"] is not None:
                    raise shared["error"]
                if stopped.is_set():
                    return
                shm_writer.slot_count = ready["shm_slot_count"]
                print(f"Policy input mode: {ready['input_mode']}, SHM slots: {shm_writer.slot_count}", flush=True)
                shared["ready"] = True
                _send_status(node, "ready")
            _run_requests(node, io, cond, shared, stopped, session, infer_hz=infer_hz)
    except Exception as exc:
        error = shared["error"]
        if stopped.is_set() and error is None:
            return  # Normal shutdown can interrupt connect/readline.
        error = error or exc
        _send_status(node, "error", str(error))
        raise error
    finally:
        stopped.set()
        reader.join()
        session.close()


def _run_requests(node, io, cond, shared, stopped, session, *, infer_hz=10.0):
    last_request_sent_ns = None
    last_submit = None
    last_submit_generation = None
    submit_interval = 1.0 / infer_hz
    while True:
        wait_start_ns = time.perf_counter_ns()
        with cond:
            while not stopped.is_set():
                delay = 0.0
                if last_submit is not None and shared["generation"] == last_submit_generation:
                    delay = last_submit + submit_interval - time.monotonic()
                if (shared["latest"] is not None and shared.get("waiting_chunk") is None and delay <= 0
                        and not session.requires_plan(shared["generation"], shared.get("execution_plan"))):
                    break
                cond.wait(timeout=delay if delay > 0 else None)
            if shared["error"] is not None:
                raise shared["error"]
            if stopped.is_set():
                break
            prepared = shared["latest"]
            shared["latest"] = None
            shared["in_flight_resource"] = prepared.resource
            shared["retained_resources"][prepared.request["shm"]["sequence"]] = prepared.resource
            plan = shared.get("execution_plan")
        wait_done_ns = time.perf_counter_ns()

        control = session.prepare(prepared.generation, prepared.request["metadata"],
                                  prepared.request["shm"].get("task_prompt"), plan)
        _serialize_request(prepared, control)
        send_start_ns = time.perf_counter_ns()
        request_period_ms = (
            None
            if last_request_sent_ns is None
            else (send_start_ns - last_request_sent_ns) / 1e6
        )
        last_submit = time.monotonic()
        last_submit_generation = prepared.generation
        io.write(prepared.request_json + "\n")
        io.flush()
        request_sent_ns = time.perf_counter_ns()
        last_request_sent_ns = request_sent_ns

        response = io.readline()
        response_done_ns = time.perf_counter_ns()
        if not response:
            raise ConnectionError("Policy server disconnected")
        result = json.loads(response)
        response_parsed_ns = time.perf_counter_ns()
        _validate_input_ack(prepared, result)

        with cond:
            # Releases still apply when Start/Stop or a task change discards the actions.
            for sequence in result["released_input_sequences"]:
                shared["retained_resources"].pop(sequence)
            if result.get("error"):
                raise RuntimeError(result["error"])
            dropped_stale = (
                not shared["active"] or shared["generation"] != prepared.generation
            )
            actions = {"positions": [], "timing": result.get("timing", {})}
            if not dropped_stale:
                actions = session.complete(control, result)
                metadata = actions["metadata"]
                if actions["positions"]:
                    wait_for_execution(shared, metadata)
                if _send_actions(node, actions):
                    session.sent(actions)
        loop_done_ns = time.perf_counter_ns()

        with cond:
            if shared["in_flight_resource"] == prepared.resource:
                shared["in_flight_resource"] = None

        if prepared.should_log_timing:
            _print_local_timing(
                mode="latest",
                prepared=prepared,
                t_send_start_ns=send_start_ns,
                t_request_sent_ns=request_sent_ns,
                t_response_done_ns=response_done_ns,
                t_response_parsed_ns=response_parsed_ns,
                t_loop_done_ns=loop_done_ns,
                actions=actions,
                request_period_ms=request_period_ms,
                next_request_wait_ms=(wait_done_ns - wait_start_ns) / 1e6,
                dropped_stale=dropped_stale,
            )


def main():
    """Communicate with a local policy server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Communicate with a local policy server"
    )
    parser.add_argument(
        "--socket",
        default=os.getenv("SOCKET"),
        help="The local socket to communicate",
        type=str,
    )
    parser.add_argument(
        "--infer-hz", type=float, default=None,
        help="Maximum request submission frequency; wait before selecting latest input",
    )
    args = parser.parse_args()
    options = launch_options()
    configured_hz = options.pop("infer_hz", 10.0)
    infer_hz = float(configured_hz if args.infer_hz is None else args.infer_hz)
    if not math.isfinite(infer_hz) or infer_hz <= 0:
        parser.error("--infer-hz must be positive and finite")

    with tempfile.TemporaryDirectory(
        prefix="dora-openarm-local-policy-server", dir="/dev/shm"
    ) as shared_dir:
        shm_writer = SharedMemoryRingWriter(shared_dir)
        print(f"Local policy input transport: {SHM_RING_TRANSPORT}", flush=True)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                _main_dora(
                    sock,
                    socket_path=args.socket,
                    infer_hz=infer_hz,
                    shm_writer=shm_writer,
                    **options,
                )
        finally:
            shm_writer.close()


if __name__ == "__main__":
    main()
