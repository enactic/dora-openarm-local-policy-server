"""Readiness uses the inference socket and never admits a premature Start."""

import json
import queue
import socket
import threading
from types import SimpleNamespace

import numpy as np
from openarm_policy_runtime import Backend, ModelSession, Prediction
from openarm_policy_runtime.logging import json_default
import pyarrow as pa
import pytest

from dora_openarm_local_policy_server import main as bridge
from dora_openarm_local_policy_server.shm_ring import SharedMemoryRingWriter


def run_bridge(sock, directory, **kwargs):
    writer = SharedMemoryRingWriter(directory)
    try:
        return bridge._main_dora(sock, shm_writer=writer, **kwargs)
    finally:
        writer.close()


def prediction(request, execution=None):
    return {"positions": [[0.0] * 16], "interval": 33333333,
            "generated_timestamp_ns": 100, "execution": execution,
            "reset_applied": request["reset_reason"] is not None,
            "input_sequence": request["shm"]["sequence"],
            "released_input_sequences": [request["shm"]["sequence"]]}


def command(value):
    """Build a lifecycle input for the fake Dora node."""
    return {
        "type": "INPUT",
        "id": "command",
        "value": pa.array([value]),
        "metadata": {},
    }


class Node:
    """Drive readiness scenarios without a running Dora dataflow."""

    def __init__(self, mode):
        """Initialize event queues and readiness signals."""
        self.mode, self.events, self.outputs = mode, queue.Queue(), []
        self.early_consumed, self.ready = threading.Event(), threading.Event()
        self.failed = threading.Event()

    def next(self, timeout):
        """Read a queued event using Dora-like timeout behavior."""
        try:
            event = self.events.get(timeout=timeout)
            if event.get("id") == "command":
                self.early_consumed.set()
            return event
        except queue.Empty:
            return {
                "type": "ERROR",
                "error": "Timeout event stream error: Receiver timed out",
            }

    def send_output(self, name, values, metadata):
        """Record outputs and enqueue the scenario's next inputs."""
        self.outputs.append((name, values, metadata))
        if name == "status" and values[0].as_py() == "loading":
            self.events.put(
                {"type": "STOP"} if self.mode == "stopped" else command("start")
            )
        elif name == "status" and values[0].as_py() == "ready":
            self.ready.set()
            if self.mode in {"ok", "cancel-in-flight"}:
                self.events.put(
                    {
                        "type": "INPUT",
                        "id": "observation",
                        "value": observation(),
                        "metadata": {"timestamp": 1},
                    }
                )
                # With no accepted Start, the preceding observation must be ignored.
                self.events.put(command("start"))
                self.events.put({"type": "INPUT", "id": "execution_plan",
                                 "value": pa.array([{"chunk_id": "accepted"}]), "metadata": {}})
                self.events.put(
                    {
                        "type": "INPUT",
                        "id": "observation",
                        "value": observation(),
                        "metadata": {"timestamp": 2},
                    }
                )
        elif name == "status" and values[0].as_py() == "error":
            self.failed.set()
        elif name == "actions":
            self.events.put({"type": "STOP"})


def observation():
    """Build the minimal observation used by the mock policy."""
    return pa.StructArray.from_arrays(
        [pa.array([1]), pa.array([[0.0] * 16], type=pa.list_(pa.float32()))],
        names=["id", "position"],
    )


@pytest.mark.parametrize("mode", ["ok", "rejected", "disconnected", "cancel-in-flight"])
def test_socket_readiness(monkeypatch, tmp_path, mode):
    """Check the handshake, premature Start, disconnect, and shutdown paths."""
    node, requests, failures = Node(mode), [], []
    monkeypatch.setattr(bridge.dora, "Node", lambda: node)
    monkeypatch.setenv("LOCAL_POLICY_TIMING_EVERY", "0")
    path = str(tmp_path / "policy.sock")

    def server():
        try:
            assert node.early_consumed.wait(2)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(path)
                listener.listen(1)
                conn, _ = listener.accept()
                with conn, conn.makefile("rw") as io:
                    requests.append(json.loads(io.readline()))
                    io.write(
                        json.dumps({"ready": mode != "rejected", "positions": [],
                                    "input_mode": "copy", "shm_slot_count": 3})
                        + "\n"
                    )
                    io.flush()
                    if mode == "ok":
                        requests.append(json.loads(io.readline()))
                        io.write(
                            json.dumps(
                                prediction(requests[-1])
                            )
                            + "\n"
                        )
                        io.flush()
                        assert io.readline() == ""
                    elif mode == "cancel-in-flight":
                        requests.append(json.loads(io.readline()))
                        node.events.put({"type": "STOP"})
                        assert io.readline() == ""
                    elif mode == "disconnected":
                        assert node.ready.wait(2)
                if mode == "disconnected":
                    assert not node.failed.wait(0.35)
                    node.events.put(command("start"))
                    node.events.put(
                        {
                            "type": "INPUT",
                            "id": "observation",
                            "value": observation(),
                            "metadata": {"timestamp": 3},
                        }
                    )
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=server, daemon=True)
    worker.start()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        if mode in {"ok", "cancel-in-flight"}:
            run_bridge(sock, tmp_path, socket_path=path)
        else:
            with pytest.raises((ConnectionError, RuntimeError)):
                run_bridge(sock, tmp_path, socket_path=path)
    worker.join(2)
    assert not worker.is_alive() and not failures
    assert requests[0] == {"ping": True}
    statuses = [value[0].as_py() for name, value, _ in node.outputs if name == "status"]
    if mode == "ok":
        assert statuses == ["loading", "ready"]
        assert len(requests) == 2 and requests[1]["metadata"]["timestamp"] == 2
        assert requests[1]["reset_reason"] == "request"
        assert requests[1]["execution_plan"] is None  # Restart cannot reuse a prior plan.
        assert node.outputs[-1][0] == "actions" and node.outputs[-1][2]["reset"]
    elif mode == "cancel-in-flight":
        assert statuses == ["loading", "ready"]
        assert len(requests) == 2 and requests[1]["reset_reason"] == "request"
        assert not any(name == "actions" for name, _, _ in node.outputs)
    else:
        assert statuses == (
            ["loading", "error"]
            if mode == "rejected"
            else ["loading", "ready", "error"]
        )
        assert not any(name == "actions" for name, _, _ in node.outputs)


@pytest.mark.parametrize("mode", ["timeout", "stopped"])
def test_startup_timeout_or_cancel(monkeypatch, tmp_path, mode):
    """Report a startup timeout while allowing orderly cancellation."""
    node = Node(mode)
    monkeypatch.setattr(bridge.dora, "Node", lambda: node)
    monkeypatch.setenv("POLICY_START_TIMEOUT_SEC", "0" if mode == "timeout" else "1")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        if mode == "timeout":
            with pytest.raises(TimeoutError):
                run_bridge(
                    sock, tmp_path, socket_path=str(tmp_path / "absent.sock")
                )
        else:
            run_bridge(sock, tmp_path, socket_path=str(tmp_path / "absent.sock"))
    expected = ["loading", "error"] if mode == "timeout" else ["loading"]
    assert [v[0].as_py() for n, v, _ in node.outputs if n == "status"] == expected


@pytest.mark.parametrize("switch_at", ["in-flight", "waiting-for-adoption"])
def test_task_switch_unblocks_rtc_and_discards_old_attempt(monkeypatch, tmp_path, switch_at):
    """A new observation attempt replaces outstanding work or an old ACK wait."""
    requests, failures = [], []
    ids = iter(("bootstrap", "old", "new", "next"))
    monkeypatch.setattr("dora_openarm_local_policy_server.session.uuid.uuid4",
                        lambda: SimpleNamespace(hex=next(ids)))

    def obs(attempt, timestamp):
        return {"type": "INPUT", "id": "observation", "value": observation(),
                "metadata": {"timestamp": timestamp, "episode_attempt_id": attempt}}

    def feedback(attempt, chunk):
        return {"type": "INPUT", "id": "execution_plan",
                "value": pa.array([{"chunk_id": chunk, "sample_chunk_id": chunk,
                                    "positions": [[0.0] * 16]}]),
                "metadata": {"episode_attempt_id": attempt}}

    class SwitchingNode(Node):
        def __init__(self):
            super().__init__("switch")
            self.switched = threading.Event()

        def next(self, timeout):
            event = super().next(timeout)
            if event.get("id") == "barrier":
                self.switched.set()
            return event

        def switch(self):
            for event in (obs("b", 4), feedback("a", "old"),
                          {"type": "INPUT", "id": "barrier"}):
                self.events.put(event)

        def send_output(self, name, values, metadata):
            self.outputs.append((name, values, metadata))
            if name == "status" and values[0].as_py() == "ready":
                start = command("start")
                start["metadata"]["episode_attempt_id"] = "a"
                self.events.put(start)
                self.events.put(obs("stale-before-start", 0))
                self.events.put(obs("a", 1))
            elif name == "actions":
                if metadata["chunk_id"] == "bootstrap":
                    self.events.put(feedback("a", "bootstrap"))
                    self.events.put(obs("a", 2))
                elif metadata["chunk_id"] == "old":
                    assert switch_at == "waiting-for-adoption"
                    self.switch()
                elif metadata["chunk_id"] == "new":
                    self.events.put(feedback("b", "new"))
                    self.events.put(feedback("a", "old"))
                    self.events.put(obs("b", 5))
                else:
                    self.events.put({"type": "STOP"})

    node = SwitchingNode()
    monkeypatch.setattr(bridge.dora, "Node", lambda: node)
    monkeypatch.setenv("LOCAL_POLICY_TIMING_EVERY", "0")
    path = str(tmp_path / "task-switch.sock")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(path)
        listener.listen(1)

        def server():
            try:
                conn, _ = listener.accept()
                conn.settimeout(3)
                with conn, conn.makefile("rw") as io:
                    assert json.loads(io.readline()) == {"ping": True}
                    io.write('{"ready":true,"input_mode":"copy","shm_slot_count":3}\n')
                    io.flush()
                    for chunk_id in ("bootstrap", "old", "new", "next"):
                        request = json.loads(io.readline())
                        requests.append(request)
                        if chunk_id == "old" and switch_at == "in-flight":
                            node.switch()
                            assert node.switched.wait(2)
                        assert request["chunk_id"] == chunk_id
                        io.write(json.dumps(prediction(request, {"based_on_chunk_id": "",
                                                                 "action_window_start": 0})) + '\n')
                        io.flush()
                    assert io.readline() == ""
            except BaseException as exc:
                failures.append(exc)
                node.events.put({"type": "STOP"})

        worker = threading.Thread(target=server, daemon=True)
        worker.start()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            run_bridge(sock, tmp_path, socket_path=path, infer_hz=1000)
        worker.join(3)
    assert not worker.is_alive() and not failures
    assert [r["metadata"]["timestamp"] for r in requests] == [1, 2, 4, 5]
    assert [bool(r["reset_reason"]) for r in requests] == [True, False, True, False]
    assert requests[2]["execution_plan"] is None
    assert requests[3]["execution_plan"]["chunk_id"] == "new"
    outputs = [m for name, _, m in node.outputs if name == "actions"]
    assert [m["chunk_id"] for m in outputs] == (
        ["bootstrap", "new", "next"] if switch_at == "in-flight"
        else ["bootstrap", "old", "new", "next"])


def test_borrowed_ring_survives_bursts_and_stale_response(monkeypatch, tmp_path):
    """Actual bridge/socket/mmap, with delayed model reads and many unsent observations."""
    consumed, failures, requests, replies = [], [], [], []

    class BorrowBackend(Backend):
        supports_borrowed_observations = True
        max_retained_inputs = 2

        def __init__(self):
            self.held = None

        def check(self):
            if self.held is not None:
                obs, expected = self.held
                np.testing.assert_array_equal(obs.qpos, expected)

        def reset(self, reason):
            self.check()
            if self.held is not None:
                self.held[0].release_input()
                self.held = None
            return False

        def predict(self, obs):
            self.reset("consumed")
            consumed.append(int(obs.qpos[0, 0]))
            self.held = obs, obs.qpos.copy()
            return Prediction(np.zeros((1, 16), dtype=np.float32))

    class BurstingNode(Node):
        def __init__(self):
            super().__init__("borrow")
            self.barrier = threading.Event()
            self.actions = 0

        def next(self, timeout):
            event = super().next(timeout)
            if event.get("id") == "barrier":
                self.barrier.set()
            return event

        def observe(self, number, attempt):
            value = pa.StructArray.from_arrays(
                [pa.array([number]), pa.array([[float(number)] * 16], type=pa.list_(pa.float32()))],
                names=["id", "position"],
            )
            self.events.put({"type": "INPUT", "id": "observation", "value": value,
                             "metadata": {"timestamp": number, "episode_attempt_id": attempt}})

        def send_output(self, name, values, metadata):
            self.outputs.append((name, values, metadata))
            if name == "status" and values[0].as_py() == "ready":
                self.events.put(command("start"))
                self.observe(1, "a")
            elif name == "actions":
                self.actions += 1
                if self.actions == 2:  # The middle reply changes task while in flight.
                    self.events.put({"type": "STOP"})

    node, backend = BurstingNode(), BorrowBackend()
    monkeypatch.setattr(bridge.dora, "Node", lambda: node)
    monkeypatch.setenv("LOCAL_POLICY_TIMING_EVERY", "0")
    path = str(tmp_path / "borrow.sock")
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(path)
        listener.listen(1)

        def server():
            session = ModelSession(backend, borrow_inputs=True)
            try:
                conn, _ = listener.accept()
                conn.settimeout(3)
                with conn, conn.makefile("rw") as io:
                    io.write(json.dumps(session.handle(json.loads(io.readline()))) + '\n')
                    io.flush()
                    for begin, end, attempt in ((2, 20, "a"), (21, 40, "b"), (41, 60, "b")):
                        request = json.loads(io.readline())
                        requests.append(request)
                        reply = session.handle(request)
                        replies.append(reply)
                        node.barrier.clear()
                        for number in range(begin, end + 1):
                            node.observe(number, attempt)
                        node.events.put({"type": "INPUT", "id": "barrier"})
                        assert node.barrier.wait(2)
                        backend.check()  # Five slots have wrapped many times; held input is intact.
                        io.write(json.dumps(reply, default=json_default) + '\n')
                        io.flush()
                    assert io.readline() == ""
            except BaseException as exc:
                failures.append(exc)
                node.events.put({"type": "STOP"})
            finally:
                session.close()

        worker = threading.Thread(target=server, daemon=True)
        worker.start()
        with socket.socket(socket.AF_UNIX) as sock:
            run_bridge(sock, tmp_path, socket_path=path, infer_hz=50)
        worker.join(4)
    assert not worker.is_alive() and not failures
    assert consumed == [1, 20, 40]
    assert [r["shm"]["sequence"] for r in requests] == [1, 20, 40]
    assert all(r["shm"]["slot_count"] == 5 for r in requests)
    assert [r["released_input_sequences"] for r in replies] == [[], [1], [20]]
    assert node.actions == 2 and backend.held is None
