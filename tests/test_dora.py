"""CPU-only direct-node contracts. No Dora daemon, model weights or hardware."""

import queue
import threading

import numpy as np
import pyarrow as pa
import pytest

from openarm_policy_runtime import Backend, ObservationReader, Prediction
from dora_openarm_local_policy_server.dora_runner import LatestInput, serve_dora


def observation_event(index=0, attempt="one"):
    values = pa.StructArray.from_arrays(
        [
            pa.array([index]),
            pa.array([[1.0] * 16], type=pa.list_(pa.float32())),
            pa.array([[7] * 12], type=pa.list_(pa.uint8())),
        ],
        names=["id", "position", "camera_ceiling"],
    )
    return {
        "type": "INPUT",
        "id": "observation",
        "value": values,
        "metadata": {
            "timestamp": 123,
            "history_delta_indices": "0",
            "episode_attempt_id": attempt,
            "camera_ceiling.height": 2,
            "camera_ceiling.width": 2,
        },
    }


def command(value, attempt="one"):
    return {
        "type": "INPUT",
        "id": "command",
        "value": pa.array([value]),
        "metadata": {"episode_attempt_id": attempt},
    }


def test_borrowed_arrow_is_readonly_and_owned_mode_still_copies():
    event = observation_event()
    reader = ObservationReader()
    borrowed = reader.from_arrow(event["value"], event["metadata"], copy=False)
    owned = reader.from_arrow(event["value"], event["metadata"])
    source = event["value"].field("position").values.to_numpy()
    assert borrowed.owner is event["value"] and owned.owner is None
    assert np.shares_memory(borrowed.qpos, source)
    assert not np.shares_memory(owned.qpos, source)
    with pytest.raises(ValueError):
        borrowed.qpos[0, 0] = 9
    owned.qpos[0, 0] = 9
    assert borrowed.qpos[0, 0] == 1


def test_latest_generation_and_stop_boundaries():
    state = LatestInput()
    state.accept(command("start"))
    assert not state.active
    state.ready = True
    state.accept(command("start"))
    state.accept(observation_event(10))
    state.accept(observation_event(11))
    assert state.generation == 1 and state.latest[1]["value"].field("id")[0].as_py() == 11
    state.accept(observation_event(0))
    assert state.generation == 2
    state.accept(command("stop"))
    state.accept(observation_event(1))
    assert not state.active and state.latest is None
    state.accept(command("start", "two"))
    state.accept(observation_event(12, "one"))
    assert state.latest is None  # Start's first observation must belong to its attempt.
    state.accept(observation_event(0, "two"))
    assert state.latest[0] == 3
    state.waiting_chunk = "old-pending"
    state.execution_plan = {"chunk_id": "old"}
    # In-place task changes are carried by the ordered observation stream.
    state.accept(observation_event(1, "three"))
    assert state.generation == 4 and state.active
    assert state.execution_plan is state.waiting_chunk is None
    assert state.latest[1]["metadata"]["episode_attempt_id"] == "three"
    stale_plan = {"type": "INPUT", "id": "execution_plan",
                  "value": pa.array([{"chunk_id": "old", "sample_chunk_id": "old-pending"}]),
                  "metadata": {"episode_attempt_id": "two"}}
    state.waiting_chunk = "new"
    state.accept(stale_plan)
    assert state.execution_plan is None and state.waiting_chunk == "new"
    current_plan = {**stale_plan, "value": pa.array([{"chunk_id": "new", "sample_chunk_id": "new"}]),
                    "metadata": {"episode_attempt_id": "three"}}
    state.accept(current_plan)
    assert state.waiting_chunk is None and state.execution_plan["chunk_id"] == "new"
    state.accept(observation_event(3, "three"))
    assert state.generation == 4 and state.latest[1]["metadata"]["episode_attempt_id"] == "three"
    state.accept(command("start", "four"))
    state.accept(observation_event(4, "three"))
    state.accept(current_plan)
    assert state.latest is state.execution_plan is None
    assert state.generation == 5 and state.attempt_id == "four"


@pytest.mark.parametrize("change", ["latest", "restart", "stop", "task"])
def test_submission_wait_keeps_latest_live(change, monkeypatch):
    state = LatestInput()
    now, seen, waits = [0.0], [], []
    finished = threading.Event()
    monkeypatch.setattr("dora_openarm_local_policy_server.dora_runner.LatestInput", lambda: state)
    monkeypatch.setattr("dora_openarm_local_policy_server.dora_runner.time.monotonic", lambda: now[0])

    def observe(index, attempt="one"):
        event = observation_event(index, attempt)
        event["metadata"]["id"] = index
        state.accept(event)

    def wait(timeout=None):
        waits.append(timeout)
        assert len(waits) <= 3
        if len(waits) == 1:
            assert timeout == pytest.approx(0.25)
            now[0] = 0.1
            if change in {"restart", "stop"}:
                state.accept(command("stop"))
            if change == "restart":
                state.accept(command("start"))
            observe(2, "two" if change == "task" else "one")
        elif timeout is not None:
            now[0] += timeout
        else:
            assert change == "stop" and state.latest is None
            state.done = True
            finished.set()

    monkeypatch.setattr(state.condition, "wait", wait)

    class Model(Backend):
        def predict(self, observation):
            seen.append((now[0], observation.metadata["id"]))
            return Prediction(np.zeros((1, 16), dtype=np.float32))

    class Node:
        def next(self, timeout):
            if finished.wait(timeout):
                return {"type": "STOP"}
            return {"type": "ERROR", "error": "Timeout"}

        def send_output(self, name, value, metadata):
            if name == "status" and value[0].as_py() == "ready":
                state.accept(command("start"))
                observe(0)
            elif name == "actions":
                if len(seen) == 1:
                    observe(1)
                else:
                    state.done = True
                    finished.set()

    serve_dora(Model, node=Node(), infer_hz=4, timing_log_every=0)
    assert [index for _, index in seen] == ([0] if change == "stop" else [0, 2])
    if len(seen) == 2:
        assert seen[1][0] == pytest.approx(0.1 if change in {"restart", "task"} else 0.25)


@pytest.mark.parametrize("borrow", [False, True])
@pytest.mark.parametrize("dependent", [False, True])
def test_node_warmup_then_first_action_and_shutdown(borrow, dependent, monkeypatch):
    events = queue.Queue()
    outputs, lifecycle = [], []
    state = LatestInput()
    monkeypatch.setattr("dora_openarm_local_policy_server.dora_runner.LatestInput", lambda: state)

    class Model(Backend):
        supports_borrowed_observations = True

        def warmup(self):
            lifecycle.append("warmup")

        def predict(self, observation):
            assert (observation.owner is not None) == borrow
            lifecycle.append("predict")
            execution = {"based_on_chunk_id": "", "action_window_start": 0} if dependent else None
            return Prediction(np.arange(48, dtype=np.float32).reshape(3, 16), execution=execution)

        def after_response_sent(self):
            lifecycle.append("handoff")

        def close(self):
            lifecycle.append("close")

    class Node:
        def next(self, timeout):
            try:
                return events.get(timeout=timeout)
            except queue.Empty:
                return {"type": "ERROR", "error": "Timeout event stream error: Receiver timed out"}

        def send_output(self, name, value, metadata):
            outputs.append((name, value, metadata))
            if name == "status" and value[0].as_py() == "ready":
                assert lifecycle == ["warmup"]
                events.put(command("start"))
                events.put(observation_event())
            elif name == "actions":
                assert state.waiting_chunk == (metadata["chunk_id"] if dependent else None)
                events.put({"type": "STOP"})

    serve_dora(
        Model,
        node=Node(),
        borrow_inputs=borrow,
        action_window_start=1,
        action_window_size=2,
        timing_log_every=0,
    )
    action = next(item for item in outputs if item[0] == "actions")
    assert action[2]["reset"] and action[2]["action_window_start"] == 0
    assert len(action[1]) == 2
    assert action[1][0].as_py() == list(range(16))
    assert (
        action[2]["policy_node_received_timestamp_ns"] <= action[2]["policy_node_sent_timestamp_ns"]
    )
    assert lifecycle == ["warmup", "predict", "handoff", "close"]


@pytest.mark.parametrize("cause", ["timeout", "reader", "backend"])
def test_startup_error_is_reported_once(cause):
    import threading

    released, statuses = threading.Event(), []

    class Model(Backend):
        def warmup(self):
            if cause == "backend":
                raise RuntimeError("Model load failed")
            assert released.wait(2)

    class Node:
        def next(self, timeout):
            if cause == "reader":
                return {"type": "ERROR", "error": "Input failed"}
            released.wait(timeout)
            return {"type": "ERROR", "error": "Timeout event stream error: Receiver timed out"}

        def send_output(self, name, value, metadata):
            statuses.append(value[0].as_py())
            if statuses[-1] == "error":
                released.set()

    error = TimeoutError if cause == "timeout" else RuntimeError
    with pytest.raises(error):
        serve_dora(Model, node=Node(), startup_timeout=0.02 if cause == "timeout" else 2)
    assert statuses == ["loading", "error"]
