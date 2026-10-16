"""Strict command-completion scheduling without hardware or model weights."""

# ruff: noqa: D103

import json
import threading
from types import SimpleNamespace

import pyarrow as pa
import pytest

from dora_openarm_local_policy_server import main as bridge
from dora_openarm_local_policy_server.dora_runner import LatestInput, serve_dora
from dora_openarm_local_policy_server.execution import accept_execution_plan, fresh_observation
from dora_openarm_local_policy_server.session import Session
from test_dora import command, observation_event
from test_session import Model, run


def feedback(chunk="current", status="completed", attempt="one", timestamp=200, **extra):
    return {
        "id": "execution_plan", "type": "INPUT",
        "metadata": {"episode_attempt_id": attempt},
        "value": pa.array([{
            "sample_chunk_id": chunk, "chunk_id": chunk, "positions": [[0.0]],
            "execution_status": status, "feedback_timestamp_ns": timestamp,
            "completed_timestamp_ns": timestamp if status == "completed" else None,
            **extra,
        }]),
    }


def state_waiting_for_chunk():
    state = LatestInput()
    state.stop_and_go = state.ready = True
    state.accept(command("start"))
    state.accept(observation_event())
    state.waiting_chunk = "current"
    return state


@pytest.mark.parametrize("event", [
    feedback(status="adopted"), feedback(chunk="old"), feedback(attempt="old"),
    feedback(chunk_id="different-active-plan"), feedback(completed_timestamp_ns=None),
    {"id": "execution_plan", "value": pa.array([]), "metadata": {"episode_attempt_id": "one"}},
    {"id": "execution_plan", "value": pa.array([None]), "metadata": {"episode_attempt_id": "one"}},
    command("unrelated-command"),
])
def test_only_matching_completion_releases_wait(event):
    state = state_waiting_for_chunk()
    state.accept(event)
    assert state.waiting_chunk == "current"
    assert state.latest is not None


@pytest.mark.parametrize("status", ["completed", "rejected"])
def test_terminal_feedback_requires_observation_after_boundary(status):
    state = state_waiting_for_chunk()
    event = feedback(status=status, **({"chunk_id": "old-active-plan"} if status == "rejected" else {}))
    state.accept(event)
    assert state.waiting_chunk is state.latest is None
    for index, timestamp in enumerate((199, 200, 201), 1):
        observation = observation_event(index)
        observation["metadata"]["timestamp"] = timestamp
        state.accept(observation)
        assert (state.latest is not None) == (timestamp == 201)
    latest = state.latest
    state.accept(event)  # Duplicate must not discard a fresh observation.
    assert state.latest is latest


@pytest.mark.parametrize("reset", ["start", "stop", "intervene", "quit", "task"])
def test_lifecycle_clears_completion_boundary(reset):
    state = state_waiting_for_chunk()
    state.accept(feedback())
    state.waiting_chunk = "pending"
    if reset == "task":
        state.accept(observation_event(1, attempt="two"))
    else:
        state.accept(command(reset, attempt="two"))
    assert state.waiting_chunk is state.execution_plan is state.observation_after_ns is None
    state.accept(feedback(chunk="pending", attempt="one"))
    assert state.observation_after_ns is None


def test_sync_always_starts_at_zero_without_rtc_plan_or_idle_reset():
    session, model = Session(inference_mode="stop-and-go", action_window_size=3), Model()
    for _ in range(2):
        request, result = run(session, model, plan={"chunk_id": "old"})
        assert request["execution_plan"] is None
        response = session.complete(request, result)
        assert response["metadata"]["inference_mode"] == "stop-and-go"
        assert response["metadata"]["action_window_start"] == 0
        assert len(response["positions"]) == 3
        session.sent(response)
        session.last_end -= 10
    assert model.resets == ["request"]
    model.execution = {"based_on_chunk_id": "old"}
    with pytest.raises(ValueError, match="RTC disabled"):
        session.complete(*run(session, model))


def test_invalid_sync_settings():
    with pytest.raises(ValueError, match="inference_mode"):
        Session(inference_mode="typo")
    with pytest.raises(ValueError, match="window start"):
        Session(inference_mode="stop-and-go", action_window_start=1)


def test_socket_does_not_submit_next_request_until_done_and_fresh(monkeypatch):
    cond, stopped = threading.Condition(), threading.Event()
    shared = dict(latest=None, generation=1, active=True, error=None, attempt_id="one",
                  in_flight_resource=None, waiting_chunk=None, retained_resources={}, stop_and_go=True)
    sent, phases = [], []
    now = [0.0]
    monkeypatch.setattr(bridge.time, "monotonic", lambda: now[0])

    def observe(index, timestamp):
        metadata = {"timestamp": timestamp, "id": index, "episode_attempt_id": "one"}
        if fresh_observation(shared, metadata):
            shared["latest"] = bridge._PreparedRequest(
                request={"metadata": metadata, "shm": {"sequence": index + 1}},
                request_json="", event_ns=0, ready_ns=0, event_period_ms=None,
                prepare_input_ms=0, json_request_ms=0, reset=False, generation=1, should_log_timing=False,
            )

    def wait(timeout=None):
        assert len(sent) == 1
        chunk = sent[0]["chunk_id"]
        phases.append(len(phases))
        now[0] += 1
        if len(phases) == 1:
            accept_execution_plan(shared, feedback(chunk, "adopted"))
            observe(1, 150)
            assert shared["waiting_chunk"] == chunk
        elif len(phases) == 2:
            accept_execution_plan(shared, feedback(chunk))
            observe(2, 200)
            assert shared["latest"] is None
        elif len(phases) == 3:
            observe(3, 201)
        else:
            pytest.fail("Completion did not release the socket worker")

    monkeypatch.setattr(cond, "wait", wait)

    class IO:
        def write(self, value):
            sent.append(json.loads(value))

        def flush(self):
            pass

        def readline(self):
            request = sent[-1]
            return json.dumps(dict(
                positions=[[0.0]], interval=33_333_333, generated_timestamp_ns=100,
                reset_applied=True, input_sequence=request["shm"]["sequence"],
                released_input_sequences=[request["shm"]["sequence"]],
            ))

    def output(name, value, metadata):
        assert shared["waiting_chunk"] == metadata["chunk_id"]
        if len(sent) == 2:
            stopped.set()

    observe(0, 100)
    bridge._run_requests(SimpleNamespace(send_output=output), IO(), cond, shared, stopped,
                         Session(inference_mode="stop-and-go", timing_log_every=0))
    assert [request["metadata"]["id"] for request in sent] == [0, 3]
    assert shared["retained_resources"] == {}


def test_direct_worker_arms_gate_before_send_and_waits_for_new_observation(monkeypatch):
    state, finished = LatestInput(), threading.Event()
    seen, waits, now = [], [], [0.0]
    monkeypatch.setattr("dora_openarm_local_policy_server.dora_runner.LatestInput", lambda: state)
    monkeypatch.setattr("dora_openarm_local_policy_server.dora_runner.time.monotonic", lambda: now[0])

    def observe(index, timestamp):
        event = observation_event(index)
        event["metadata"].update(id=index, timestamp=timestamp)
        state.accept(event)

    def wait(timeout=None):
        assert len(seen) == 1
        now[0] += 1
        waits.append(1)
        if len(waits) == 1:
            observe(1, 150)
            state.accept(feedback(state.waiting_chunk, "adopted"))
            assert state.waiting_chunk is not None
        elif len(waits) == 2:
            state.accept(feedback(state.waiting_chunk))
            observe(2, 200)
            assert state.latest is None
        elif len(waits) == 3:
            observe(3, 201)
        else:
            pytest.fail("Direct worker did not resume")

    monkeypatch.setattr(state.condition, "wait", wait)

    class Backend(Model):
        def predict(self, observation):
            seen.append(observation.metadata["id"])
            return super().predict(observation)

    class Node:
        def next(self, timeout):
            if finished.wait(timeout):
                return {"type": "STOP"}
            return {"type": "ERROR", "error": "Timeout"}

        def send_output(self, name, value, metadata):
            if name == "status" and value[0].as_py() == "ready":
                state.accept(command("start"))
                observe(0, 100)
            elif name == "actions":
                assert state.waiting_chunk == metadata["chunk_id"]
                if len(seen) == 2:
                    state.done = True
                    finished.set()

    serve_dora(Backend, node=Node(), inference_mode="stop-and-go", timing_log_every=0)
    assert seen == [0, 3]
