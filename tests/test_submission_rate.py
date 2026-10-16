"""Pacing must leave the latest slot available to the input thread."""

import json
import threading
from types import SimpleNamespace

import pytest

from dora_openarm_local_policy_server import main as bridge
from dora_openarm_local_policy_server.session import Session


@pytest.mark.parametrize("change", ["latest", "restart", "stop", "ack", "slow"])
def test_wait_before_taking_latest(change, monkeypatch):
    now, sent, waits = [0.0], [], []
    cond, stopped = threading.Condition(), threading.Event()
    shared = dict(latest=None, generation=1, active=True, error=None,
                  in_flight_resource=None, waiting_chunk=None, retained_resources={})
    monkeypatch.setattr(bridge.time, "monotonic", lambda: now[0])

    def prepare(index):
        shared["latest"] = bridge._PreparedRequest(
            request={"metadata": {"id": index, "timestamp": index},
                     "shm": {"sequence": index + 1, "task_prompt": "task"}},
            request_json="", event_ns=0, ready_ns=0,
            event_period_ms=None, prepare_input_ms=0, json_request_ms=0,
            reset=False, generation=shared["generation"], should_log_timing=False,
        )

    def wait(timeout=None):
        waits.append(timeout)
        assert len(waits) <= 3
        if len(waits) == 1:
            assert timeout == pytest.approx(0.25)
            now[0] = 0.1
            if change == "restart":
                shared["generation"] += 1
            prepare(2)
            if change == "stop":
                shared.update(active=False, latest=None)
        elif timeout is not None:
            now[0] += timeout
        else:
            if change == "stop":
                stopped.set()
            else:
                assert shared["waiting_chunk"] == sent[0][1]["chunk_id"]
                shared["waiting_chunk"] = None
                shared["execution_plan"] = {"chunk_id": sent[0][1]["chunk_id"], "positions": [[0.0]]}

    monkeypatch.setattr(cond, "wait", wait)

    class IO:
        def write(self, line):
            sent.append((now[0], json.loads(line)))

        def flush(self):
            pass

        def readline(self):
            if change == "slow":
                now[0] += 0.35
            prepare(1)
            request = sent[-1][1]
            execution = {"based_on_chunk_id": "", "action_window_start": 0} if change == "ack" else None
            return json.dumps(dict(positions=[[0.0]], interval=33_333_333,
                                   generated_timestamp_ns=100, execution=execution,
                                   reset_applied=request["reset_reason"] is not None,
                                   input_sequence=request["shm"]["sequence"],
                                   released_input_sequences=[request["shm"]["sequence"]]))

    def output(*args):
        if len(sent) == 2:
            stopped.set()

    prepare(0)
    bridge._run_requests(SimpleNamespace(send_output=output), IO(), cond, shared, stopped,
                         Session(timing_log_every=0), infer_hz=4)
    expected = [0] if change == "stop" else [0, 1 if change == "slow" else 2]
    assert shared["retained_resources"] == {}
    assert [request["metadata"]["id"] for _, request in sent] == expected
    if len(sent) == 2:
        assert sent[1][0] == pytest.approx({"restart": 0.1, "slow": 0.35}.get(change, 0.25))
        assert bool(sent[1][1]["reset_reason"]) == (change == "restart")
