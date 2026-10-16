"""Caller windows and lifecycle against the full-result model interface; CPU only."""

import json

import numpy as np
import pytest

from dora_openarm_local_policy_server.session import Session
from openarm_policy_runtime import Backend, ModelSession, Observation, Prediction


class Model(Backend):
    def __init__(self):
        self.resets = []
        self.prefill = self.restart = False
        self.execution = None

    def reset(self, reason):
        self.resets.append(reason)
        return False

    def predict(self, observation):
        self.observation = observation
        return Prediction(
            None if self.prefill else np.repeat(np.arange(8)[:, None], 16, axis=1),
            execution=self.execution,
            restart_execution=self.restart,
        )


def run(caller, model, generation=1, *, prompt="task", trial="one", plan=None):
    metadata = {
        "timestamp": 100,
        "inference_trial_id": trial,
        "episode_attempt_id": str(generation),
    }
    request = caller.prepare(generation, metadata, prompt, plan)
    obs = Observation(100, np.zeros((1, 16)), {}, prompt, request["metadata"].copy())
    obs.execution_plan = request["execution_plan"]
    result = ModelSession(model).handle_observation(obs, control=request)
    return request, result


def test_full_result_and_first_window_not_consumed_until_sent():
    caller, model = Session(action_window_start=3, action_window_size=2), Model()
    request, result = run(caller, model)
    assert len(result["positions"]) == 8 and "action_window_start" not in result
    action = caller.complete(request, result)
    assert caller.first and action["positions"][0][0] == 0
    assert action["metadata"]["chunk_id"] == request["chunk_id"]
    caller.sent(action)
    request, result = run(caller, model)
    action = caller.complete(request, result)
    assert action["positions"][0][0] == 3 and len(action["positions"]) == 2
    assert model.resets == ["request"]


def test_prefill_and_discarded_inflight_result_preserve_first_window():
    caller, model = Session(action_window_start=3), Model()
    model.prefill = True
    request, result = run(caller, model)
    caller.sent(caller.complete(request, result))
    assert caller.first and not caller.reset_pending
    model.prefill = False
    old_request, old_result = run(caller, model)
    # Transport rejects this old result rather than handing it to complete().
    request, result = run(caller, model, 2)
    action = caller.complete(request, result)
    assert action["metadata"]["reset"] and action["positions"][0][0] == 0
    assert action["metadata"]["chunk_id"] != old_request["chunk_id"]
    assert len(old_result["positions"]) == 8


def test_cache_resets_and_task_restart_are_distinct():
    caller, model = Session(action_window_start=3, action_window_size=2), Model()
    request, result = run(caller, model)
    caller.sent(caller.complete(request, result))
    request, result = run(caller, model, prompt="changed")
    assert request["reset_reason"] == "prompt" and not request["restart_execution"]
    caller.sent(caller.complete(request, result))
    caller.last_end -= 5
    request, result = run(caller, model, prompt="changed")
    assert request["reset_reason"] == "idle"
    caller.sent(caller.complete(request, result))
    request, result = run(caller, model, prompt="changed", trial="two", plan={"chunk_id": "old"})
    assert request["reset_reason"] == "trial" and request["restart_execution"]
    assert request["execution_plan"] is None
    assert caller.complete(request, result)["positions"][0][0] == 0


def test_rtc_handoff_uses_model_offset_once_and_requires_adopted_plan():
    caller, model = Session(action_window_start=3, action_window_size=2), Model()
    model.execution = {
        "based_on_chunk_id": "",
        "action_window_start": 0,
        "action_origin_timestamp_ns": 0,
        "takeover_timestamp_ns": 0,
    }
    request, result = run(caller, model)
    caller.sent(caller.complete(request, result))
    assert caller.requires_plan(1, None)
    assert caller.requires_plan(1, {"positions": []})
    assert not caller.requires_plan(2, None)
    plan = {"chunk_id": request["chunk_id"], "positions": [[1]]}
    assert not caller.requires_plan(1, plan)
    model.execution = {
        "based_on_chunk_id": plan["chunk_id"],
        "action_window_start": 4,
        "action_origin_timestamp_ns": 100,
        "takeover_timestamp_ns": 200,
    }
    request, result = run(caller, model, plan=plan)
    action = caller.complete(request, result)
    assert action["positions"][0][0] == 4 and len(action["positions"]) == 2
    assert action["metadata"]["takeover_timestamp_ns"] == 200


def test_restart_during_prefill_and_caller_log(tmp_path):
    log = tmp_path / "chunks.jsonl"
    caller, model = Session(action_window_start=3, chunk_log_path=log), Model()
    request, result = run(caller, model)
    caller.sent(caller.complete(request, result))
    model.prefill = model.restart = True
    request, result = run(caller, model)
    caller.sent(caller.complete(request, result))
    assert caller.first
    model.prefill = model.restart = False
    request, result = run(caller, model)
    assert request["restart_execution"]
    action = caller.complete(request, result)
    caller.sent(action)
    caller.close()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[-1]["chunk_id"] == request["chunk_id"]
    assert len(records[-1]["full_positions"]) == 8
    assert records[-1]["selected_positions"][0][0] == 0


@pytest.mark.parametrize("start,size", [(-1, 2), (1, 0)])
def test_invalid_window(start, size):
    with pytest.raises(ValueError, match="window"):
        Session(action_window_start=start, action_window_size=size)


def test_yaml_variable_values_are_converted_once():
    caller = Session(
        action_window_start="3",
        action_window_size="18",
        reset_gap="1.0",
        timing_log_every="0",
        chunk_log_queue_size="0",
    )
    assert (caller.window_start, caller.window_size, caller.reset_gap, caller.timing_log_every) == (
        3,
        18,
        1.0,
        0,
    )
