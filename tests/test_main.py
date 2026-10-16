"""Tests for local policy request generation and reset delivery."""

import json
import mmap

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_local_policy_server.main import (
    _PreparedRequest,
    _new_prepare_state,
    _send_actions,
    _serialize_request,
    _start_episode,
    _update_observation_generation,
    _validate_input_ack,
)
from dora_openarm_local_policy_server.shm_ring import SharedMemoryRingWriter
from dora_openarm_local_policy_server.session import Session


def _observation(*ids):
    return pa.StructArray.from_arrays(
        [pa.array(ids, type=pa.int64())],
        names=["id"],
    )


def _policy_observation():
    position = pa.array(
        [[1.0, 2.0], [3.0, 4.0]],
        type=pa.list_(pa.float32()),
    )
    camera = pa.array(
        [list(range(18)), list(range(18, 36))],
        type=pa.list_(pa.uint8()),
    )
    return pa.StructArray.from_arrays(
        [
            position,
            camera,
            pa.array(["old", "pick up the cup"]),
            pa.array([1, 2], type=pa.int64()),
        ],
        names=["position", "camera_wrist_right", "task_prompt", "id"],
    )


def _prepared_request(generation):
    return _PreparedRequest(
        request={"name": "inference"},
        request_json="",
        event_ns=0,
        ready_ns=0,
        event_period_ms=None,
        prepare_input_ms=0.0,
        json_request_ms=0.0,
        reset=False,
        generation=generation,
        should_log_timing=False,
    )


class _FakeNode:
    def __init__(self):
        self.outputs = []

    def send_output(self, *args):
        self.outputs.append(args)


def test_start_advances_generation_once_before_first_observation():
    """A command start and its first observation belong to one generation."""
    state = _new_prepare_state()

    assert _start_episode(state) == 1
    assert _update_observation_generation(_observation(0, 0), state) == 1

    assert _start_episode(state) == 2
    assert _update_observation_generation(_observation(0, 0), state) == 2


def test_latest_observation_id_decrease_is_reset_fallback():
    """A decreasing latest observation ID starts a fallback generation."""
    state = _new_prepare_state()
    _start_episode(state)

    assert _update_observation_generation(_observation(0, 10), state) == 1
    assert _update_observation_generation(_observation(0, 11), state) == 1
    assert _update_observation_generation(_observation(0, 0), state) == 2


def test_request_reset_is_consumed_only_after_completed_response():
    caller = Session()
    control = caller.prepare(2, {"timestamp":1}, "task")
    prepared = _prepared_request(2)
    _serialize_request(prepared, control)
    assert json.loads(prepared.request_json)["reset_reason"] == "request"
    assert caller.reset_pending
    caller.complete(control, {"positions":[], "prefill":True, "reset_applied":True,
                              "interval":1, "generated_timestamp_ns":2})
    assert not caller.reset_pending and caller.first


def test_empty_response_without_ack_keeps_request_reset_pending():
    caller = Session()
    control = caller.prepare(2, {"timestamp":1}, "task")
    prepared = _prepared_request(2)
    _serialize_request(prepared, control)
    caller.complete(control, {"positions":[], "prefill":True, "reset_applied":False,
                              "interval":1, "generated_timestamp_ns":2})
    assert caller.reset_pending


def test_output_reset_is_consumed_only_after_nonempty_actions():
    node, caller = _FakeNode(), Session()
    control = caller.prepare(1, {"timestamp":1}, "task")
    result = {"positions":[], "prefill":True, "reset_applied":True,
              "interval":1, "generated_timestamp_ns":2}
    empty = caller.complete(control, result)
    if _send_actions(node, empty):
        caller.sent(empty)
    assert caller.first and not node.outputs
    actions = caller.complete(control, {**result, "positions":[[1.0,2.0]], "prefill":False})
    if _send_actions(node, actions):
        caller.sent(actions)
    assert not caller.first and node.outputs[0][2]["reset"]
    assert node.outputs[0][2]["chunk_id"] == control["chunk_id"]


def test_shm_ring_slot_is_released_only_after_matching_ack():
    """A ring slot must not be released for a stale or missing response."""
    prepared = _prepared_request(1)
    prepared.transport = "shm_ring_v1"
    prepared.request["shm"] = {"sequence": 17}

    _validate_input_ack(prepared, {"input_sequence": 17})

    with pytest.raises(RuntimeError, match="expected 17"):
        _validate_input_ack(prepared, {"input_sequence": 16})
    with pytest.raises(RuntimeError, match="expected 17"):
        _validate_input_ack(prepared, {})


def test_shm_ring_copies_dense_fields_and_protects_live_slots(tmp_path):
    """Ring writes preserve arrays and never reuse protected slots."""
    metadata = {
        "timestamp": 123,
        "camera_wrist_right.height": 2,
        "camera_wrist_right.width": 3,
    }
    observation = _policy_observation()
    writer = SharedMemoryRingWriter(tmp_path, slot_count=3)
    try:
        first = writer.write(observation, metadata)
        second = writer.write(observation, metadata, {first.resource})
        third = writer.write(
            observation,
            metadata,
            {first.resource, second.resource},
        )

        assert {first.resource[2], second.resource[2], third.resource[2]} == {0, 1, 2}
        with pytest.raises(RuntimeError, match="No free"):
            writer.write(
                observation,
                metadata,
                {first.resource, second.resource, third.resource},
            )

        descriptor = first.descriptor
        with open(descriptor["path"], "rb") as fp:
            mapping = mmap.mmap(
                fp.fileno(), descriptor["ring_size"], access=mmap.ACCESS_READ
            )
            try:
                position_field = descriptor["fields"]["position"]
                position = np.ndarray(
                    position_field["shape"],
                    dtype=np.dtype(position_field["dtype"]),
                    buffer=mapping,
                    offset=(
                        descriptor["slot"] * descriptor["slot_size"]
                        + position_field["offset"]
                    ),
                )
                np.testing.assert_array_equal(
                    position,
                    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                )
                assert descriptor["task_prompt"] == "pick up the cup"
            finally:
                mapping.close()
    finally:
        writer.close()
