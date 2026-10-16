"""Output forwards the reset already resolved by the caller session."""

import pytest

from dora_openarm_local_policy_server.main import _send_actions


@pytest.mark.parametrize(
    "reset", [False, True],
)
def test_execution_reset_is_forwarded(reset):
    """Preserve the caller's resolved reset without mutating response metadata."""
    outputs = []

    class Node:
        def send_output(self, *args):
            outputs.append(args)

    response = {
        "positions": [[0.0] * 16],
        "interval": 33_333_333,
        "reset_applied": True,
        "metadata": {"reset": reset, "chunk_id": "one"},
    }
    assert _send_actions(Node(), response)
    assert outputs[0][2]["reset"] is reset
    assert outputs[0][2]["chunk_id"] == "one"
    assert response["metadata"]["reset"] is reset
