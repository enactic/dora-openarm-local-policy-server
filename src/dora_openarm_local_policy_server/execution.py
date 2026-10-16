"""Execution feedback and fresh-observation gates shared by both transports.

Call these helpers while holding the transport's input condition lock.
"""

import pyarrow as pa


def reset_execution(state):
    """Forget the previous attempt's plan, pending chunk and observation boundary."""
    state.update(execution_plan=None, waiting_chunk=None, observation_after_ns=None)


def fresh_observation(state, metadata):
    """Require an observation assembled strictly after terminal feedback."""
    after = state.get("observation_after_ns")
    return after is None or int(metadata["timestamp"]) > after


def wait_for_execution(state, metadata):
    """Arm the feedback gate before publishing a nonempty action chunk."""
    if state.get("stop_and_go") or "based_on_chunk_id" in metadata:
        state["waiting_chunk"] = metadata["chunk_id"]


def accept_execution_plan(state, event):
    """Only a matching terminal feedback releases a synchronous chunk."""
    attempt = event["metadata"].get("episode_attempt_id")
    if not state["active"] or attempt != state["attempt_id"]:
        return
    value = event["value"]
    plan = value[0].as_py() if len(value) else None
    state["execution_plan"] = plan
    waiting = state.get("waiting_chunk")
    if not plan or waiting is None or plan.get("sample_chunk_id") != waiting:
        return
    if state.get("stop_and_go"):
        status = plan.get("execution_status")
        if status not in {"completed", "rejected"}:
            return
        if status == "completed" and plan.get("chunk_id") != waiting:
            return
        field = "completed_timestamp_ns" if status == "completed" else "feedback_timestamp_ns"
        if plan.get(field) is None:
            return  # Never reuse a buffered observation without a completion boundary.
        boundary = value[0][field].cast(pa.int64()).as_py()
        state["observation_after_ns"] = boundary
        state["latest"] = None
    state["waiting_chunk"] = None
