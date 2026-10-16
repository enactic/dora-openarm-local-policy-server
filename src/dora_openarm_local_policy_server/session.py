"""Caller-owned execution state shared by socket and direct Dora inference."""

import json
import logging
import os
import time
import uuid

import numpy as np
from openarm_policy_runtime import AsyncChunkLogger

LOGGER = logging.getLogger(__name__)


def launch_options():
    """Read caller settings supplied by the workspace launcher."""
    return json.loads(os.getenv("OPENARM_POLICY_SESSION", "{}"))


class Session:
    def __init__(
        self,
        *,
        prompt="",
        action_window_start=0,
        action_window_size=None,
        reset_gap=1.0,
        chunk_log_path=None,
        chunk_log_queue_size=0,
        timing_log_every=1,
        inference_mode="async",
    ):
        if inference_mode not in {"async", "stop-and-go"}:
            raise ValueError("inference_mode must be async or stop-and-go")
        self.stop_and_go = inference_mode == "stop-and-go"
        self.prompt = str(prompt or "")
        self.window_start = int(action_window_start or 0)
        self.window_size = None if action_window_size in (None, "") else int(action_window_size)
        self.reset_gap = None if reset_gap is None else float(reset_gap)
        self.timing_log_every = int(timing_log_every)
        if self.timing_log_every < 0:
            raise ValueError("timing_log_every must be nonnegative")
        self.generation = None
        self.reset_pending = self.first = True
        self.trial_id = self.last_end = None
        self.rtc = False
        self.chunk_index = 0
        self._validate_window(self.window_start, self.window_size)
        self.chunk_log = (
            AsyncChunkLogger(chunk_log_path, int(chunk_log_queue_size)) if chunk_log_path else None
        )

    def _validate_window(self, start, size):
        if start < 0 or (size is not None and size <= 0):
            raise ValueError("Action window requires start >= 0 and size > 0 (or None)")
        if self.stop_and_go and start != 0:
            raise ValueError("stop-and-go requires action window start = 0")

    def requires_plan(self, generation, plan):
        """After RTC bootstrap, infer only against an adopted executor plan."""
        return (
            generation == self.generation
            and self.rtc
            and not self.first
            and (not plan or not plan.get("positions"))
        )

    def prepare(self, generation, metadata, prompt, plan=None):
        """Choose reset semantics before inference; no model or image processing here."""
        if generation != self.generation:
            self.generation = generation
            self.reset_pending = self.first = True
            self.last_end = self.trial_id = None
            self.rtc = False
        raw_start = metadata.get("action_window_start")
        start = self.window_start if raw_start is None else int(raw_start)
        size = metadata.get("action_window_size", self.window_size)
        size = None if size is None else int(size)
        self._validate_window(start, size)
        self.window_start, self.window_size = start, size
        prompt = prompt or self.prompt
        trial = metadata.get("inference_trial_id", self.trial_id)
        reason = None
        if self.reset_pending:
            reason = "request"
        elif self.trial_id is not None and trial != self.trial_id:
            reason = "trial"
            self.first = True
        elif prompt != self.prompt:
            reason = "prompt"
        elif (
            not self.stop_and_go  # Waiting for execution is intentional in synchronous mode.
            and self.last_end is not None
            and self.reset_gap is not None
            and time.monotonic() - self.last_end > self.reset_gap
        ):
            reason = "idle"
        return {
            "metadata": dict(metadata),
            "prompt": prompt,
            "reset_reason": reason,
            "restart_execution": self.first,
            "execution_plan": None if self.first or self.stop_and_go else plan,
            "chunk_id": uuid.uuid4().hex,
            "record_log": self.chunk_log is not None,
        }

    def complete(self, request, result):
        """Select a window only after the caller rejects stale generations."""
        if result.get("error"):
            raise RuntimeError(result["error"])
        self.last_end = time.monotonic()
        self.prompt = request["prompt"]
        self.trial_id = request["metadata"].get("inference_trial_id", self.trial_id)
        if result.get("reset_applied"):
            self.reset_pending = False
        self.first |= bool(result.get("restart_execution"))
        prefill = result.get("prefill", False)
        execution = result.get("execution") or {}
        if self.stop_and_go and "based_on_chunk_id" in execution:
            raise ValueError("stop-and-go requires RTC disabled in the model")
        start = 0 if self.first or self.stop_and_go else int(execution.get("action_window_start", self.window_start))
        full = result["positions"]
        if not prefill:
            start = min(start, len(full) - 1)
        end = None if self.window_size is None else start + self.window_size
        selected = [] if prefill else full[start:end]
        if isinstance(selected, np.ndarray):
            selected = selected.tolist()
        metadata = {
            **request["metadata"],
            **execution,
            "timestamp": result["generated_timestamp_ns"],
            "generated_timestamp_ns": result["generated_timestamp_ns"],
            "observation_timestamp_ns": request["metadata"]["timestamp"],
            "action_window_start": start,
            "action_window_size": len(selected),
            "reset": self.first and not prefill,
        }
        if not prefill:
            metadata["chunk_id"] = request["chunk_id"]
        if self.stop_and_go:
            metadata["inference_mode"] = "stop-and-go"
        response = {
            "positions": selected,
            "interval": result["interval"],
            "metadata": metadata,
            "timing": dict(result.get("timing", {})),
        }
        if "cutoff_hz" in result:
            response["cutoff_hz"] = result["cutoff_hz"]
        self.chunk_index += 1
        if self.chunk_log is not None:
            self.chunk_log.log(
                {
                    **result.get("log_data", {}),
                    "schema": "openarm_policy_chunk_v1",
                    "backend": result.get("backend"),
                    "chunk_index": self.chunk_index,
                    "chunk_id": metadata.get("chunk_id"),
                    "episode_number": metadata.get("episode_number"),
                    "episode_attempt_id": metadata.get("episode_attempt_id"),
                    "observation_timestamp": metadata["observation_timestamp_ns"],
                    "prompt": self.prompt,
                    "full_positions": full,
                    "selected_positions": selected,
                    "action_window": {"start": start, "size": len(selected)},
                    **response,
                }
            )
        if self.timing_log_every and self.chunk_index % self.timing_log_every == 0:
            LOGGER.info(
                "chunk=%s actions=%d policy=%.2fms prefill=%s",
                metadata.get("chunk_id"),
                len(selected),
                response["timing"].get("policy_ms", 0),
                prefill,
            )
        return response

    def sent(self, response):
        """Consume the first window only after a nonempty current result was published."""
        if response["positions"]:
            self.first = self.reset_pending = False
            self.rtc = "based_on_chunk_id" in response["metadata"]

    def close(self):
        if self.chunk_log is not None:
            self.chunk_log.close()
