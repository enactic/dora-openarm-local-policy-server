# Caller-Side Policy Serving

`session.py` owns reset decisions, first-window state, window selection, chunk IDs
and JSONL logging. `main.py` uses a model socket; `dora_runner.py` invokes a backend
directly. Both use the same session. Models return full decoded absolute qpos.

## Scheduling

A reader keeps the latest unsent observation while one inference is in flight.
Wait for `infer_hz` before selecting that observation, not afterwards. Slow models
do not cause catch-up bursts. This is not state-similarity-based skipping.

Start, observation-ID rollback and running-task attempt changes advance a local
generation. Stop/Intervene disable output. Responses from older generations are
discarded before window selection or first-window consumption. Publication and the
generation check share the same lock. An in-flight model call is not cancelled.

The first observation after Start must match its attempt; subsequent attempt
changes can switch tasks without restarting the arms. Executor plans are accepted
only for the current attempt. Feedback does not create a new attempt.

## Windows and Reset

The caller sends explicit `reset_reason` values: request/trial restart execution;
prompt/idle normally only clear model caches. The backend may additionally request
an execution restart. The first effective window starts at 0 and keeps the configured
size. Prefill, errors and stale results never consume it; only successful nonempty
publication does. Subsequent windows use the configured start, or the model's RTC
`execution.action_window_start`. Never add those two starts together.

Window metadata overrides persist in the caller session. Size null means the rest
of the horizon; oversize is clipped. The existing N1.7 final-point fallback remains
for a start beyond the horizon. Interpolation, timed handoff, blend and filtering
remain executor responsibilities.

Results containing `based_on_chunk_id` require adoption/rejection feedback before
the next inference. The wait is established before publication to avoid a fast-ACK
race. Only a matching `sample_chunk_id` releases it. After RTC bootstrap a valid
adopted plan is required; Start/task changes clear the wait. There is no periodic
progress polling or ACK timeout. Missing feedback wiring therefore stalls RTC
submission, not ordinary inference. Model-side `rtc.sent` is no longer needed.

## Transport and Settings

Socket mode writes observations into a SHM ring, with mode and slot count learned
from the model's readiness reply. Default copying uses three slots; N1.7's
`policy.args.borrow-inputs: true` uses five. Pending, in-flight and model-retained
slots are protected. JSON carries descriptors and full prediction responses.
`input_sequence` only correlates the response with the request;
`released_input_sequences` permits reuse of the corresponding retained slots.
These notifications are processed even when the action response is stale. A task
change or Stop alone cannot release input still used by a model worker.

Sequence numbers count writes, not slots or inference calls. Payload capacity is
fixed by the first observation; a borrowed connection uses one ring throughout.
There is no extra ACK channel, completion polling, or fixed-round expiry.
Socket Arrow files remain removed. Native Dora still uses Arrow, optionally borrowed
without a bulk input copy. Its allocation lifetime is managed by Arrow ownership.

`OPENARM_POLICY_SESSION` is JSON with `infer_hz`, `prompt`, `action_window_start`,
`action_window_size`, `reset_gap`, `chunk_log_path`, `chunk_log_queue_size`, and
`timing_log_every`. The workspace launcher constructs it from existing YAML
`policy.args`, removing these execution flags from the model CLI (prompt also
remains for model warmup). `SOCKET`/`--socket` selects the endpoint;
`--infer-hz` can override pacing when running the bridge manually.

`POLICY_START_TIMEOUT_SEC` defaults to 600. `LOCAL_POLICY_TIMING_EVERY` defaults to
20, or 0 to silence transport timing. Loading/ready/error use the same status
output in both modes. Ready follows model warmup; the socket handshake is on the
inference connection. Idle socket health is not polled. There is no reconnect/replay.

## Recording

The caller assigns one opaque chunk ID before each request. Returned full actions,
the selected window, episode identity, observation/model timestamps and model
diagnostics are queued to the common async JSONL writer. Empty prefill has no
published chunk ID. Full actions are never recropped by the model.

CRA saves large latent artifacts model-side with the request's chunk ID; tensors
are not shipped through JSON. Ordinary N1.7/OpenPI diagnostics use `log_data`.
Model generation timestamps are not replaced by bridge receipt times. Logs and
executor feedback describe software plans, not measured motor execution.
