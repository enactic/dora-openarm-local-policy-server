# dora-openarm-local-policy-server

OpenArm's caller-side policy control: latest observations, request pacing,
execution boundaries, action windows and asynchronous chunk logging.

- Socket mode uses a separate model process and **SHM-only** observation transport.
- Direct Dora mode runs the same caller session with an in-process backend in
  that model's Python environment. Native Arrow is available on this path.
- Input borrowing is opt-in for supported backends on either transport.
- Neither mode blends or filters commands; those operations remain in the executor.

See [serving and transport](docs/serving.md) for the lifecycle, reset and wire contracts.
In the workspace, use `launch_inference.sh experiment.yaml` with the fixed
`openarm-policy-entry` dataflow node. Install the node together with the local
`packages/openarm-policy-runtime` source package; it is not assumed to be on PyPI.

## Synchronous stop-and-go

Set `policy.args.inference-mode: stop-and-go` in the workspace experiment YAML
(or `inference_mode` in `OPENARM_POLICY_SESSION`). The default is `async`.
Use `action-window-start: 0`, choose `action-window-size`, and disable model RTC.
The caller tags nonempty actions with `inference_mode: stop-and-go` and waits on
the existing executor `execution_plan` input. Socket and direct Dora use the same
feedback rules:

1. `adopted` does not release the pending chunk.
2. `completed` releases only the matching `sample_chunk_id`, active `chunk_id`
   and episode attempt. It is sent after the executor sends both arms' final
   commands, not after measured joint settling.
3. Discard buffered observations and require `metadata.timestamp` strictly after
   `completed_timestamp_ns` before starting the next inference.
4. A matching `rejected` sample permits retry after a new observation newer than
   `feedback_timestamp_ns`. A lifecycle command clears the gate and boundary.

No timeout silently switches back to asynchronous inference. Missing completion
feedback leaves the caller waiting; Stop/Start still resets the session. The
observation timestamp describes observer assembly, not every camera's exposure
time. Intentional execution waits do not trigger `reset-gap` idle resets in this
mode; Start, task and prompt resets still apply. Pacing via `infer-hz` remains an
upper bound. No model/TRT changes are required.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
Copyright 2026 Enactic, Inc.

All participation follows our [Code of Conduct](CODE_OF_CONDUCT.md).
