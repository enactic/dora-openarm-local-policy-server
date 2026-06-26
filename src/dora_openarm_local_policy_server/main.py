# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to communicate with a local policy server."""

import argparse
import dora
import json
import os
import pyarrow as pa
import socket
import tempfile


def _main_dora(io, shared_dir):
    n_keep_data = 5  # TODO: Customizable?
    data_files = []

    node = dora.Node()
    previous_observation_id = None
    for event in node:
        if event["type"] != "INPUT":
            continue

        # Main process
        def prepare_request():
            nonlocal previous_observation_id
            observation = event["value"]
            observation_id = observation.field("observation_id")[0].as_py()
            # The observation id increments per observation and drops back to a
            # lower value when a new episode starts, so a decrease (or the very
            # first observation) signals a reset to the policy server.
            reset = (
                previous_observation_id is None
                or observation_id < previous_observation_id
            )
            previous_observation_id = observation_id
            data_file = tempfile.NamedTemporaryFile(
                suffix=".arrow", dir=shared_dir, delete_on_close=False
            )
            record_batch = pa.RecordBatch.from_struct_array(observation)
            record_batch = record_batch.append_column(
                "reset", pa.array([reset], type=pa.bool_())
            )
            with pa.output_stream(data_file) as output:
                with pa.ipc.new_file(output, record_batch.schema) as writer:
                    writer.write(record_batch)
            data_files.append(data_file)
            if len(data_files) > n_keep_data:
                data_files.pop(0)
            return reset, {
                "name": "inference",
                "data_path": data_file.name,
                "metadata": event["metadata"],
            }

        # dora-rs node -> Policy server: Inference request
        #   {"name": "inference", "data_path": "/data/path.arrow", ...}
        #
        # "/data/path.arrow" has a record batch:
        #   {
        #     # element len: 8 (7 joints + 1 gripper) * 2 (right + left)
        #     # "arm_right" + "arm_left"
        #     "position": pa.list_(pa.float32()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_wrist_right": pa.list_(pa.uint8()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_wrist_left": pa.list_(pa.uint8()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_head": pa.list_(pa.uint8()),
        #     # element len: 600 (height) * 960 (width) * 3 (RGB)
        #     # element shape: (height, width, color)
        #     "camera_ceiling": pa.list_(pa.uint8()),
        #     }
        #   }
        reset, request = prepare_request()
        io.write(json.dumps(request) + "\n")
        io.flush()

        # Policy server -> dora-rs node: Inferred actions
        #   {
        #     "interval": interval_in_ns,
        #     "positions": [
        #        [...],
        #        ...
        #     ]
        #   }
        #
        #   Arm position: Motor positions
        #     Bimanual: 8 (7 joints + 1 gripper) * 2 (right + left)
        #     Unimanual: 8 (7 joints + 1 gripper)
        response = io.readline()
        if not response:
            break
        actions = json.loads(response)
        if actions["positions"]:
            # "reset" signals that these actions are the first of a new episode,
            # so downstream nodes can drop any state carried over from the
            # previous episode.
            metadata = {"interval": actions["interval"], "reset": reset}
            if "cutoff_hz" in actions:
                metadata["cutoff_hz"] = actions["cutoff_hz"]
            node.send_output(
                "actions",
                pa.array(actions["positions"], type=pa.list_(pa.float32())),
                metadata,
            )


def main():
    """Communicate with a local policy server."""
    parser = argparse.ArgumentParser(
        description="Communicate with a local policy server"
    )
    parser.add_argument(
        "--socket",
        default=os.getenv("SOCKET"),
        help="The local socket to communicate",
        type=str,
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="dora-openarm-local-policy-server", dir="/dev/shm"
    ) as shared_dir:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(args.socket)
            with sock.makefile("rw") as io:
                _main_dora(io, shared_dir)


if __name__ == "__main__":
    main()
