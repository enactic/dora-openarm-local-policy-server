#!/usr/bin/env python3
#
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

"""An example local policy server."""

import sys
import json
import pyarrow as pa
import socket


def _infer(observation):
    positions = []
    position = observation["position"].values.to_numpy()
    delta = 0.01
    for i in range(10):
        positions.append(position.tolist())
        position += delta  # Move a bit
    return {
        "interval": 1_000_000,  # Action per millisecond
        "positions": positions,
    }


def main():
    """Infer the next actions from observations."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(sys.argv[1])
        sock.listen()
        with sock.accept()[0] as connection:
            with connection.makefile("rw") as io:
                for request_json in io:
                    request = json.loads(request_json)

                    with pa.OSFile(request["data_path"], "rb") as source:
                        with pa.ipc.open_file(source) as reader:
                            observation = reader.get_batch(0).to_struct_array()[0]
                            actions = _infer(observation)
                            io.write(json.dumps(actions) + "\n")
                            io.flush()


if __name__ == "__main__":
    main()
