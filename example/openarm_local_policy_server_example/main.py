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

"""A checkpoint-free full-chunk model service for the SHM caller."""

import sys
import numpy as np
from openarm_policy_runtime import Backend, ModelSession, Prediction, serve


class ExamplePolicy(Backend):
    def predict(self, observation):
        qpos = observation.qpos[observation.current_index()]
        positions = qpos[None, :] + np.arange(10, dtype=np.float32)[:, None] * 0.01
        return Prediction(positions, interval_ns=1_000_000)


def main():
    """Serve complete predictions; the caller selects the execution window."""
    backend = ExamplePolicy()
    serve(sys.argv[1], lambda: ModelSession(backend))


if __name__ == "__main__":
    main()
