#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run Pi0.5 inference on a single LIBERO-10 task and save local videos."""

from __future__ import annotations

import os
import sys

from vlm_eval.cli import main


if __name__ == "__main__":
    exit_code = 1
    try:
        main()
        exit_code = 0
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code == 0:
            # Some simulator / rendering native extensions can crash during
            # interpreter teardown after a successful run. Exit immediately once
            # outputs are flushed to avoid post-run destructor crashes.
            os._exit(0)
