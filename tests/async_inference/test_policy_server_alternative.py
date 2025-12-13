# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Unit-tests for `PolicyServerAlternative` core logic.

These tests target the minimal-fixes areas:
- predicted-timestep bookkeeping is thread-safe
- enqueue semantics are race-safe and keep only the latest observation
- reset drains state instead of swapping the queue object

No real model inference is performed; the policy is stubbed.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import require_package


class MockPolicy:
    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            return {}

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        return self

    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@require_package("grpc")
def policy_server_alternative():
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server_alternative import PolicyServerAlternative

    cfg = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServerAlternative(cfg)

    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    # Minimal processors so `_predict_action_chunk` can run if needed
    server.preprocessor = lambda obs: obs
    server.postprocessor = lambda tensor: tensor

    return server


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


def test_enqueue_keeps_latest_only(policy_server_alternative):
    s = policy_server_alternative

    obs1 = _make_obs(torch.zeros(6), timestep=1, must_go=True)
    obs2 = _make_obs(torch.ones(6), timestep=2, must_go=True)

    assert s._enqueue_observation(obs1) is True
    assert s.observation_queue.qsize() == 1

    # Enqueue another observation; queue should still be size 1 and contain obs2
    assert s._enqueue_observation(obs2) is True
    assert s.observation_queue.qsize() == 1

    queued = s.observation_queue.get_nowait()
    assert queued is obs2


def test_reset_drains_queue_and_predicted(policy_server_alternative):
    s = policy_server_alternative

    # Fill queue
    obs = _make_obs(torch.zeros(6), timestep=7, must_go=True)
    assert s._enqueue_observation(obs) is True
    assert s.observation_queue.qsize() == 1

    # Add predicted timestep
    s._predicted_add(7)
    assert s._predicted_contains(7) is True

    s._reset_server()

    assert s.observation_queue.empty() is True
    assert s._predicted_contains(7) is False


def test_obs_sanity_checks_thread_safe_under_updates(policy_server_alternative):
    s = policy_server_alternative

    prev = _make_obs(torch.zeros(6), timestep=0, must_go=True)
    s.last_processed_obs = prev

    stop = threading.Event()
    errors: list[BaseException] = []

    def writer():
        t = 0
        while not stop.is_set():
            try:
                s._predicted_add(t)
                t += 1
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                stop.set()

    def reader():
        t = 0
        while not stop.is_set():
            try:
                obs = _make_obs(torch.ones(6) * 5, timestep=t)
                _ = s._obs_sanity_checks(obs, prev)
                t += 1
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                stop.set()

    threads = [threading.Thread(target=writer, daemon=True), threading.Thread(target=reader, daemon=True)]
    for th in threads:
        th.start()

    time.sleep(0.2)
    stop.set()

    for th in threads:
        th.join(timeout=1)

    assert errors == []
