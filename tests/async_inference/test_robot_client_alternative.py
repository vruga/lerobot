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
"""Unit-tests for the `RobotClientAlternative` schedule/merge logic (pure Python, no gRPC).

We keep the existing two-thread architecture but ensure:
- no action loss when merging overlapping chunks
- strict monotonic action execution (by timestep)
- must_go behavior matches protocol intent

No real robot hardware is accessed (uses Dummy/Mock robot).
"""

from __future__ import annotations

import time

import pytest
import torch

# Skip entire module if grpc is not available (client imports grpc)
pytest.importorskip("grpc")


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30
    actions: list[TimedAction] = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


@pytest.fixture()
def robot_client_alternative():
    """Fresh RobotClientAlternative instance for each test (no threads started)."""
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client_alternative import RobotClientAlternative
    from tests.mocks.mock_robot import MockRobotConfig

    robot_cfg = MockRobotConfig()

    cfg = RobotClientConfig(
        robot=robot_cfg,
        server_address="localhost:9999",  # not used by unit tests
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
    )

    client = RobotClientAlternative(cfg)
    yield client

    if client.robot.is_connected:
        client.stop()


def test_schedule_merge_discards_stale(robot_client_alternative):
    """Merging must drop actions with timestep <= latest_action."""
    c = robot_client_alternative
    c.latest_action = 4

    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7
    c._incoming_action_chunks.put(incoming)
    c._drain_incoming_action_chunks()

    assert c._schedule._keys == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
    ],
)
def test_schedule_merge_preserves_and_aggregates_overlap(robot_client_alternative, weight_old, weight_new):
    """Overlapping timesteps must be aggregated; non-overlap preserved."""
    c = robot_client_alternative

    # Pretend we already executed up to action #4
    c.latest_action = 4

    # Seed schedule with timesteps 5..6 using distinct tensors
    from lerobot.async_inference.helpers import TimedAction

    current = _make_actions(start_ts=time.time(), start_t=5, count=2)
    current = [
        TimedAction(
            action=10.0 * a.get_action(),
            timestep=a.get_timestep(),
            timestamp=a.get_timestamp(),
        )
        for a in current
    ]

    c._incoming_action_chunks.put(current)
    c._drain_incoming_action_chunks()

    # Incoming chunk overlaps on 5..6 and extends to 7
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    c.config.aggregate_fn = lambda x1, x2: weight_old * x1 + weight_new * x2

    c._incoming_action_chunks.put(incoming)
    c._drain_incoming_action_chunks()

    assert c._schedule._keys == [5, 6, 7]

    # Check overlap aggregation for 5 and 6
    torch.testing.assert_close(
        c._schedule._items[5].get_action(),
        weight_old * current[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    torch.testing.assert_close(
        c._schedule._items[6].get_action(),
        weight_old * current[1].get_action() + weight_new * incoming[-2].get_action(),
    )

    # Check non-overlap at 7
    torch.testing.assert_close(c._schedule._items[7].get_action(), incoming[-1].get_action())


def test_pop_next_is_monotonic(robot_client_alternative):
    """Actions should be popped in increasing timestep order."""
    c = robot_client_alternative
    c.latest_action = -1

    # Provide out-of-order input, schedule should order it.
    chunk = _make_actions(start_ts=time.time(), start_t=5, count=3)  # 5,6,7
    chunk = [chunk[2], chunk[0], chunk[1]]  # 7,5,6

    c._incoming_action_chunks.put(chunk)
    c._drain_incoming_action_chunks()

    popped: list[int] = []
    while not c._schedule.empty():
        act = c._schedule.pop_next(latest_action=c.latest_action)
        assert act is not None
        popped.append(act.get_timestep())
        c.latest_action = act.get_timestep()

    assert popped == [5, 6, 7]


@pytest.mark.parametrize(
    "chunk_size, schedule_len, expected",
    [
        (20, 12, False),  # 12/20 = 0.6 > g=0.5
        (20, 8, True),
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation_ratio(robot_client_alternative, chunk_size, schedule_len, expected):
    """Validate `_ready_to_send_observation` ratio logic."""
    c = robot_client_alternative
    c.action_chunk_size = chunk_size
    c._chunk_size_threshold = 0.5

    # Clear schedule
    c._schedule = c._schedule.__class__()

    actions = _make_actions(start_ts=time.time(), start_t=0, count=schedule_len)
    c._incoming_action_chunks.put(actions)
    c._drain_incoming_action_chunks()

    assert c._ready_to_send_observation() is expected


def test_must_go_arms_on_chunk_and_fires_on_empty(monkeypatch, robot_client_alternative):
    """must_go should be armed by receiving actions and only fire once when schedule becomes empty."""
    c = robot_client_alternative

    captured: list[object] = []

    def _capture_send_observation(obs):
        captured.append(obs)
        return True

    monkeypatch.setattr(c, "send_observation", _capture_send_observation)

    # After draining a chunk, must_go should be armed.
    c._must_go_on_empty = False
    actions = _make_actions(start_ts=time.time(), start_t=0, count=1)
    c._incoming_action_chunks.put(actions)
    c._drain_incoming_action_chunks()
    assert c._must_go_on_empty is True

    # If schedule not empty, must_go should not be set.
    _ = c.control_loop_observation(task="t")
    assert captured[-1].must_go is False

    # Empty the schedule, next observation should be must_go and it clears.
    _ = c._schedule.pop_next(latest_action=-1)
    assert c._schedule.empty() is True

    _ = c.control_loop_observation(task="t")
    assert captured[-1].must_go is True
    assert c._must_go_on_empty is False

    # Subsequent observation while still empty should not be must_go (until new chunk arrives).
    _ = c.control_loop_observation(task="t")
    assert captured[-1].must_go is False
