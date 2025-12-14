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

"""Alternative RobotClient implementation.

This version keeps the original async-inference protocol behavior but replaces
`Queue`-introspection + locks with a single-owner ordered action schedule.

Key idea:
- The action-receiver thread only enqueues *incoming action chunks* into a
  thread-safe queue.
- The control-loop thread is the single owner of the ordered action schedule
  (SortedDict-like via `bisect` + `dict`) and performs all merges and pops.

Example command:
```shell
python src/lerobot/async_inference/robot_client_alternative.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

from __future__ import annotations

import bisect
import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class _ActionSchedule:
    """A small SortedDict-like structure keyed by timestep.

    Implementation:
    - `self._keys`: sorted list of timesteps
    - `self._items`: mapping timestep -> TimedAction

    The schedule is intended to be single-thread-owned.
    """

    def __init__(self) -> None:
        self._keys: list[int] = []
        self._items: dict[int, TimedAction] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def empty(self) -> bool:
        return len(self._keys) == 0

    def peek_range(self) -> tuple[int, int] | None:
        if not self._keys:
            return None
        return self._keys[0], self._keys[-1]

    def merge(
        self,
        incoming: list[TimedAction],
        *,
        latest_action: int,
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None,
        max_size: int | None,
    ) -> dict[str, int]:
        """Merge incoming actions into the schedule.

        - Drops stale actions: timestep <= latest_action
        - Aggregates overlaps on identical timestep
        - Preserves existing non-overlapping scheduled actions
        """

        if aggregate_fn is None:

            def aggregate_fn(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
                return x2

        stats = {
            "inserted": 0,
            "aggregated": 0,
            "stale_dropped": 0,
        }

        for act in incoming:
            t = act.get_timestep()
            if t <= latest_action:
                stats["stale_dropped"] += 1
                continue

            if t in self._items:
                old = self._items[t]
                merged_action = aggregate_fn(old.get_action(), act.get_action())
                # Use the newer timestamp (act) for bookkeeping/logging
                self._items[t] = TimedAction(timestamp=act.get_timestamp(), timestep=t, action=merged_action)
                stats["aggregated"] += 1
                continue

            self._items[t] = act
            bisect.insort(self._keys, t)
            stats["inserted"] += 1

        if max_size is not None and len(self._keys) > max_size:
            # Keep the earliest actions (closest in time) and drop far-future ones.
            while len(self._keys) > max_size:
                t_drop = self._keys.pop()
                self._items.pop(t_drop, None)

        return stats

    def pop_next(self, *, latest_action: int) -> TimedAction | None:
        """Pop the earliest valid action (strictly newer than latest_action)."""
        while self._keys:
            t = self._keys[0]
            if t <= latest_action:
                # Safety prune: should be rare but keeps invariants clean.
                self._keys.pop(0)
                self._items.pop(t, None)
                continue

            self._keys.pop(0)
            return self._items.pop(t)

        return None
class RobotClientAlternative:
    prefix = "robot_client_alternative"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClientAlternative with unified configuration."""
        self.config = config

        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )

        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Protocol state (single-owner: control-loop thread)
        self.latest_action = -1
        self.action_chunk_size = max(1, config.actions_per_chunk)
        self._chunk_size_threshold = config.chunk_size_threshold

        self._schedule = _ActionSchedule()
        self.action_queue_size: list[int] = []

        # Receiver -> control-loop message passing
        # Bounded queue to prevent backpressure when the control loop slows down.
        # Overflow policy: drop oldest chunks and keep the newest (best responsiveness).
        self._incoming_action_chunks: Queue[list[TimedAction]] = Queue(maxsize=10)

        # Keep the same synchronized start behavior as the reference implementation
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # must_go logic (single-owner: control-loop thread)
        # Match original behavior: first empty-queue observation is must_go.
        self._must_go_on_empty = True

        self.logger.info("Robot connected and ready")

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        """Start the robot client and connect to the policy server."""
        try:
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()
            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self) -> None:
        """Stop the robot client."""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(self, obs: TimedObservation) -> bool:
        """Send observation to the policy server."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClientAlternative.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            self.logger.debug(f"Sent observation #{obs.get_timestep()} | ")
            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_schedule(self) -> tuple[int, tuple[int, int] | None]:
        size = len(self._schedule)
        rng = self._schedule.peek_range()
        self.logger.debug(f"Schedule size: {size}, timestep range: {rng}")
        return size, rng

    def _drain_incoming_action_chunks(self, verbose: bool = False) -> None:
        """Drain receiver messages and merge them into the schedule.

        This must be called only from the control-loop thread.
        """
        drained_any = False
        while True:
            try:
                chunk = self._incoming_action_chunks.get_nowait()

                # Quick staleness check to avoid trying to merge fully stale chunks.
                # IMPORTANT: compare timestep (int) against latest_action (int), not timestamp (float seconds).
                if not chunk:
                    continue
                chunk_max_timestep = max(a.get_timestep() for a in chunk)
                if chunk_max_timestep <= self.latest_action:
                    continue
            except Empty:
                break

            drained_any = True
            self.action_chunk_size = max(self.action_chunk_size, len(chunk) or 1)

            if verbose:
                incoming_timesteps = [a.get_timestep() for a in chunk]
                if incoming_timesteps:
                    self.logger.info(
                        "Received action chunk | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Chunk size: {len(incoming_timesteps)}"
                    )

            _ = self._schedule.merge(
                chunk,
                latest_action=self.latest_action,
                aggregate_fn=self.config.aggregate_fn,
                max_size=2 * self.action_chunk_size,
            )

        if drained_any:
            # After receiving any actions, the next time the schedule empties,
            # the next observation must be forced through server processing.
            self._must_go_on_empty = True

    def receive_actions(self, verbose: bool = False) -> None:
        """Receive actions from the policy server and enqueue chunks for the control loop."""
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                timed_actions: list[TimedAction] = pickle.loads(actions_chunk.data)  # nosec
                if not timed_actions:
                    continue

                # Only message-passing here (no shared state mutation).
                # Keep-latest bounding: if the queue is full, drop the oldest chunk and enqueue the newest.
                try:
                    self._incoming_action_chunks.put_nowait(timed_actions)
                except Full:
                    with suppress(Empty):
                        _ = self._incoming_action_chunks.get_nowait()
                    # If still full (rare), drop this newest chunk.
                    with suppress(Full):
                        self._incoming_action_chunks.put_nowait(timed_actions)

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self) -> bool:
        return not self._schedule.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        """Pop and perform the next scheduled action."""
        self.action_queue_size.append(len(self._schedule))

        timed_action = self._schedule.pop_next(latest_action=self.latest_action)
        if timed_action is None:
            return None

        performed_action = self.robot.send_action(self._action_tensor_to_action_dict(timed_action.get_action()))
        self.latest_action = timed_action.get_timestep()

        if verbose:
            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Schedule size: {len(self._schedule)}"
            )

        return performed_action

    def _ready_to_send_observation(self) -> bool:
        schedule_len = len(self._schedule)
        chunk_size = self.action_chunk_size
        if chunk_size <= 0:
            return True
        return (schedule_len / chunk_size) <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation | None:
        try:
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(self.latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            schedule_empty = self._schedule.empty()
            observation.must_go = self._must_go_on_empty and schedule_empty

            _ = self.send_observation(observation)

            self.logger.debug(f"SCHEDULE SIZE: {len(self._schedule)} (Must go: {observation.must_go})")

            if observation.must_go:
                # must_go is armed again whenever we receive any new action chunk.
                self._must_go_on_empty = False

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )
                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")
            return None

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation | None, Action | None]:
        """Main control loop: drain chunks, execute actions, and stream observations."""
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        performed_action: Action | None = None
        captured_observation: Observation | None = None

        while self.running:
            control_loop_start = time.perf_counter()

            # (0) Integrate any newly received chunks before deciding what to execute next.
            self._drain_incoming_action_chunks(verbose=verbose)

            # (1) Perform one action if available.
            if self.actions_available():
                performed_action = self.control_loop_action(verbose=verbose)

            # (2) Stream observations when queue level drops below threshold.
            if self._ready_to_send_observation():
                captured_observation = self.control_loop_observation(task, verbose=verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            time.sleep(max(0.0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return captured_observation, performed_action


@draccus.wrap()
def async_client_alternative(cfg: RobotClientConfig) -> None:
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClientAlternative(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join(timeout=2)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client_alternative()
