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

"""PolicyServer alternative (minimal fixes).

This file mirrors `policy_server.py` but applies small concurrency/robustness fixes
while keeping the same architecture and explicit locking strategy:
- Keep gRPC unary endpoints (Ready, SendObservations, GetActions)
- Keep threadpool concurrency
- Keep observation handoff via Queue(maxsize=1)
- Keep an explicit lock for predicted-timestep bookkeeping

Fixes vs `policy_server.py`:
- `_predicted_timesteps_lock` now actually protects read + write usage
- observation enqueue is race-safe without relying on `Queue.full()` snapshots
- reset drains the existing queue instead of swapping the Queue instance
- optional cap on predicted timesteps to avoid unbounded growth

Example:
```shell
python -m lerobot.async_inference.policy_server_alternative \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

from __future__ import annotations

import logging
import pickle  # nosec
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


class PolicyServerAlternative(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server_alternative"
    logger = get_logger(prefix)

    # Minimal, non-breaking bound to avoid unbounded growth in long-running sessions.
    # This only affects very old timesteps and does not change short-horizon behavior.
    _PREDICTED_TIMESTEPS_MAX = 10_000

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        self.fps_tracker = FPSTracker(target_fps=config.fps)

        # Keep a stable queue object; reset drains it.
        self.observation_queue: Queue[TimedObservation] = Queue(maxsize=1)

        # Predicted-timestep bookkeeping (explicit lock strategy, but now correct).
        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps: set[int] = set()
        self._predicted_timesteps_order: deque[int] = deque()

        self.last_processed_obs: TimedObservation | None = None

        # Attributes will be set by SendPolicyInstructions
        self.device: str | None = None
        self.policy_type: str | None = None
        self.lerobot_features: dict[str, Any] | None = None
        self.actions_per_chunk: int | None = None
        self.policy: Any | None = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _drain_observation_queue(self) -> None:
        while True:
            try:
                _ = self.observation_queue.get_nowait()
            except Empty:
                break

    def _predicted_contains(self, timestep: int) -> bool:
        with self._predicted_timesteps_lock:
            return timestep in self._predicted_timesteps

    def _predicted_add(self, timestep: int) -> None:
        with self._predicted_timesteps_lock:
            if timestep in self._predicted_timesteps:
                return

            self._predicted_timesteps.add(timestep)
            self._predicted_timesteps_order.append(timestep)

            # Cap size by evicting the oldest timesteps.
            while len(self._predicted_timesteps_order) > self._PREDICTED_TIMESTEPS_MAX:
                old = self._predicted_timesteps_order.popleft()
                self._predicted_timesteps.discard(old)

    def _predicted_clear(self) -> None:
        with self._predicted_timesteps_lock:
            self._predicted_timesteps.clear()
            self._predicted_timesteps_order.clear()

    def _reset_server(self) -> None:
        """Flush server state when a new client connects."""
        self.shutdown_event.set()

        # Drain queue instead of swapping the queue object.
        self._drain_observation_queue()

        self._predicted_clear()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client."""
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()
        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client."""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, self.logger)
        timed_observation: TimedObservation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)
        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )
        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(timed_observation):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Return action chunks for the latest available observation."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})")

            self._predicted_add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )
            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(max(0.0, self.config.inference_latency - max(0.0, time.perf_counter() - getactions_starts)))
            return actions

        except Empty:
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in GetActions: {e}")
            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy."""
        if self._predicted_contains(obs.get_timestep()):
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        if observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        return True

    def _put_latest_observation(self, obs: TimedObservation) -> None:
        """Race-safe maxsize=1 enqueue that keeps the latest observation."""
        while True:
            try:
                self.observation_queue.put_nowait(obs)
                return
            except Full:
                # Evict the currently queued obs (if any) and retry.
                try:
                    _ = self.observation_queue.get_nowait()
                except Empty:
                    # Another consumer drained it; retry put.
                    continue

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it should be processed."""
        if obs.must_go or self.last_processed_obs is None or self._obs_sanity_checks(obs, self.last_processed_obs):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}")
            self._put_latest_observation(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference

        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        processed_actions = []
        for i in range(chunk_size):
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)

        action_chunk = self._time_action_chunk(observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep())
        postprocess_stops = time.perf_counter()

        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )
        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self) -> None:
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve_alternative(cfg: PolicyServerConfig) -> None:
    logging.info(pformat(asdict(cfg)))

    policy_server = PolicyServerAlternative(cfg)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServerAlternative started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve_alternative()
