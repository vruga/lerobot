# Old Async Inference Formal Specification (TLA+-friendly)

This document is a **protocol-focused** specification of the original LeRobot async inference implementation, aligned with:

- `src/lerobot/async_inference/robot_client.py`
- `src/lerobot/async_inference/policy_server.py`

It is intentionally written to be a good starting point for a TLA+ model.

Notes:
- We **ignore the gRPC handshake** and focus on the steady-state async-inference protocol.
- We model server-side observation filtering via an **abstract predicate** `shouldProcess(...)` that is bypassed by `must_go=True`.
- We model client-side overlap aggregation as **Replace** (new action overwrites old action at same timestep).
- Key difference from alternative: **single shared Queue** with atomic swap vs message-passing architecture.

---

## Architectural Overview

### Key Difference: Single Shared Queue

The old implementation uses a **single shared `Queue`** (`action_queue`) between the receiver thread and control-loop thread, protected by a lock. The receiver thread performs aggregation and atomically swaps the entire queue.

```text
┌─────────────────────────────────────────────────────────────────┐
│                         RobotClient                              │
│  ┌─────────────────────┐      ┌──────────────────────────────┐  │
│  │  Receiver Thread    │      │     Control-Loop Thread      │  │
│  │                     │      │                              │  │
│  │  GetActions() ──────┼──────┼──► action_queue ◄────────────┤  │
│  │        │            │      │        │                     │  │
│  │        ▼            │      │        ▼                     │  │
│  │  _aggregate_action_ │      │  control_loop_action()       │  │
│  │  _queues()          │      │        │                     │  │
│  │        │            │      │        ▼                     │  │
│  │        ▼            │      │  robot.send_action()         │  │
│  │  ATOMIC SWAP ───────┼──────┼──► (new queue)               │  │
│  │                     │      │                              │  │
│  │  must_go.set() ─────┼──────┼──► threading.Event           │  │
│  └─────────────────────┘      └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

Contrast with the alternative implementation's message-passing architecture:
- Alternative: `_incoming_action_chunks` (Queue) → `_schedule` (_ActionSchedule)
- Old: Single `action_queue` (Queue) with atomic replacement

---

## State Variables

### Types (informal)

- `Timestep` is an integer index (`Int`) for environment steps.
- `Timestamp` is a wall-clock time (`Real`) used only for logging/latency (not required for safety).
- `Action` is an uninterpreted value (policy output).
- `ObservationPayload` is an uninterpreted value (robot observation).

Define records:

```text
TimedObservation = {
  timestep: Timestep,
  timestamp: Timestamp,
  payload: ObservationPayload,
  must_go: Bool
}

TimedAction = {
  timestep: Timestep,
  timestamp: Timestamp,
  action: Action
}

ActionChunk = Seq(TimedAction)  // often length ~= ActionsPerChunk
```

### Constants / Parameters

```text
EnvironmentDt ∈ Real, EnvironmentDt > 0
InferenceTime ∈ Real, InferenceTime > 0              // forward-pass + network round-trip (abstracted)
ActionsPerChunk ∈ Nat, ActionsPerChunk > 0
ChunkSizeThreshold ∈ Real, 0 ≤ ChunkSizeThreshold ≤ 1   // "g" in the blog post

ServerObsQueueMax ∈ Nat, ServerObsQueueMax = 1

ScheduleMaxFactor ∈ Nat, ScheduleMaxFactor = 2
PredictedObsTimestepsMax ∈ Nat  // optional bound, e.g. 10_000
```

Note: Unlike the alternative implementation, the old client does NOT have an `IncomingChunkQueueMax` because there is no separate incoming chunks queue.

### Client State Variables

```text
// Protected by latest_action_lock (threading.Lock)
// - Receiver thread reads under lock in _aggregate_action_queues (for each action!)
// - Control loop writes under lock in control_loop_action
// - Control loop reads under lock in control_loop_observation
latest_action_timestep ∈ Int

// Protected by action_queue_lock (threading.Lock)
// Single shared action queue (modeled as a function [Timestep -> Action]).
// - Receiver thread reads queue.queue (internal deque) under lock
// - Receiver thread atomically swaps the entire queue under lock
// - Control loop pops actions under lock via get_nowait()
action_queue ⊆ (Timestep × Action)

// Monotone estimate of effective chunk size observed so far.
action_chunk_size ∈ Nat

// must_go flag: implemented as threading.Event in the old code.
// - Control loop checks is_set() and calls clear() in control_loop_observation
// - Receiver thread calls set() after receiving actions in receive_actions
// When TRUE (set), the next observation sent when the queue is empty must have must_go=TRUE.
must_go_event ∈ Bool
```

**Important lock detail**: In `_aggregate_action_queues`, the receiver reads `latest_action`
under lock **for each action** in the incoming chunk. Between iterations, the control loop
could update `latest_action`. This creates a subtle behavior where actions that become stale
mid-merge are still filtered correctly, but the filtering decision is based on a per-action
snapshot rather than a single consistent view.

Derived (helper) quantities:

```text
QueueTimesteps(action_queue) = { t : ∃a. (t,a) ∈ action_queue }
QueueLen(action_queue) = Cardinality(QueueTimesteps(action_queue))

MinQueueTimestep(action_queue) = Min(QueueTimesteps(action_queue))   // when non-empty
MaxChunkTimestep(chunk) = Max({ act.timestep : act ∈ chunk })        // when non-empty
```

### Server State Variables

```text
// Keep-latest observation slot (Queue(maxsize=1)).
obs_slot ∈ TimedObservation ∪ {None}

last_processed_obs ∈ TimedObservation ∪ {None}

// Tracks observation timesteps that were already used to generate an action chunk.
predicted_obs_timesteps ⊆ Timestep
```

---

## Nodes (client, server)

### Client node (RobotClient)
- Executes actions at the environment rate.
- Maintains a **single shared action queue** accessed by both threads under lock.
- Receives predicted **action chunks** and performs atomic merge-and-swap.
- Sends observations when the queue drops below a threshold fraction of the chunk size.
- Uses `must_go` (threading.Event) to guarantee progress when the server may filter observations.

### Server node (PolicyServer)
- Receives observations (streamed via gRPC in implementation; abstracted here).
- Maintains a **keep-latest** observation slot of size 1.
- Has a **race condition** in enqueue: checks `full()` then `get_nowait()` non-atomically.
- Optionally filters observations using `shouldProcess(...)`, but always processes `must_go=True`.
- On request, consumes the latest queued observation and returns an action chunk.

---

## Processes

### Client Processes

#### ClientReceiveActionsThread
Performs aggregation directly (not just message-passing):
- Continuously receives `ActionChunk`s from the server via `GetActions` RPC.
- Creates a new temporary queue.
- Under lock, reads the current queue's internal deque.
- Builds a merged queue with aggregation (Replace semantics by default).
- Under lock, atomically swaps `action_queue` with the new queue.
- Sets `must_go` event (re-arms for next empty-queue observation).

#### ClientControlLoopThread
Owns execution but shares `action_queue`:
- Performs one action per tick if available (smallest timestep > `latest_action_timestep`).
- Sends observations when queue level is below threshold.
- Checks `must_go.is_set() and action_queue.empty()` to determine `must_go` flag.
- Clears `must_go` event after sending a must-go observation.

### Server Processes

#### ServerReceiveObservation
- Accepts an incoming `TimedObservation`.
- **Race-prone enqueue**:
  - Checks `observation_queue.full()` (non-atomic snapshot)
  - If full, calls `get_nowait()` to evict (may raise Empty if another thread consumed it)
  - Calls `put(obs)` (may block briefly or raise Full in edge cases)
- If `must_go=True`, always enqueues.
- Else, enqueues only if `shouldProcess(obs, last_processed_obs, predicted_obs_timesteps)`.

#### ServerGetActions
- If `obs_slot != None`, consumes it and generates an `ActionChunk`.
- Marks that observation timestep as predicted by adding it to `predicted_obs_timesteps`.
- Returns the chunk to the client.

---

## Initial Conditions

We start in the "main protocol is running" phase.

```text
// Client
latest_action_timestep = -1
action_queue = ∅
action_chunk_size = ActionsPerChunk      // or max(1, ActionsPerChunk)
must_go_event = TRUE                     // must_go.set() called in __init__

// Server
obs_slot = None
last_processed_obs = None
predicted_obs_timesteps = ∅
```

---

## Actions (State Transitions)

Below, primed variables (e.g., `x'`) indicate next-state values.

### Client actions

#### C1: ReceiveAndMergeChunk(chunk)
Models the receiver thread receiving a chunk and performing atomic merge-and-swap.
This combines reception + aggregation + swap in one atomic step (modeling the lock-protected swap).

Preconditions:
- `chunk` is an `ActionChunk` (possibly empty; empty chunks can be ignored).

Effects:
- If `chunk` is empty: no change.

- If `MaxChunkTimestep(chunk) ≤ latest_action_timestep`: chunk is fully stale, no change.

- Else (chunk contains at least one future timestep):
  - `action_chunk_size' = Max(action_chunk_size, Len(chunk))` (monotone)

  - Build new queue with **Replace** semantics for overlap:
    - Start with current `action_queue` contents for timesteps > `latest_action_timestep`
    - For each `act ∈ chunk`:
      - if `act.timestep > latest_action_timestep`:
        - `action_queue' = (action_queue' \ { (act.timestep, _) }) ∪ { (act.timestep, act.action) }`
      - else: drop it (stale action).

  - Optional queue bound (matches implementation behavior):
    - Keep only the earliest timesteps, up to `ScheduleMaxFactor * action_chunk_size'`.

  - Re-arm must-go:
    - `must_go_event' = TRUE`

All other variables unchanged.


#### C2: ExecuteNextAction
Models the control loop executing exactly one action per tick.

Preconditions:
- `QueueLen(action_queue) > 0`
- Let `t = MinQueueTimestep(action_queue)`; require `t > latest_action_timestep`.

Effects:
- "Execute" is abstract; we only update state:
  - `latest_action_timestep' = t`
  - `action_queue' = action_queue \ { (t, _) }`
- All other client variables unchanged.


#### C3: SendObservation
Models the client sending an observation to the server according to the queue-threshold policy.

Preconditions:
- `action_chunk_size > 0`
- `QueueLen(action_queue) / action_chunk_size ≤ ChunkSizeThreshold`

Effects:
- Construct `obs`:
  - `obs.timestep = Max(latest_action_timestep, 0)`
  - `obs.must_go = must_go_event ∧ (QueueLen(action_queue) = 0)`
  - `obs.payload` is unconstrained (current robot observation)

- If `obs.must_go = TRUE`:
  - `must_go_event' = FALSE`   // must_go.clear()
  else:
  - `must_go_event' = must_go_event`

- Deliver observation to server via server action `S1: ReceiveObservation(obs)` (modeled separately).

All other client variables unchanged.


### Server actions

#### S1: ReceiveObservation(obs)
Models server-side enqueue with keep-latest + filtering.
Includes modeling of the race condition in the old implementation.

Preconditions:
- `obs` is a `TimedObservation`

Effects:
- If `obs.must_go = TRUE`:
  - `obs_slot' = obs`
- Else if `shouldProcess(obs, last_processed_obs, predicted_obs_timesteps)`:
  - `obs_slot' = obs`
- Else:
  - `obs_slot' = obs_slot` (filtered)

Note: The race condition in `_enqueue_observation` (check `full()` then `get_nowait()`)
can be modeled as non-determinism, but in practice with a single-slot queue and
typical usage patterns, the observable behavior is equivalent to keep-latest.

All other server variables unchanged.


#### S1_RaceCondition: ReceiveObservationWithRace(obs)
Alternative action modeling the race condition explicitly.

In the old implementation:
```python
if self.observation_queue.full():
    _ = self.observation_queue.get_nowait()  # might raise Empty!
self.observation_queue.put(obs)
```

This can fail if between `full()` returning TRUE and `get_nowait()` executing,
another thread (GetActions) consumed the observation. The `get_nowait()` raises
`Empty`, which is not caught, causing the observation to be lost.

Effects (non-deterministic):
- Either:
  - Normal case: `obs_slot' = obs`
- Or (race case, when `obs_slot` was consumed between check and evict):
  - Observation is lost due to uncaught Empty exception
  - `obs_slot' = obs_slot` (unchanged, obs dropped)

This race is rare in practice because:
1. The queue has maxsize=1
2. GetActions is called synchronously from the receiver thread's perspective
3. The window for the race is very small

For most model checking purposes, the non-race version (S1) is sufficient.


#### S2: ProduceActionsForLatestObs
Models `GetActions`: server consumes latest queued obs and returns a chunk.

Preconditions:
- `obs_slot != None`

Effects:
- Let `obs = obs_slot`.
- Consume it:
  - `obs_slot' = None`
  - `last_processed_obs' = obs`

- Mark observation timestep as predicted:
  - `predicted_obs_timesteps' = predicted_obs_timesteps ∪ { obs.timestep }`

- Produce an action chunk:
  - `chunk = << act_0, act_1, ..., act_(H-1) >>` with `H = ActionsPerChunk`
  - For each `i ∈ 0..H-1`:
    - `act_i.timestep = obs.timestep + i`
    - `act_i.action` unconstrained

- Deliver chunk to client via `C1: ReceiveAndMergeChunk(chunk)`.


#### S2_Timeout
Models the server timing out waiting for an observation.

Preconditions:
- `obs_slot = None`

Effects:
- Server returns empty actions to client (no state change).
- Client receives empty chunk (handled by C1 as no-op).

---

## Safety (Bad things should never happen)

### S1: Monotonic action execution
The client must execute actions in strictly increasing timestep order.

```text
Always: latest_action_timestep' ≥ latest_action_timestep
And whenever ExecuteNextAction occurs: latest_action_timestep' > latest_action_timestep
```

### S2: No stale action execution
The client never executes an action at a timestep ≤ its previously executed timestep.

```text
Always: if ExecuteNextAction chooses timestep t, then t > latest_action_timestep
```

### S3: No duplicate execution
No timestep is executed more than once.

(Implies from S1 + removing executed timestep from queue.)

### S4: Server observation buffer is size 1 (keep-latest)

```text
Always: obs_slot ∈ TimedObservation ∪ {None}
```

### S5: Queue contains only future timesteps

```text
Always: ∀(t,a) ∈ action_queue: t > latest_action_timestep
```

(When modeling Replace aggregation, ensure merge drops stale actions.)

### S6: must_go is only sent when queue is empty

```text
Always: if client sends obs with must_go=TRUE, then QueueLen(action_queue)=0 at send time
```

---

## Liveness (Good things should always eventually happen)

These properties require standard fairness assumptions (e.g., weak fairness for enabled actions, network eventually delivers messages).

### L1: If actions are available, actions are eventually executed

```text
Always: (QueueLen(action_queue) > 0)  ~>  ExecuteNextAction occurs
```

### L2: Threshold-driven replanning eventually sends observations
If the queue stays below the threshold long enough, the client eventually sends an observation.

```text
Always: (QueueLen(action_queue) / action_chunk_size ≤ ChunkSizeThreshold)  ~>  SendObservation occurs
```

### L3: must_go breaks server-side filtering deadlocks
If the client runs out of actions and `must_go_event` is true, it eventually sends a must-go observation.

```text
Always: (must_go_event ∧ QueueLen(action_queue)=0)  ~>  (SendObservation with must_go=TRUE)
```

### L4: must_go observations are eventually processed
Under fairness/network delivery, a must-go observation eventually leads to a produced chunk.

```text
Always: (Server receives obs with must_go=TRUE)  ~>  ProduceActionsForLatestObs
```

### L5: Produced chunks eventually reach the client and update the queue

```text
Always: (ProduceActionsForLatestObs produces chunk)  ~>  (action_queue updated via ReceiveAndMergeChunk)
```

---

## Differences from Alternative Implementation

| Aspect | Old (robot_client.py) | Alternative (robot_client_alternative.py) |
|--------|----------------------|-------------------------------------------|
| Action storage | Single shared `Queue` | Separate `_incoming_action_chunks` + `_schedule` |
| Merge location | Receiver thread | Control-loop thread |
| Queue access | Lock-protected, atomic swap | Single-owner, no contention |
| must_go | `threading.Event` | Boolean flag |
| Backpressure | None (unbounded queue) | Bounded incoming queue (drop oldest) |
| Server enqueue | Race-prone check-then-act | Race-safe retry loop |

---

### Appendix: Abstract server filter

For TLA+, define an abstract predicate:

```text
shouldProcess(obs, lastProcessedObs, predictedObsTimesteps) ∈ Bool
```

Constraints (recommended):
- If `obs.timestep ∈ predictedObsTimesteps`, then `shouldProcess` may be FALSE.
- Otherwise, `shouldProcess` may be TRUE or FALSE (models similarity filter / debouncing).
- **must_go bypass**: if `obs.must_go=TRUE`, server enqueues regardless of `shouldProcess`.
