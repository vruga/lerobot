# Async Inference Formal Specification (TLA+-friendly)

This document is a **protocol-focused** specification of LeRobot async inference, aligned with:

- `src/lerobot/async_inference/robot_client_alternative.py`
- `src/lerobot/async_inference/policy_server_alternative.py`

It is intentionally written to be a good starting point for a TLA+ model.

Notes:
- We **ignore the gRPC handshake** and focus on the steady-state async-inference protocol.
- We model server-side observation filtering via an **abstract predicate** `shouldProcess(...)` that is bypassed by `must_go=True`.
- We model client-side overlap aggregation as **Replace** (new action overwrites old action at same timestep).

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
ChunkSizeThreshold ∈ Real, 0 ≤ ChunkSizeThreshold ≤ 1   // “g” in the blog post

IncomingChunkQueueMax ∈ Nat, IncomingChunkQueueMax = 10
ServerObsQueueMax ∈ Nat, ServerObsQueueMax = 1

ScheduleMaxFactor ∈ Nat, ScheduleMaxFactor = 2
PredictedObsTimestepsMax ∈ Nat  // optional bound, e.g. 10_000
```

### Timing model and the ratio c = EnvironmentDt / InferenceTime

Async inference behavior is governed by two time scales:
- `EnvironmentDt`: the control-loop period (robot can execute ~1 action per `EnvironmentDt`)
- `InferenceTime`: the effective latency to obtain a fresh chunk (policy forward pass + network RTT)

Define the dimensionless ratio:

```text
c = EnvironmentDt / InferenceTime
```

Interpretation (matching the blog post):
- `c << 1`: environment evolves faster than inference. The client will **frequently run out of scheduled actions**
  and the protocol **degenerates toward sequential control** (periodic empty schedule), although `must_go` should
  still prevent deadlock.
- `c >= 1`: server can keep up. The client’s schedule should be **nearly always non-empty** and often “nearly full”
  in the sense of hovering around the threshold refill policy.

The protocol cannot “force” `c >= 1`; it can only:
- request new inference **early enough** (via `ChunkSizeThreshold = g`)
- avoid wasting compute and bandwidth (keep-latest buffers; bounded backpressure; stale-chunk rejection)

#### How g (ChunkSizeThreshold) changes request frequency

The client sends a new observation for inference when:

```text
ScheduleLen(action_schedule) / action_chunk_size ≤ g
```

Extreme regimes:
- `g = 0`: request only when empty ⇒ sequential behavior (max idle risk, minimal server load)
- `g = 1`: request every tick ⇒ maximal server load, minimal lag (if server keeps up)

Empirically, `g ≈ 0.5–0.7` is often a good trade-off when observations are not filtered out (or are `must_go`),
as described in the blog post.

### Client State Variables

```text
latest_action_timestep ∈ Int

// Receiver-owned FIFO of incoming action chunks (bounded).
incoming_action_chunks ∈ Seq(ActionChunk)

// Control-loop-owned schedule of executable actions keyed by timestep.
// In TLA+ you might model this as a function [Timestep -> Action] with finite domain.
action_schedule ⊆ (Timestep × Action)

// Monotone estimate of effective chunk size observed so far.
action_chunk_size ∈ Nat

// must_go arming flag: when TRUE, the next observation sent when the schedule is empty must have must_go=TRUE.
must_go_armed ∈ Bool
```

Derived (helper) quantities:

```text
ScheduleTimesteps(action_schedule) = { t : ∃a. (t,a) ∈ action_schedule }
ScheduleLen(action_schedule) = Cardinality(ScheduleTimesteps(action_schedule))

MinScheduleTimestep(action_schedule) = Min(ScheduleTimesteps(action_schedule))   // when non-empty
MaxChunkTimestep(chunk) = Max({ act.timestep : act ∈ chunk })                   // when non-empty
```

### Server State Variables

```text
// Keep-latest observation slot (Queue(maxsize=1)).
obs_slot ∈ TimedObservation ∪ {None}

last_processed_obs ∈ TimedObservation ∪ {None}

// Tracks observation timesteps that were already used to generate an action chunk.
// (Implementation uses a set with an optional eviction bound.)
predicted_obs_timesteps ⊆ Timestep
```

---

## Nodes (client, server)

### Client node (RobotClientAlternative)
- Executes actions at the environment rate.
- Maintains a local **action schedule** keyed by timestep.
- Receives predicted **action chunks** asynchronously and merges them into the schedule.
- Sends observations when the schedule drops below a threshold fraction of the chunk size.
- Uses `must_go` to guarantee progress when the server may filter observations.

### Server node (PolicyServerAlternative)
- Receives observations (streamed via gRPC in implementation; abstracted here).
- Maintains a **keep-latest** observation slot of size 1.
- Optionally filters observations using `shouldProcess(...)`, but always processes `must_go=True`.
- On request, consumes the latest queued observation and returns an action chunk.

---

## Processes

### Client Processes

#### ClientReceiveActionsThread
Owns only `incoming_action_chunks`.
- Continuously receives `ActionChunk`s from the server.
- Enqueues them into `incoming_action_chunks`.
- Enforces bounded backpressure: when full, **drops the oldest** chunk(s) and keeps the newest.

#### ClientControlLoopThread
Owns `action_schedule`, `latest_action_timestep`, `action_chunk_size`, `must_go_armed`.
Every environment tick:
- Drain/merge any queued incoming chunks into the schedule.
- Execute **at most one** action (the smallest timestep > `latest_action_timestep`).
- Potentially send an observation if the schedule level is below threshold.

### Server Processes

#### ServerReceiveObservation
- Accepts an incoming `TimedObservation`.
- If `must_go=True`, overwrites `obs_slot`.
- Else, enqueues it only if `shouldProcess(obs, last_processed_obs, predicted_obs_timesteps)`.
- Enqueue policy is keep-latest: overwrite `obs_slot`.

#### ServerGetActions
- If `obs_slot != None`, consumes it and generates an `ActionChunk`.
- Marks that observation timestep as predicted by adding it to `predicted_obs_timesteps`.
- Returns the chunk to the client.

---

## Initial Conditions

We start in the “main protocol is running” phase.

```text
// Client
latest_action_timestep = -1
incoming_action_chunks = << >>
action_schedule = ∅
action_chunk_size = ActionsPerChunk      // or max(1, ActionsPerChunk)
must_go_armed = TRUE                    // first empty-queue observation is must_go

// Server
obs_slot = None
last_processed_obs = None
predicted_obs_timesteps = ∅
```

---

## Actions (State Transitions)

Below, primed variables (e.g., `x'`) indicate next-state values.

### Client actions

#### C1: EnqueueIncomingChunk(chunk)
Models the receiver thread enqueuing a received chunk with bounded overflow behavior.

Preconditions:
- `chunk` is an `ActionChunk` (possibly empty; empty chunks can be ignored).

Effects:
- If `Len(incoming_action_chunks) < IncomingChunkQueueMax`:
  - `incoming_action_chunks' = Append(incoming_action_chunks, chunk)`
- Else (queue full): drop oldest then append newest:
  - `incoming_action_chunks' = Append(Tail(incoming_action_chunks), chunk)`
- All other client variables unchanged.


#### C2: DrainOneIncomingChunk
Models the control loop draining **one** chunk from the bounded queue and merging it.

Preconditions:
- `Len(incoming_action_chunks) > 0`

Effects:
- Let `chunk = Head(incoming_action_chunks)`
- Let `incoming_action_chunks' = Tail(incoming_action_chunks)`

- If `chunk` is empty: no merge; continue.

- **Fast stale-chunk rejection**:
  - If `MaxChunkTimestep(chunk) ≤ latest_action_timestep` then the chunk is fully stale:
    - `action_schedule' = action_schedule` (no change)
    - `must_go_armed' = must_go_armed` (no re-arming)
    - `action_chunk_size' = action_chunk_size` (optional: unchanged)

- Else (chunk contains at least one future timestep):
  - `action_chunk_size' = Max(action_chunk_size, Len(chunk))` (monotone)

  - Merge with **Replace** semantics for overlap:
    - For each `act ∈ chunk`:
      - if `act.timestep > latest_action_timestep` then set:
        - `action_schedule' = (action_schedule \ { (act.timestep, _) }) ∪ { (act.timestep, act.action) }`
      - else drop it (stale action).

  - Optional schedule bound (matches implementation intent of limiting far-future actions):
    - Keep only the earliest timesteps, up to `ScheduleMaxFactor * action_chunk_size'`.

  - Re-arm must-go on next-empty:
    - `must_go_armed' = TRUE`

All other variables unchanged.

(You can model draining-all as repeated application of `DrainOneIncomingChunk` under fairness.)


#### C3: ExecuteNextAction
Models the control loop executing exactly one action per tick.

Preconditions:
- `ScheduleLen(action_schedule) > 0`
- Let `t = MinScheduleTimestep(action_schedule)`; require `t > latest_action_timestep`.

Effects:
- “Execute” is abstract; we only update state:
  - `latest_action_timestep' = t`
  - `action_schedule' = action_schedule \ { (t, _) }`
- All other client variables unchanged.


#### C4: SendObservation
Models the client sending an observation to the server according to the queue-threshold policy.

Preconditions:
- `action_chunk_size > 0`
- `ScheduleLen(action_schedule) / action_chunk_size ≤ ChunkSizeThreshold`

Effects:
- Construct `obs`:
  - `obs.timestep = Max(latest_action_timestep, 0)`
  - `obs.must_go = must_go_armed ∧ (ScheduleLen(action_schedule) = 0)`
  - `obs.payload` is unconstrained (current robot observation)

- If `obs.must_go = TRUE`:
  - `must_go_armed' = FALSE`
  else:
  - `must_go_armed' = must_go_armed`

- Deliver observation to server via server action `S1: ReceiveObservation(obs)` (modeled separately).

All other client variables unchanged.


### Server actions

#### S1: ReceiveObservation(obs)
Models server-side enqueue with keep-latest + filtering.

Preconditions:
- `obs` is a `TimedObservation`

Effects:
- If `obs.must_go = TRUE`:
  - `obs_slot' = obs`
- Else if `shouldProcess(obs, last_processed_obs, predicted_obs_timesteps)`:
  - `obs_slot' = obs`
- Else:
  - `obs_slot' = obs_slot` (filtered)

All other server variables unchanged.


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
  - (Optional eviction to keep set bounded; not required for safety properties.)

- Produce an action chunk:
  - `chunk = << act_0, act_1, ..., act_(H-1) >>` with `H = ActionsPerChunk`
  - For each `i ∈ 0..H-1`:
    - `act_i.timestep = obs.timestep + i`
    - `act_i.action` unconstrained

- Deliver chunk to client via `C1: EnqueueIncomingChunk(chunk)`.

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

(Implies from S1 + removing executed timestep from schedule.)

### S4: Bounded incoming chunk backlog (client-side backpressure)

```text
Always: Len(incoming_action_chunks) ≤ IncomingChunkQueueMax
```

### S5: Server observation buffer is size 1 (keep-latest)

```text
Always: obs_slot ∈ TimedObservation ∪ {None}
```

### S6: Schedule contains only future timesteps

```text
Always: ∀(t,a) ∈ action_schedule: t > latest_action_timestep
```

(When modeling Replace aggregation, ensure merge drops stale actions.)

### S7: must_go is only sent when schedule is empty

```text
Always: if client sends obs with must_go=TRUE, then ScheduleLen(action_schedule)=0 at send time
```

---

## Liveness (Good things should always eventually happen)

These properties require standard fairness assumptions (e.g., weak fairness for enabled actions, network eventually delivers messages).

### L1: If actions are available, actions are eventually executed

```text
Always: (ScheduleLen(action_schedule) > 0)  ~>  ExecuteNextAction occurs
```

### L2: Threshold-driven replanning eventually sends observations
If the schedule stays below the threshold long enough, the client eventually sends an observation.

```text
Always: (ScheduleLen(action_schedule) / action_chunk_size ≤ ChunkSizeThreshold)  ~>  SendObservation occurs
```

### L3: must_go breaks server-side filtering deadlocks
If the client runs out of actions and `must_go_armed` is true, it eventually sends a must-go observation.

```text
Always: (must_go_armed ∧ ScheduleLen(action_schedule)=0)  ~>  (SendObservation with must_go=TRUE)
```

### L4: must_go observations are eventually processed
Under fairness/network delivery, a must-go observation eventually leads to a produced chunk.

```text
Always: (Server receives obs with must_go=TRUE)  ~>  ProduceActionsForLatestObs
```

### L5: Produced chunks eventually reach the client and become executable

```text
Always: (ProduceActionsForLatestObs produces chunk)  ~>  (client incoming_action_chunks grows or schedule updated after draining)
```

### L6: Queue stays non-empty in the c >= 1 regime (conditional liveness)

The “queue is always (nearly) full” claim from the blog post is a **timing-dependent** statement. In a TLA+ model,
this is typically encoded as an *assumption* about service time vs consumption rate, plus the refill policy.

One convenient sufficient assumption is:

```text
Assume: InferenceTime ≤ g * ActionsPerChunk * EnvironmentDt
```

Intuition: when the client triggers replanning at roughly `g * ActionsPerChunk` remaining actions, it has enough
buffered execution time to cover the server’s latency and receive the next chunk before the schedule empties.

Under this assumption (plus fairness/delivery and no permanent filtering of must-go observations), you can aim to
prove a conditional property like:

```text
Always: (system running)  ~>  (ScheduleLen(action_schedule) > 0)
```

When the assumption does not hold (the `c << 1` regime), the best you can guarantee is eventual progress via `must_go`
(see L3–L4), not “schedule never empties”.

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
