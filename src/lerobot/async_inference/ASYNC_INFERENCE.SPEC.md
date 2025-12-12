# Async Inference Formal Specification

## Overview

This specification formally defines the state machine for the async inference system in LeRobot, capturing the interaction between the RobotClient and PolicyServer for real-time robotic control.

## State Variables

```
// Server State
server_state ∈ {IDLE, PROCESSING, STREAMING}
observation_queue: Queue[TimedObservation]
last_processed_obs_id: Int
predicted_timesteps: Set[Int]
action_chunks: Map[Int, ActionChunk]
inference_in_progress: Boolean

// Client State  
client_state ∈ {DISCONNECTED, CONNECTED, RUNNING, STOPPED}
action_queue: Queue[TimedAction]
latest_action_timestep: Int
current_observation_id: Int
must_go: Boolean
chunk_size_threshold: Float ∈ [0, 1]
actions_per_chunk: Int > 0

// Shared State
connection: Channel ∈ {CLOSED, OPEN, ERROR}
environment_dt: Float > 0  // Control loop period (e.g., 1/30 sec)
inference_latency: Float > 0  // Time to compute action chunk

// Thread States
observation_sender_alive: Boolean
action_receiver_alive: Boolean
control_loop_alive: Boolean
server_stream_alive: Boolean

// Metrics
total_chunks_sent: Int
total_actions_executed: Int
queue_empty_count: Int
dropped_observations: Int
```

## Initial Conditions

```
// Server Initial State
server_state = IDLE
observation_queue = ∅
last_processed_obs_id = -1
predicted_timesteps = ∅
action_chunks = ∅
inference_in_progress = FALSE

// Client Initial State
client_state = DISCONNECTED
action_queue = ∅
latest_action_timestep = -1
current_observation_id = 0
must_go = FALSE
chunk_size_threshold = 0.5
actions_per_chunk = 20

// Connection Initial State
connection = CLOSED
environment_dt = 1/30  // 30 Hz default
inference_latency = 0.1  // 100ms default

// Thread Initial State
observation_sender_alive = FALSE
action_receiver_alive = FALSE
control_loop_alive = FALSE
server_stream_alive = FALSE

// Metrics Initial State
total_chunks_sent = 0
total_actions_executed = 0
queue_empty_count = 0
dropped_observations = 0
```

## Actions (State Transitions)

### Client Actions

#### Connect
```
Precondition: client_state = DISCONNECTED ∧ connection = CLOSED
Effect: 
  client_state' = CONNECTED
  connection' = OPEN
  observation_sender_alive' = TRUE
  action_receiver_alive' = TRUE
  control_loop_alive' = TRUE
```

#### SendObservation
```
Precondition: client_state ∈ {CONNECTED, RUNNING} ∧ connection = OPEN
Effect:
  current_observation_id' = current_observation_id + 1
  // Server receives observation
  observation_queue' = observation_queue ∪ {new_observation}
  // Trigger must_go if queue is too small
  IF |action_queue| / actions_per_chunk < chunk_size_threshold THEN
    must_go' = TRUE
```

#### ReceiveActionChunk
```
Precondition: client_state = RUNNING ∧ connection = OPEN ∧ action_receiver_alive = TRUE
Effect:
  // Aggregate new actions into queue
  FOR action IN received_chunk:
    IF action.timestep > latest_action_timestep THEN
      action_queue' = aggregate(action_queue, action)
  total_chunks_sent' = total_chunks_sent + 1
```

#### ExecuteAction
```
Precondition: client_state = RUNNING ∧ |action_queue| > 0 ∧ control_loop_alive = TRUE
Effect:
  action = dequeue(action_queue)
  latest_action_timestep' = action.timestep
  total_actions_executed' = total_actions_executed + 1
  IF |action_queue| = 0 THEN
    queue_empty_count' = queue_empty_count + 1
```

#### Disconnect
```
Precondition: client_state ∈ {CONNECTED, RUNNING}
Effect:
  client_state' = DISCONNECTED
  connection' = CLOSED
  observation_sender_alive' = FALSE
  action_receiver_alive' = FALSE
  control_loop_alive' = FALSE
```

### Server Actions

#### ProcessObservation
```
Precondition: server_state ≠ PROCESSING ∧ |observation_queue| > 0 ∧ ¬inference_in_progress
Effect:
  obs = dequeue(observation_queue)
  IF should_process(obs) THEN
    server_state' = PROCESSING
    inference_in_progress' = TRUE
    last_processed_obs_id' = obs.id
  ELSE
    dropped_observations' = dropped_observations + 1

Where should_process(obs) = 
  (obs.id ∉ predicted_timesteps) ∧ 
  (must_go ∨ obs.id > last_processed_obs_id + skip_threshold)
```

#### GenerateActionChunk
```
Precondition: server_state = PROCESSING ∧ inference_in_progress = TRUE
Effect:
  // After inference_latency time
  chunk = compute_actions(last_observation, actions_per_chunk)
  action_chunks' = action_chunks ∪ {(last_processed_obs_id, chunk)}
  predicted_timesteps' = predicted_timesteps ∪ chunk.timesteps
  inference_in_progress' = FALSE
  server_state' = STREAMING
```

#### StreamActions
```
Precondition: server_state = STREAMING ∧ connection = OPEN ∧ server_stream_alive = TRUE
Effect:
  // Send chunk to client
  send(action_chunks[last_processed_obs_id])
  server_state' = IDLE
```

### Error Actions

#### ThreadDeath
```
Precondition: Any thread_alive variable = TRUE
Effect:
  thread_alive' = FALSE
  IF critical_thread THEN
    connection' = ERROR
    client_state' = STOPPED
```

## Safety Properties

> Bad things should never happen

### S1: No Duplicate Action Execution
```
□ ∀ t1, t2 ∈ executed_actions: 
  t1.timestep = t2.timestep ⟹ t1 = t2
```
*The same timestep should never be executed twice*

### S2: Monotonic Action Execution
```
□ ∀ actions a1, a2: 
  (a1 executed before a2) ⟹ (a1.timestep < a2.timestep)
```
*Actions must be executed in strictly increasing timestep order*

### S3: No Action Loss
```
□ ∀ chunk ∈ received_chunks, ∀ action ∈ chunk:
  action.timestep > latest_action_timestep ⟹ 
    ◇ (action ∈ executed_actions ∨ action ∈ action_queue)
```
*Valid actions from received chunks must either be executed or remain in queue*

### S4: Bounded Queue Size
```
□ |action_queue| ≤ 2 * actions_per_chunk
```
*Action queue should not grow unbounded*

### S5: No Observation Buffer Overflow
```
□ |observation_queue| ≤ max_observation_buffer
```
*Observation queue must remain bounded*

### S6: Thread Consistency
```
□ (client_state = RUNNING) ⟹ 
  (observation_sender_alive ∧ action_receiver_alive ∧ control_loop_alive)
```
*All critical threads must be alive when client is running*

## Liveness Properties

> Good things should always eventually happen

### L1: Action Generation Progress
```
□ (|observation_queue| > 0 ∧ server_state = IDLE) ⟹ 
  ◇ (server_state = PROCESSING)
```
*Server must eventually process pending observations*

### L2: Action Delivery
```
□ (chunk generated) ⟹ ◇ (chunk delivered to client)
```
*Generated action chunks must eventually be delivered*

### L3: Action Execution Progress
```
□ (|action_queue| > 0 ∧ client_state = RUNNING) ⟹ 
  ◇ (total_actions_executed' > total_actions_executed)
```
*Queued actions must eventually be executed*

### L4: Queue Refill Under Must-Go
```
□ (must_go = TRUE ∧ connection = OPEN) ⟹ 
  ◇ (|action_queue| ≥ chunk_size_threshold * actions_per_chunk)
```
*When must_go is set, queue must eventually be refilled*

### L5: Periodic Inference
```
□ (client_state = RUNNING) ⟹ 
  ◇≤2*environment_dt (new chunk received)
```
*New chunks must arrive within bounded time during operation*

### L6: No Permanent Starvation
```
□ (client_state = RUNNING) ⟹ 
  (queue_empty_count - last_queue_empty_count < 3 in any 1-second window)
```
*Queue should not repeatedly starve*

### L7: Inference Completion
```
□ (inference_in_progress = TRUE) ⟹ 
  ◇≤2*inference_latency (inference_in_progress = FALSE)
```
*Inference must complete within bounded time*

## Invariants

### I1: Timestep Ordering
```
□ ∀ a ∈ action_queue: a.timestep > latest_action_timestep
```
*All queued actions have timestamps greater than the last executed*

### I2: Connection State Consistency
```
□ (client_state ∈ {RUNNING, CONNECTED}) ⟺ (connection = OPEN)
```
*Client running state implies open connection and vice versa*

### I3: Chunk Size
```
□ ∀ chunk: |chunk.actions| = actions_per_chunk
```
*All chunks contain exactly actions_per_chunk actions*

### I4: Observation ID Monotonicity
```
□ current_observation_id ≥ last_processed_obs_id
```
*Current observation ID never decreases below last processed*

## Fairness Assumptions

### F1: Fair Scheduling
```
WF(ProcessObservation) ∧ WF(ExecuteAction)
```
*Both observation processing and action execution get fair CPU time*

### F2: Network Reliability
```
SF(connection = OPEN ⟹ messages eventually delivered)
```
*Network delivers messages with strong fairness when connection is open*

### F3: Bounded Computation
```
□ inference_latency < 10 * environment_dt
```
*Inference is not arbitrarily slow relative to control rate*

## Known Violations (Current Bugs)

Based on test results, the current implementation violates:

1. **L3 (Action Execution Progress)**: Actions remain in queue and are never executed
2. **L4 (Queue Refill)**: Setting must_go doesn't reliably trigger new chunks
3. **L5 (Periodic Inference)**: Chunks stop arriving after the first one
4. **S3 (No Action Loss)**: Actions are received but disappear without execution
5. **L6 (No Starvation)**: System completely starves under latency

## Verification Goals

To verify this specification:

1. Model check safety properties S1-S6 hold in all reachable states
2. Verify liveness properties L1-L7 under fairness assumptions F1-F3
3. Prove invariants I1-I4 are maintained by all actions
4. Show that fixing known violations restores all properties

## Implementation Requirements

For correct implementation:

1. **Aggregation**: Must preserve all actions with timestep > latest_executed
2. **Concurrency**: Must protect shared state with proper synchronization
3. **Flow Control**: Must trigger inference when queue falls below threshold
4. **Error Recovery**: Must detect and recover from thread failures
5. **Backpressure**: Must handle inference slower than control rate

---

*This specification can be translated to TLA+, Alloy, or other formal verification tools to mathematically prove correctness of the async inference protocol.*
