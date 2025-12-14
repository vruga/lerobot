---- MODULE OldAsyncInference ----
EXTENDS Integers, Sequences, FiniteSets, TLC

(***************************************************************************
Finite, TLC-friendly model of the OLD async inference protocol in:
  src/lerobot/async_inference/robot_client.py
  src/lerobot/async_inference/policy_server.py

See: src/lerobot/async_inference/OLD_ASYNC_INFERENCE.SPEC.md

Key differences from AsyncInference.tla (alternative implementation):
- NO separate incoming_action_chunks queue
- Single shared action_queue with atomic merge-and-swap
- Receiver thread performs aggregation directly
- must_go is a threading.Event (set/clear semantics)
- Server has race-prone enqueue (optionally modeled)

Key modeling choices:
- Finite domains (MaxTimestep, Actions, Payloads).
- No reals: threshold uses rationals ChunkSizeThresholdNum/Den.
- action_queue is a total function [Timestep -> Actions \cup {NoAction}].
- Network failures: observation and chunk delivery can non-deterministically fail.
- Server timeout: models the obs_queue_timeout returning empty actions.
- Aggregation abstraction: merged actions can be any valid action (not just replace).
***************************************************************************)

CONSTANTS
  MaxTimestep,
  Actions,
  Payloads,
  ActionsPerChunk,
  ScheduleMaxFactor,
  ChunkSizeThresholdNum,
  ChunkSizeThresholdDen,
  \* Network failure modeling: when TRUE, messages can be dropped
  EnableNetworkFailures,
  \* Server timeout modeling: when TRUE, server can timeout waiting for obs
  EnableServerTimeout,
  \* Server race condition: when TRUE, server enqueue can fail due to race
  EnableServerEnqueueRace

(***************************************************************************
Sentinel values
***************************************************************************)
None == [kind |-> "None"]
NoAction == [kind |-> "NoAction"]

Timestep == 0..MaxTimestep

TimedObservation == [timestep: Timestep, payload: Payloads, must_go: BOOLEAN]
TimedAction == [timestep: Timestep, action: Actions]

(***************************************************************************
Bounded sequences (TLC cannot enumerate Seq(S) which is infinite).
***************************************************************************)
SeqOfLen(S, n) == IF n = 0 THEN {<< >>} ELSE [1..n -> S]

SeqUpTo(S, n) == UNION { SeqOfLen(S, m) : m \in 0..n }

ActionChunk == SeqUpTo(TimedAction, ActionsPerChunk)

(***************************************************************************
Queue helpers
***************************************************************************)
QueueSet(queue) == {t \in Timestep : queue[t] # NoAction}
QueueLen(queue) == Cardinality(QueueSet(queue))

MinElem(S) == CHOOSE t \in S : \A u \in S : t <= u
MaxElem(S) == CHOOSE t \in S : \A u \in S : u <= t

MinQueueTimestep(queue) == MinElem(QueueSet(queue))

ChunkTimesteps(chunk) == {chunk[i].timestep : i \in 1..Len(chunk)}
MaxChunkTimestep(chunk) == MaxElem(ChunkTimesteps(chunk))

MaxInt(a, b) == IF a >= b THEN a ELSE b
MinInt(a, b) == IF a <= b THEN a ELSE b

ChunkHasTimestep(chunk, t) == \E i \in 1..Len(chunk) : chunk[i].timestep = t

ChunkActionAt(chunk, t) ==
  CHOOSE a \in Actions : \E i \in 1..Len(chunk) : chunk[i].timestep = t /\ chunk[i].action = a

(***************************************************************************
Merge chunk into queue with Replace semantics.
The old implementation's _aggregate_action_queues uses Replace by default
(new action overwrites old at same timestep).
***************************************************************************)
MergeChunkReplace(queue, chunk, latest) ==
  [t \in Timestep |->
    IF t <= latest
      THEN NoAction
      ELSE IF ChunkHasTimestep(chunk, t)
        THEN ChunkActionAt(chunk, t)
        ELSE queue[t]
  ]

\* Abstract aggregation: when both queue and chunk have an action at t,
\* the result can be ANY action (models weighted_average, etc.)
PossibleMergedQueues(queue, chunk, latest) ==
  { merged \in [Timestep -> Actions \cup {NoAction}] :
      \A t \in Timestep :
        IF t <= latest
          THEN merged[t] = NoAction
          ELSE IF ChunkHasTimestep(chunk, t) /\ queue[t] # NoAction
            \* Both have action at t: aggregation can produce any action
            THEN merged[t] \in Actions
            ELSE IF ChunkHasTimestep(chunk, t)
              THEN merged[t] = ChunkActionAt(chunk, t)
              ELSE merged[t] = queue[t]
  }

\* Default merge uses Replace semantics for backward compatibility
MergeChunk(queue, chunk, latest) == MergeChunkReplace(queue, chunk, latest)

KeepFirstK(queue, latest, k) ==
  {t \in Timestep :
    /\ t > latest
    /\ queue[t] # NoAction
    /\ Cardinality({u \in Timestep : u > latest /\ queue[u] # NoAction /\ u < t}) < k
  }

TrimQueue(queue, latest, k) ==
  [t \in Timestep |->
    IF t <= latest
      THEN NoAction
      ELSE IF t \in KeepFirstK(queue, latest, k)
        THEN queue[t]
        ELSE NoAction
  ]

ShouldProcess(obs, predictedObsTimesteps) == obs.timestep \notin predictedObsTimesteps

(***************************************************************************
Variables

Key difference from AsyncInference.tla:
- NO incoming_action_chunks (receiver merges directly)
- action_queue is the single shared structure
- must_go_event models threading.Event (set/clear)

Lock semantics in the old implementation:
- latest_action is protected by latest_action_lock (threading.Lock)
- Receiver thread reads latest_action under lock in _aggregate_action_queues
- Control loop writes latest_action under lock in control_loop_action
- Control loop reads latest_action under lock in control_loop_observation
- TLA+ models this correctly: each action is atomic, so lock semantics are implicit
***************************************************************************)
VARIABLES
  \* Models self.latest_action protected by self.latest_action_lock
  \* Accessed by both receiver thread (read) and control loop (read/write)
  latest_action_timestep,
  \* Models self.action_queue protected by self.action_queue_lock
  \* Receiver does atomic swap; control loop pops actions
  action_queue,
  action_chunk_size,
  must_go_event,             \* threading.Event: TRUE = set, FALSE = cleared
  did_execute,
  did_produce,
  obs_slot,
  last_processed_obs,
  predicted_obs_timesteps

vars == <<
  latest_action_timestep,
  action_queue,
  action_chunk_size,
  must_go_event,
  did_execute,
  did_produce,
  obs_slot,
  last_processed_obs,
  predicted_obs_timesteps
>>

(***************************************************************************
Init
***************************************************************************)
Init ==
  /\ latest_action_timestep = -1
  /\ action_queue = [t \in Timestep |-> NoAction]
  /\ action_chunk_size = ActionsPerChunk
  /\ must_go_event = TRUE    \* must_go.set() in __init__
  /\ did_execute = FALSE
  /\ did_produce = FALSE
  /\ obs_slot = None
  /\ last_processed_obs = None
  /\ predicted_obs_timesteps = {}

(***************************************************************************
Type correctness (use as a TLC invariant)
***************************************************************************)
TypeOK ==
  /\ latest_action_timestep \in -1..MaxTimestep
  /\ action_queue \in [Timestep -> Actions \cup {NoAction}]
  /\ action_chunk_size \in 1..ActionsPerChunk
  /\ must_go_event \in BOOLEAN
  /\ did_execute \in BOOLEAN
  /\ did_produce \in BOOLEAN
  /\ obs_slot = None \/ obs_slot \in TimedObservation
  /\ last_processed_obs = None \/ last_processed_obs \in TimedObservation
  /\ predicted_obs_timesteps \subseteq Timestep

(***************************************************************************
Client actions
***************************************************************************)

(***************************************************************************
C1_ReceiveAndMergeChunk: Receiver thread gets chunk and atomically merges.

In the old implementation, the receiver thread (_aggregate_action_queues):
1. Calls GetActions() to receive a chunk
2. Creates a new Queue (future_action_queue)
3. Under action_queue_lock, reads current queue's internal deque
4. For EACH action in chunk:
   - Under latest_action_lock, reads latest_action to filter stale actions
   - Aggregates with existing action at same timestep (if any)
5. Under action_queue_lock, atomically swaps action_queue with new queue
6. Calls must_go.set() to re-arm

Lock note: The implementation reads latest_action under lock FOR EACH action
in the chunk. Between iterations, the control loop could update latest_action.
This TLA+ model abstracts this by reading latest_action_timestep once at the
start of the merge, which is sound for safety (any action stale at merge-end
is correctly filtered).

This models the atomic merge-and-swap as a single step.
Network failures can drop the chunk (when enabled).
***************************************************************************)
C1_ReceiveAndMergeChunk ==
  \E chunk \in ActionChunk :
    /\ did_execute' = FALSE
    /\ did_produce' = FALSE
    /\ IF Len(chunk) = 0
         THEN
           /\ UNCHANGED << action_queue, action_chunk_size, must_go_event >>
         ELSE IF MaxChunkTimestep(chunk) <= latest_action_timestep
           THEN
             \* Fully stale chunk, no change
             /\ UNCHANGED << action_queue, action_chunk_size, must_go_event >>
           ELSE
             LET new_chunk_size == MaxInt(action_chunk_size, Len(chunk)) IN
             LET merged == MergeChunk(action_queue, chunk, latest_action_timestep) IN
             LET k == ScheduleMaxFactor * new_chunk_size IN
             LET trimmed == TrimQueue(merged, latest_action_timestep, k) IN
             /\ action_chunk_size' = new_chunk_size
             /\ action_queue' = trimmed
             /\ must_go_event' = TRUE  \* must_go.set() after receiving actions
             /\ Assert(\A t \in Timestep : t <= latest_action_timestep => trimmed[t] = NoAction,
                       "C1 produced queue with past timesteps")
    /\ UNCHANGED <<latest_action_timestep, obs_slot, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
C1_ReceiveWithAggregation: Variant that uses abstract aggregation.
WARNING: This is significantly more expensive for TLC due to non-determinism.
***************************************************************************)
C1_ReceiveWithAggregation ==
  \E chunk \in ActionChunk :
    /\ did_execute' = FALSE
    /\ did_produce' = FALSE
    /\ IF Len(chunk) = 0
         THEN
           /\ UNCHANGED << action_queue, action_chunk_size, must_go_event >>
         ELSE IF MaxChunkTimestep(chunk) <= latest_action_timestep
           THEN
             /\ UNCHANGED << action_queue, action_chunk_size, must_go_event >>
           ELSE
             LET new_chunk_size == MaxInt(action_chunk_size, Len(chunk)) IN
             LET k == ScheduleMaxFactor * new_chunk_size IN
             \E merged \in PossibleMergedQueues(action_queue, chunk, latest_action_timestep) :
               LET trimmed == TrimQueue(merged, latest_action_timestep, k) IN
               /\ action_chunk_size' = new_chunk_size
               /\ action_queue' = trimmed
               /\ must_go_event' = TRUE
               /\ Assert(\A t \in Timestep : t <= latest_action_timestep => trimmed[t] = NoAction,
                         "C1 produced queue with past timesteps")
    /\ UNCHANGED <<latest_action_timestep, obs_slot, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
C2_ExecuteNextAction: Control loop pops and executes one action.

In the old implementation (control_loop_action):
1. Under action_queue_lock, pops action from queue (get_nowait)
2. Sends action to robot
3. Under latest_action_lock, writes latest_action = timed_action.get_timestep()

Lock note: The write to latest_action is protected by latest_action_lock,
ensuring the receiver thread sees a consistent value when filtering stale
actions.
***************************************************************************)
C2_ExecuteNextAction ==
  /\ QueueLen(action_queue) > 0
  /\ LET t == MinQueueTimestep(action_queue) IN
     /\ t > latest_action_timestep
     /\ latest_action_timestep' = t
     /\ action_queue' = [action_queue EXCEPT ![t] = NoAction]
     /\ did_execute' = TRUE
     /\ did_produce' = FALSE
     /\ UNCHANGED <<action_chunk_size, must_go_event, obs_slot, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
C3_SendObservation: Client sends observation to server.

In the old implementation (control_loop_observation):
1. Captures robot observation
2. Under latest_action_lock, reads latest_action to set observation.timestep
3. Checks must_go.is_set() AND action_queue.empty() to set observation.must_go
4. Sends observation to server
5. If must_go was set: calls must_go.clear()

Lock note: The read of latest_action is protected by latest_action_lock.

Network failure modeling: When EnableNetworkFailures is TRUE, the observation
may be dropped during network delivery.
***************************************************************************)
C3_SendObservation ==
  /\ action_chunk_size > 0
  /\ QueueLen(action_queue) * ChunkSizeThresholdDen <= ChunkSizeThresholdNum * action_chunk_size
  /\ LET sendT == IF latest_action_timestep < 0 THEN 0 ELSE latest_action_timestep IN
     \E payload \in Payloads :
       LET sendMustGo == must_go_event /\ QueueLen(action_queue) = 0 IN
       LET obs == [timestep |-> sendT, payload |-> payload, must_go |-> sendMustGo] IN
       /\ must_go_event' = IF sendMustGo THEN FALSE ELSE must_go_event  \* must_go.clear() if sent
       /\ \/ \* Success case: observation delivered
             /\ obs_slot' =
                  IF obs.must_go
                    THEN obs
                    ELSE IF ShouldProcess(obs, predicted_obs_timesteps)
                      THEN obs
                      ELSE obs_slot
          \/ \* Network failure case: observation dropped (only when enabled)
             /\ EnableNetworkFailures
             /\ obs_slot' = obs_slot
       /\ did_execute' = FALSE
       /\ did_produce' = FALSE
       /\ Assert(~obs.must_go \/ QueueLen(action_queue) = 0,
                "must_go sent when queue was non-empty")
       /\ UNCHANGED <<latest_action_timestep, action_queue, action_chunk_size, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
Server actions
***************************************************************************)

(***************************************************************************
S1_ProduceActionsForLatestObs: Server processes observation and produces chunk.

Network failure modeling: When EnableNetworkFailures is TRUE, the produced
action chunk may be dropped during network delivery (client never receives it).

In the old implementation, the chunk goes directly to the receiver thread which
then merges it. We model this as the chunk being available for C1_ReceiveAndMergeChunk.
***************************************************************************)
S1_ProduceActionsForLatestObs ==
  /\ obs_slot # None
  /\ LET obs == obs_slot IN
     LET maxLen == MinInt(ActionsPerChunk, MaxTimestep - obs.timestep + 1) IN
     /\ maxLen >= 1
     /\ \E chunk \in ActionChunk :
          /\ Len(chunk) = maxLen
          /\ \A i \in 1..Len(chunk) : chunk[i].timestep = obs.timestep + (i - 1)
          \* In old implementation, chunk is returned to client which then merges
          \* This is modeled by making the chunk available for C1 to pick up
          \* For simplicity, we model immediate delivery by merging here
          /\ LET new_chunk_size == MaxInt(action_chunk_size, Len(chunk)) IN
             LET merged == MergeChunk(action_queue, chunk, latest_action_timestep) IN
             LET k == ScheduleMaxFactor * new_chunk_size IN
             LET trimmed == TrimQueue(merged, latest_action_timestep, k) IN
             \/ \* Success case: chunk delivered and merged
                /\ action_queue' = trimmed
                /\ action_chunk_size' = new_chunk_size
                /\ must_go_event' = TRUE  \* Re-arm must_go after receiving
             \/ \* Network failure case: chunk dropped (only when enabled)
                /\ EnableNetworkFailures
                /\ UNCHANGED <<action_queue, action_chunk_size, must_go_event>>
     /\ obs_slot' = None
     /\ last_processed_obs' = obs
     /\ predicted_obs_timesteps' = predicted_obs_timesteps \cup {obs.timestep}
     /\ did_execute' = FALSE
     /\ did_produce' = TRUE
     /\ UNCHANGED <<latest_action_timestep>>

(***************************************************************************
S1_Timeout: Server times out waiting for observation.

This models the obs_queue_timeout in the Python implementation.
When the server times out, it returns empty actions to the client.
***************************************************************************)
S1_Timeout ==
  /\ EnableServerTimeout
  /\ obs_slot = None  \* No observation available
  \* Server returns empty response - this is a no-op for the client
  /\ did_execute' = FALSE
  /\ did_produce' = FALSE
  /\ UNCHANGED <<latest_action_timestep, action_queue, action_chunk_size, must_go_event, obs_slot, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
S1_EnqueueRace: Models the server's race condition in _enqueue_observation.

In the old implementation:
  if self.observation_queue.full():
      _ = self.observation_queue.get_nowait()  # might raise Empty!
  self.observation_queue.put(obs)

If between full() and get_nowait() another thread consumes the observation,
get_nowait() raises Empty which is not caught, causing the current observation
to be lost.

This is a rare race but can be modeled for completeness.
***************************************************************************)
S1_EnqueueRace ==
  /\ EnableServerEnqueueRace
  /\ obs_slot # None  \* There was an observation
  \* Race: observation consumed between full() check and get_nowait()
  \* The incoming observation is lost
  /\ obs_slot' = None  \* Lost due to race
  /\ did_execute' = FALSE
  /\ did_produce' = FALSE
  /\ UNCHANGED <<latest_action_timestep, action_queue, action_chunk_size, must_go_event, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
Next-state relation + fairness
***************************************************************************)

\* Standard Next relation using Replace semantics for merging
Next ==
  C1_ReceiveAndMergeChunk \/
  C2_ExecuteNextAction \/
  C3_SendObservation \/
  S1_ProduceActionsForLatestObs \/
  S1_Timeout \/
  S1_EnqueueRace

\* Alternative Next relation using abstract aggregation for merging
\* WARNING: This is significantly more expensive for TLC due to non-determinism
NextWithAggregation ==
  C1_ReceiveWithAggregation \/
  C2_ExecuteNextAction \/
  C3_SendObservation \/
  S1_ProduceActionsForLatestObs \/
  S1_Timeout \/
  S1_EnqueueRace

Fairness ==
  /\ WF_vars(C1_ReceiveAndMergeChunk)
  /\ WF_vars(C2_ExecuteNextAction)
  /\ WF_vars(C3_SendObservation)
  /\ WF_vars(S1_ProduceActionsForLatestObs)
  \* Note: S1_Timeout and S1_EnqueueRace do not have fairness - they're failure cases

FairnessWithAggregation ==
  /\ WF_vars(C1_ReceiveWithAggregation)
  /\ WF_vars(C2_ExecuteNextAction)
  /\ WF_vars(C3_SendObservation)
  /\ WF_vars(S1_ProduceActionsForLatestObs)

\* Standard specification (Replace semantics, may include network failures/timeouts/races)
Spec == Init /\ [][Next]_vars /\ Fairness

\* Alternative specification with abstract aggregation
SpecWithAggregation == Init /\ [][NextWithAggregation]_vars /\ FairnessWithAggregation

(***************************************************************************
Safety invariants (checked in .cfg)
***************************************************************************)
Inv_ObsSlotType == obs_slot = None \/ obs_slot \in TimedObservation

Inv_QueueOnlyFuture == \A t \in Timestep : t <= latest_action_timestep => action_queue[t] = NoAction

Inv_ActionChunkSizeBound == action_chunk_size \in 1..ActionsPerChunk

\* Safety: Aggregation never produces invalid actions
Inv_AggregationProducesValidActions ==
  \A t \in Timestep : action_queue[t] \in Actions \cup {NoAction}

(***************************************************************************
Liveness properties (checked in .cfg)

NOTE: When EnableNetworkFailures is TRUE, some liveness properties may not
hold because messages can be dropped. In that case, use the weaker
"under fairness of delivery" variants or disable network failures.
***************************************************************************)

\* Original liveness: if queue non-empty, eventually execute
\* Holds regardless of network failures (local action)
Live_IfQueueNonEmptyEventuallyExec ==
  [](QueueLen(action_queue) > 0 => <> (did_execute = TRUE))

\* Original liveness: must_go observation eventually produces chunk
\* NOTE: May NOT hold when EnableNetworkFailures is TRUE
Live_MustGoObsEventuallyProducesChunk ==
  []((obs_slot # None /\ obs_slot.must_go) => <> (did_produce = TRUE))

(***************************************************************************
Additional properties
***************************************************************************)

\* Invariant: System is not deadlocked (can always make progress)
Inv_NotDeadlocked ==
  \/ QueueLen(action_queue) > 0
  \/ obs_slot # None
  \/ must_go_event  \* Can still send must_go to make progress

====
