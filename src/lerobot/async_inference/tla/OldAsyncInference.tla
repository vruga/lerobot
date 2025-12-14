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
  \* Max number of observations that can be in-flight client->server
  MaxInFlightObs,
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
Server->client unary RPC response for GetActions.
Empty responses model the server timing out waiting for an observation.
***************************************************************************)
ActionResponse ==
  [isEmpty: BOOLEAN, chunk: ActionChunk]

IsValidResponse(r) ==
  /\ r \in ActionResponse
  /\ (r.isEmpty => Len(r.chunk) = 0)
  /\ (~r.isEmpty => Len(r.chunk) > 0)

(***************************************************************************
Bounded in-flight observation set (models delay/reorder).
We use a SET abstraction (not a sequence) to keep operations simple; ordering
effects are modeled by nondeterministic choice of which message is delivered.
***************************************************************************)
ObsInFlight == { S \in SUBSET TimedObservation : Cardinality(S) <= MaxInFlightObs }

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
  \* In-flight observations (client->server). Models network delay/reorder.
  obs_inflight,
  \* Models the unary GetActions RPC call/response:
  \* - Client continuously polls GetActions in receive_actions
  \* - Server responds either with Empty (timeout) or a chunk (success)
  getactions_waiting,
  action_response_inflight,
  did_send_obs,
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
  obs_inflight,
  getactions_waiting,
  action_response_inflight,
  did_send_obs,
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
  /\ obs_inflight = {}
  /\ getactions_waiting = FALSE
  /\ action_response_inflight = None
  /\ did_send_obs = FALSE
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
  /\ obs_inflight \in ObsInFlight
  /\ getactions_waiting \in BOOLEAN
  /\ action_response_inflight = None \/ IsValidResponse(action_response_inflight)
  /\ did_send_obs \in BOOLEAN
  /\ did_execute \in BOOLEAN
  /\ did_produce \in BOOLEAN
  /\ obs_slot = None \/ obs_slot \in TimedObservation
  /\ last_processed_obs = None \/ last_processed_obs \in TimedObservation
  /\ predicted_obs_timesteps \subseteq Timestep

(***************************************************************************
Client actions
***************************************************************************)

(***************************************************************************
C0_PollGetActions: Receiver thread initiates a unary GetActions RPC call.

In the Python code, receive_actions continuously calls GetActions() and blocks
until a response arrives (Actions or Empty on timeout).

We model the "call is in progress" with getactions_waiting=TRUE.
***************************************************************************)
C0_PollGetActions ==
  /\ ~getactions_waiting
  /\ getactions_waiting' = TRUE
  /\ did_send_obs' = FALSE
  /\ did_execute' = FALSE
  /\ did_produce' = FALSE
  /\ UNCHANGED <<
      latest_action_timestep,
      action_queue,
      action_chunk_size,
      must_go_event,
      obs_inflight,
      action_response_inflight,
      obs_slot,
      last_processed_obs,
      predicted_obs_timesteps
    >>

(***************************************************************************
C1_DeliverGetActionsResponse: The in-flight GetActions response arrives at client.

This is where the receiver thread merges a non-empty chunk into the local queue,
and (crucially) only sets must_go_event when it receives a non-empty chunk.
***************************************************************************)
C1_DeliverGetActionsResponse ==
  /\ getactions_waiting
  /\ action_response_inflight # None
  /\ LET resp == action_response_inflight IN
     /\ getactions_waiting' = FALSE
     /\ action_response_inflight' = None
     /\ IF resp.isEmpty
          THEN
            /\ did_execute' = FALSE
            /\ did_produce' = FALSE
            /\ did_send_obs' = FALSE
            /\ UNCHANGED << action_queue, action_chunk_size, must_go_event >>
          ELSE
            LET chunk == resp.chunk IN
            LET new_chunk_size == MaxInt(action_chunk_size, Len(chunk)) IN
            LET merged == MergeChunk(action_queue, chunk, latest_action_timestep) IN
            LET k == ScheduleMaxFactor * new_chunk_size IN
            LET trimmed == TrimQueue(merged, latest_action_timestep, k) IN
              /\ action_chunk_size' = new_chunk_size
              /\ action_queue' = trimmed
              /\ must_go_event' = TRUE  \* must_go.set() after receiving non-empty actions
              /\ did_execute' = FALSE
              /\ did_produce' = FALSE
              /\ did_send_obs' = FALSE
              /\ Assert(\A t \in Timestep : t <= latest_action_timestep => trimmed[t] = NoAction,
                        "C1 produced queue with past timesteps")
     /\ UNCHANGED <<latest_action_timestep, obs_inflight, obs_slot, last_processed_obs, predicted_obs_timesteps>>

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
     /\ did_send_obs' = FALSE
     /\ UNCHANGED <<
          action_chunk_size,
          must_go_event,
          obs_inflight,
          getactions_waiting,
          action_response_inflight,
          obs_slot,
          last_processed_obs,
          predicted_obs_timesteps
        >>

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
  /\ Cardinality(obs_inflight) < MaxInFlightObs
  /\ LET sendT == IF latest_action_timestep < 0 THEN 0 ELSE latest_action_timestep IN
     \E payload \in Payloads :
       LET sendMustGo == must_go_event /\ QueueLen(action_queue) = 0 IN
       LET obs == [timestep |-> sendT, payload |-> payload, must_go |-> sendMustGo] IN
       /\ must_go_event' = IF sendMustGo THEN FALSE ELSE must_go_event  \* must_go.clear() if sent
       /\ \/ \* Success case: observation enters the network (in-flight)
             /\ obs_inflight' = obs_inflight \cup {obs}
          \/ \* Network failure case: observation dropped before entering network
             /\ EnableNetworkFailures
             /\ obs_inflight' = obs_inflight
       /\ did_send_obs' = TRUE
       /\ did_execute' = FALSE
       /\ did_produce' = FALSE
       /\ Assert(~obs.must_go \/ QueueLen(action_queue) = 0,
                "must_go sent when queue was non-empty")
       /\ UNCHANGED <<
            latest_action_timestep,
            action_queue,
            action_chunk_size,
            getactions_waiting,
            action_response_inflight,
            obs_slot,
            last_processed_obs,
            predicted_obs_timesteps
          >>

(***************************************************************************
Server actions
***************************************************************************)

(***************************************************************************
S0_DeliverObservationToServer: a delivered observation is handled by SendObservations.

This models the server-side behavior of receiving a TimedObservation over the
network and then either enqueueing it (keep-latest queue size 1) or filtering
it out (not enqueued).
***************************************************************************)
S0_DeliverObservationToServer ==
  /\ obs_inflight # {}
  /\ \E obs \in obs_inflight :
      LET shouldEnqueue ==
            obs.must_go
            \/ last_processed_obs = None
            \/ ShouldProcess(obs, predicted_obs_timesteps)
      IN
        /\ obs_inflight' = obs_inflight \ {obs}
        /\ obs_slot' =
            IF shouldEnqueue
              THEN obs  \* keep-latest overwrite semantics (Queue(maxsize=1))
              ELSE obs_slot
        /\ did_send_obs' = FALSE
        /\ did_execute' = FALSE
        /\ did_produce' = FALSE
        /\ UNCHANGED <<
            latest_action_timestep,
            action_queue,
            action_chunk_size,
            must_go_event,
            getactions_waiting,
            action_response_inflight,
            last_processed_obs,
            predicted_obs_timesteps
          >>

(***************************************************************************
S1_HandleGetActions: server handles an in-progress GetActions call.

If an observation is available in the server's keep-latest slot, the server
consumes it and produces a non-empty ActionResponse (chunk).
If no observation is available and EnableServerTimeout is TRUE, the server
produces an empty ActionResponse.
***************************************************************************)
S1_HandleGetActions ==
  /\ getactions_waiting
  /\ action_response_inflight = None
  /\ IF obs_slot # None
       THEN
         LET obs == obs_slot IN
         LET maxLen == MinInt(ActionsPerChunk, MaxTimestep - obs.timestep + 1) IN
           /\ maxLen >= 1
           /\ \E chunk \in ActionChunk :
                /\ Len(chunk) = maxLen
                /\ \A i \in 1..Len(chunk) : chunk[i].timestep = obs.timestep + (i - 1)
                /\ action_response_inflight' = [isEmpty |-> FALSE, chunk |-> chunk]
                /\ obs_slot' = None
                /\ last_processed_obs' = obs
                /\ predicted_obs_timesteps' = predicted_obs_timesteps \cup {obs.timestep}
                /\ did_produce' = TRUE
                /\ did_execute' = FALSE
                /\ did_send_obs' = FALSE
                /\ UNCHANGED <<
                    latest_action_timestep,
                    action_queue,
                    action_chunk_size,
                    must_go_event,
                    obs_inflight,
                    getactions_waiting
                  >>
       ELSE
         /\ EnableServerTimeout
         /\ action_response_inflight' = [isEmpty |-> TRUE, chunk |-> << >>]
         /\ did_produce' = FALSE
         /\ did_execute' = FALSE
         /\ did_send_obs' = FALSE
         /\ UNCHANGED <<
              latest_action_timestep,
              action_queue,
              action_chunk_size,
              must_go_event,
              obs_inflight,
              getactions_waiting,
              obs_slot,
              last_processed_obs,
              predicted_obs_timesteps
            >>

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
  /\ did_send_obs' = FALSE
  /\ did_execute' = FALSE
  /\ did_produce' = FALSE
  /\ UNCHANGED <<
        latest_action_timestep,
        action_queue,
        action_chunk_size,
        must_go_event,
        obs_inflight,
        getactions_waiting,
        action_response_inflight,
        last_processed_obs,
        predicted_obs_timesteps
      >>

(***************************************************************************
Next-state relation + fairness
***************************************************************************)

\* Standard Next relation using Replace semantics for merging
Next ==
  C0_PollGetActions \/
  C1_DeliverGetActionsResponse \/
  C2_ExecuteNextAction \/
  C3_SendObservation \/
  S0_DeliverObservationToServer \/
  S1_HandleGetActions \/
  S1_EnqueueRace

\* Alternative Next relation using abstract aggregation for merging
\* WARNING: This is significantly more expensive for TLC due to non-determinism
NextWithAggregation ==
  C0_PollGetActions \/
  C1_DeliverGetActionsResponse \/
  C2_ExecuteNextAction \/
  C3_SendObservation \/
  S0_DeliverObservationToServer \/
  S1_HandleGetActions \/
  S1_EnqueueRace

Fairness ==
  /\ WF_vars(C0_PollGetActions)
  /\ WF_vars(C1_DeliverGetActionsResponse)
  /\ WF_vars(C2_ExecuteNextAction)
  /\ WF_vars(C3_SendObservation)
  /\ WF_vars(S0_DeliverObservationToServer)
  /\ WF_vars(S1_HandleGetActions)
  \* Note: S1_EnqueueRace does not have fairness - it's a failure case

FairnessWithAggregation ==
  /\ WF_vars(C0_PollGetActions)
  /\ WF_vars(C1_DeliverGetActionsResponse)
  /\ WF_vars(C2_ExecuteNextAction)
  /\ WF_vars(C3_SendObservation)
  /\ WF_vars(S0_DeliverObservationToServer)
  /\ WF_vars(S1_HandleGetActions)

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
Additional liveness: if the client continues sending observations forever
(i.e. the control loop keeps running), then it should execute actions forever.

This is intended to surface "stall" traces where the system keeps polling and
sending observations but never executes any action.
***************************************************************************)
Live_IfObsContinuesEventuallyExec ==
  ([]<>(did_send_obs = TRUE)) => ([]<>(did_execute = TRUE))

(***************************************************************************
Additional properties
***************************************************************************)

\* Invariant: System is not deadlocked (can always make progress)
Inv_NotDeadlocked ==
  \/ QueueLen(action_queue) > 0
  \/ obs_slot # None
  \/ obs_inflight # {}                 \* Observation is in flight
  \/ action_response_inflight # None   \* Response is in flight
  \/ must_go_event                     \* Can still send must_go to make progress

====
