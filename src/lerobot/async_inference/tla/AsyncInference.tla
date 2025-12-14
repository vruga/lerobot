---- MODULE AsyncInference ----
EXTENDS Integers, Sequences, FiniteSets, TLC

(***************************************************************************
Finite, TLC-friendly model of the protocol in:
  src/lerobot/async_inference/ASYNC_INFERENCE.SPEC.md

Key modeling choices:
- Finite domains (MaxTimestep, Actions, Payloads).
- No reals: threshold uses rationals ChunkSizeThresholdNum/Den.
- action_schedule is a total function [Timestep -> Actions \cup {NoAction}].
- Network failures: observation and chunk delivery can non-deterministically fail.
- Server timeout: models the obs_queue_timeout returning empty actions.
- Aggregation abstraction: merged actions can be any valid action (not just replace).
***************************************************************************)

CONSTANTS
  MaxTimestep,
  Actions,
  Payloads,
  ActionsPerChunk,
  IncomingChunkQueueMax,
  ScheduleMaxFactor,
  ChunkSizeThresholdNum,
  ChunkSizeThresholdDen,
  \* Network failure modeling: when TRUE, messages can be dropped
  EnableNetworkFailures,
  \* Server timeout modeling: when TRUE, server can timeout waiting for obs
  EnableServerTimeout

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
Schedule helpers
***************************************************************************)
ScheduleSet(schedule) == {t \in Timestep : schedule[t] # NoAction}
ScheduleLen(schedule) == Cardinality(ScheduleSet(schedule))

MinElem(S) == CHOOSE t \in S : \A u \in S : t <= u
MaxElem(S) == CHOOSE t \in S : \A u \in S : u <= t

MinScheduleTimestep(schedule) == MinElem(ScheduleSet(schedule))

ChunkTimesteps(chunk) == {chunk[i].timestep : i \in 1..Len(chunk)}
MaxChunkTimestep(chunk) == MaxElem(ChunkTimesteps(chunk))

MaxInt(a, b) == IF a >= b THEN a ELSE b
MinInt(a, b) == IF a <= b THEN a ELSE b

QueueEnqueueBounded(q, chunk) ==
  IF Len(q) < IncomingChunkQueueMax
    THEN Append(q, chunk)
    ELSE Append(Tail(q), chunk)

ChunkHasTimestep(chunk, t) == \E i \in 1..Len(chunk) : chunk[i].timestep = t

ChunkActionAt(chunk, t) ==
  CHOOSE a \in Actions : \E i \in 1..Len(chunk) : chunk[i].timestep = t /\ chunk[i].action = a

(***************************************************************************
Aggregation abstraction: when merging overlapping actions at the same timestep,
the result can be ANY valid action. This models that aggregate_fn (e.g.,
weighted_average) can produce results different from both inputs.

For Replace semantics (original behavior), use MergeChunkReplace.
For abstract aggregation, use MergeChunkWithAggregation.
***************************************************************************)

\* Original Replace semantics: new action overwrites old at same timestep
MergeChunkReplace(schedule, chunk, latest) ==
  [t \in Timestep |->
    IF t <= latest
      THEN NoAction
      ELSE IF ChunkHasTimestep(chunk, t)
        THEN ChunkActionAt(chunk, t)
        ELSE schedule[t]
  ]

\* Abstract aggregation: when both schedule and chunk have an action at t,
\* the result can be ANY action (models weighted_average, etc.)
\* This is a set of possible merged schedules (non-deterministic).
PossibleMergedSchedules(schedule, chunk, latest) ==
  { merged \in [Timestep -> Actions \cup {NoAction}] :
      \A t \in Timestep :
        IF t <= latest
          THEN merged[t] = NoAction
          ELSE IF ChunkHasTimestep(chunk, t) /\ schedule[t] # NoAction
            \* Both have action at t: aggregation can produce any action
            THEN merged[t] \in Actions
            ELSE IF ChunkHasTimestep(chunk, t)
              THEN merged[t] = ChunkActionAt(chunk, t)
              ELSE merged[t] = schedule[t]
  }

\* Default merge uses Replace semantics for backward compatibility
MergeChunk(schedule, chunk, latest) == MergeChunkReplace(schedule, chunk, latest)

KeepFirstK(schedule, latest, k) ==
  {t \in Timestep :
    /\ t > latest
    /\ schedule[t] # NoAction
    /\ Cardinality({u \in Timestep : u > latest /\ schedule[u] # NoAction /\ u < t}) < k
  }

TrimSchedule(schedule, latest, k) ==
  [t \in Timestep |->
    IF t <= latest
      THEN NoAction
      ELSE IF t \in KeepFirstK(schedule, latest, k)
        THEN schedule[t]
        ELSE NoAction
  ]

ShouldProcess(obs, predictedObsTimesteps) == obs.timestep \notin predictedObsTimesteps

(***************************************************************************
Variables
***************************************************************************)
VARIABLES
  latest_action_timestep,
  incoming_action_chunks,
  action_schedule,
  action_chunk_size,
  must_go_armed,
  did_execute,
  did_produce,
  obs_slot,
  last_processed_obs,
  predicted_obs_timesteps

vars == <<
  latest_action_timestep,
  incoming_action_chunks,
  action_schedule,
  action_chunk_size,
  must_go_armed,
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
  /\ incoming_action_chunks = << >>
  /\ action_schedule = [t \in Timestep |-> NoAction]
  /\ action_chunk_size = ActionsPerChunk
  /\ must_go_armed = TRUE
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
  /\ Len(incoming_action_chunks) <= IncomingChunkQueueMax
  /\ incoming_action_chunks \in [1..Len(incoming_action_chunks) -> ActionChunk]
  /\ action_schedule \in [Timestep -> Actions \cup {NoAction}]
  /\ action_chunk_size \in 1..ActionsPerChunk
  /\ must_go_armed \in BOOLEAN
  /\ did_execute \in BOOLEAN
  /\ did_produce \in BOOLEAN
  /\ obs_slot = None \/ obs_slot \in TimedObservation
  /\ last_processed_obs = None \/ last_processed_obs \in TimedObservation
  /\ predicted_obs_timesteps \subseteq Timestep

(***************************************************************************
Client actions
***************************************************************************)

C1_EnqueueIncomingChunk ==
  \E chunk \in ActionChunk :
    /\ incoming_action_chunks' = QueueEnqueueBounded(incoming_action_chunks, chunk)
    /\ did_execute' = FALSE
    /\ did_produce' = FALSE
    /\ UNCHANGED <<latest_action_timestep, action_schedule, action_chunk_size, must_go_armed, obs_slot, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
C2_DrainOneIncomingChunk: Drain one chunk and merge into schedule.

Uses Replace semantics (original behavior): new action overwrites old at
same timestep.
***************************************************************************)
C2_DrainOneIncomingChunk ==
  /\ Len(incoming_action_chunks) > 0
  /\ LET chunk == Head(incoming_action_chunks) IN
     LET rest == Tail(incoming_action_chunks) IN
     /\ incoming_action_chunks' = rest
     /\ did_execute' = FALSE
     /\ did_produce' = FALSE
     /\ IF Len(chunk) = 0
          THEN
            /\ UNCHANGED << action_schedule, action_chunk_size, must_go_armed >>
          ELSE IF MaxChunkTimestep(chunk) <= latest_action_timestep
            THEN
              /\ UNCHANGED << action_schedule, action_chunk_size, must_go_armed >>
            ELSE
              LET new_chunk_size == MaxInt(action_chunk_size, Len(chunk)) IN
              LET merged == MergeChunk(action_schedule, chunk, latest_action_timestep) IN
              LET k == ScheduleMaxFactor * new_chunk_size IN
              LET trimmed == TrimSchedule(merged, latest_action_timestep, k) IN
              /\ action_chunk_size' = new_chunk_size
              /\ action_schedule' = trimmed
              /\ must_go_armed' = TRUE
              /\ Assert(\A t \in Timestep : t <= latest_action_timestep => trimmed[t] = NoAction,
                        "C2 produced schedule with past timesteps")
     /\ UNCHANGED <<latest_action_timestep, obs_slot, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
C2_DrainWithAggregation: Variant that uses abstract aggregation.

When merging overlapping actions at the same timestep, the result can be
ANY valid action. This models aggregate_fn (e.g., weighted_average) which
can produce results different from both the old and new action.

This is more expensive for TLC to check due to non-determinism, but provides
higher fidelity to the actual implementation.
***************************************************************************)
C2_DrainWithAggregation ==
  /\ Len(incoming_action_chunks) > 0
  /\ LET chunk == Head(incoming_action_chunks) IN
     LET rest == Tail(incoming_action_chunks) IN
     /\ incoming_action_chunks' = rest
     /\ did_execute' = FALSE
     /\ did_produce' = FALSE
     /\ IF Len(chunk) = 0
          THEN
            /\ UNCHANGED << action_schedule, action_chunk_size, must_go_armed >>
          ELSE IF MaxChunkTimestep(chunk) <= latest_action_timestep
            THEN
              /\ UNCHANGED << action_schedule, action_chunk_size, must_go_armed >>
            ELSE
              LET new_chunk_size == MaxInt(action_chunk_size, Len(chunk)) IN
              LET k == ScheduleMaxFactor * new_chunk_size IN
              \* Non-deterministically choose from possible merged schedules
              \E merged \in PossibleMergedSchedules(action_schedule, chunk, latest_action_timestep) :
                LET trimmed == TrimSchedule(merged, latest_action_timestep, k) IN
                /\ action_chunk_size' = new_chunk_size
                /\ action_schedule' = trimmed
                /\ must_go_armed' = TRUE
                /\ Assert(\A t \in Timestep : t <= latest_action_timestep => trimmed[t] = NoAction,
                          "C2 produced schedule with past timesteps")
     /\ UNCHANGED <<latest_action_timestep, obs_slot, last_processed_obs, predicted_obs_timesteps>>

C3_ExecuteNextAction ==
  /\ ScheduleLen(action_schedule) > 0
  /\ LET t == MinScheduleTimestep(action_schedule) IN
     /\ t > latest_action_timestep
     /\ latest_action_timestep' = t
     /\ action_schedule' = [action_schedule EXCEPT ![t] = NoAction]
     /\ did_execute' = TRUE
     /\ did_produce' = FALSE
     /\ UNCHANGED <<incoming_action_chunks, action_chunk_size, must_go_armed, obs_slot, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
C4_SendObservation: Client sends observation to server.

Network failure modeling: When EnableNetworkFailures is TRUE, the observation
may be dropped during network delivery (obs_slot unchanged). This models
gRPC errors, timeouts, or network partitions.

Note: must_go is still disarmed even if delivery fails, matching the
implementation where the client doesn't know if delivery succeeded.
***************************************************************************)
C4_SendObservation ==
  /\ action_chunk_size > 0
  /\ ScheduleLen(action_schedule) * ChunkSizeThresholdDen <= ChunkSizeThresholdNum * action_chunk_size
  /\ LET sendT == IF latest_action_timestep < 0 THEN 0 ELSE latest_action_timestep IN
     \E payload \in Payloads :
       LET sendMustGo == must_go_armed /\ ScheduleLen(action_schedule) = 0 IN
       LET obs == [timestep |-> sendT, payload |-> payload, must_go |-> sendMustGo] IN
       /\ must_go_armed' = IF sendMustGo THEN FALSE ELSE must_go_armed
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
       /\ Assert(~obs.must_go \/ ScheduleLen(action_schedule) = 0,
                "must_go sent when schedule was non-empty")
       /\ UNCHANGED <<latest_action_timestep, incoming_action_chunks, action_schedule, action_chunk_size, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
Server actions
***************************************************************************)

(***************************************************************************
S2_ProduceActionsForLatestObs: Server processes observation and produces chunk.

Network failure modeling: When EnableNetworkFailures is TRUE, the produced
action chunk may be dropped during network delivery (incoming_action_chunks
unchanged). This models gRPC errors or network partitions on the return path.
***************************************************************************)
S2_ProduceActionsForLatestObs ==
  /\ obs_slot # None
  /\ LET obs == obs_slot IN
     LET maxLen == MinInt(ActionsPerChunk, MaxTimestep - obs.timestep + 1) IN
     /\ maxLen >= 1
     /\ \E chunk \in ActionChunk :
          /\ Len(chunk) = maxLen
          /\ \A i \in 1..Len(chunk) : chunk[i].timestep = obs.timestep + (i - 1)
          /\ \/ \* Success case: chunk delivered to client
                incoming_action_chunks' = QueueEnqueueBounded(incoming_action_chunks, chunk)
             \/ \* Network failure case: chunk dropped (only when enabled)
                /\ EnableNetworkFailures
                /\ incoming_action_chunks' = incoming_action_chunks
     /\ obs_slot' = None
     /\ last_processed_obs' = obs
     /\ predicted_obs_timesteps' = predicted_obs_timesteps \cup {obs.timestep}
     /\ did_execute' = FALSE
     /\ did_produce' = TRUE
     /\ UNCHANGED <<latest_action_timestep, action_schedule, action_chunk_size, must_go_armed>>

(***************************************************************************
S2_Timeout: Server times out waiting for observation.

This models the obs_queue_timeout in the Python implementation:
  obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
  except Empty:
      return services_pb2.Empty()  # Returns empty actions!

When the server times out, it returns an empty action chunk to the client.
This is modeled by enqueueing an empty chunk.
***************************************************************************)
S2_Timeout ==
  /\ EnableServerTimeout
  /\ obs_slot = None  \* No observation available
  \* Server returns empty response - client receives empty chunk
  /\ incoming_action_chunks' = QueueEnqueueBounded(incoming_action_chunks, << >>)
  /\ did_execute' = FALSE
  /\ did_produce' = FALSE  \* No actual production happened
  /\ UNCHANGED <<latest_action_timestep, action_schedule, action_chunk_size, must_go_armed, obs_slot, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
Next-state relation + fairness
***************************************************************************)

\* Standard Next relation using Replace semantics for merging
Next ==
  C1_EnqueueIncomingChunk \/
  C2_DrainOneIncomingChunk \/
  C3_ExecuteNextAction \/
  C4_SendObservation \/
  S2_ProduceActionsForLatestObs \/
  S2_Timeout

\* Alternative Next relation using abstract aggregation for merging
\* WARNING: This is significantly more expensive for TLC due to non-determinism
NextWithAggregation ==
  C1_EnqueueIncomingChunk \/
  C2_DrainWithAggregation \/
  C3_ExecuteNextAction \/
  C4_SendObservation \/
  S2_ProduceActionsForLatestObs \/
  S2_Timeout

Fairness ==
  /\ WF_vars(C2_DrainOneIncomingChunk)
  /\ WF_vars(C3_ExecuteNextAction)
  /\ WF_vars(C4_SendObservation)
  /\ WF_vars(S2_ProduceActionsForLatestObs)
  \* Note: S2_Timeout does not have fairness - it's a failure/edge case, not guaranteed to happen

FairnessWithAggregation ==
  /\ WF_vars(C2_DrainWithAggregation)
  /\ WF_vars(C3_ExecuteNextAction)
  /\ WF_vars(C4_SendObservation)
  /\ WF_vars(S2_ProduceActionsForLatestObs)

\* Standard specification (Replace semantics, may include network failures/timeouts)
Spec == Init /\ [][Next]_vars /\ Fairness

\* Alternative specification with abstract aggregation
SpecWithAggregation == Init /\ [][NextWithAggregation]_vars /\ FairnessWithAggregation

(***************************************************************************
Safety invariants (checked in .cfg)
***************************************************************************)
Inv_IncomingQueueBound == Len(incoming_action_chunks) <= IncomingChunkQueueMax

Inv_ObsSlotType == obs_slot = None \/ obs_slot \in TimedObservation

Inv_ScheduleOnlyFuture == \A t \in Timestep : t <= latest_action_timestep => action_schedule[t] = NoAction

Inv_ActionChunkSizeBound == action_chunk_size \in 1..ActionsPerChunk

(***************************************************************************
Liveness properties (checked in .cfg)

NOTE: When EnableNetworkFailures is TRUE, some liveness properties may not
hold because messages can be dropped. In that case, use the weaker
"under fairness of delivery" variants or disable network failures.
***************************************************************************)

\* Original liveness: if schedule non-empty, eventually execute
\* Holds regardless of network failures (local action)
Live_IfScheduleNonEmptyEventuallyExec ==
  [](ScheduleLen(action_schedule) > 0 => <> (did_execute = TRUE))

\* Original liveness: must_go observation eventually produces chunk
\* NOTE: May NOT hold when EnableNetworkFailures is TRUE because:
\* 1. The observation delivery to server may fail
\* 2. The chunk delivery back to client may fail
Live_MustGoObsEventuallyProducesChunk ==
  []((obs_slot # None /\ obs_slot.must_go) => <> (did_produce = TRUE))

(***************************************************************************
Additional properties for higher-fidelity model
***************************************************************************)

\* Invariant: After a timeout, client receives an empty chunk
\* (This is implicitly true by the S2_Timeout action definition)

\* Safety: Aggregation never produces invalid actions
\* (This is enforced by the type constraint in PossibleMergedSchedules)
Inv_AggregationProducesValidActions ==
  \A t \in Timestep : action_schedule[t] \in Actions \cup {NoAction}

\* Liveness variant: If network eventually delivers, progress is made
\* This is a conditional property - requires assumptions about network behavior
\* In practice, check with EnableNetworkFailures = FALSE first

(***************************************************************************
Debugging/instrumentation properties
***************************************************************************)

\* Track if we ever reach a state where:
\* - Schedule is empty
\* - No observations in flight
\* - No incoming chunks
\* This would be a "stuck" state in the protocol
Inv_NotDeadlocked ==
  \/ ScheduleLen(action_schedule) > 0
  \/ obs_slot # None
  \/ Len(incoming_action_chunks) > 0
  \/ must_go_armed  \* Can still send must_go to make progress

====
