---- MODULE AsyncInference ----
EXTENDS Integers, Sequences, FiniteSets, TLC

(***************************************************************************
Finite, TLC-friendly model of the protocol in:
  src/lerobot/async_inference/ASYNC_INFERENCE.SPEC.md

Key modeling choices:
- Finite domains (MaxTimestep, Actions, Payloads).
- No reals: threshold uses rationals ChunkSizeThresholdNum/Den.
- action_schedule is a total function [Timestep -> Actions \cup {NoAction}].
- Cross-node delivery is atomic in C4 (send+receive obs) and S2 (produce+enqueue chunk).
***************************************************************************)

CONSTANTS
  MaxTimestep,
  Actions,
  Payloads,
  ActionsPerChunk,
  IncomingChunkQueueMax,
  ScheduleMaxFactor,
  ChunkSizeThresholdNum,
  ChunkSizeThresholdDen

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

MergeChunk(schedule, chunk, latest) ==
  [t \in Timestep |->
    IF t <= latest
      THEN NoAction
      ELSE IF ChunkHasTimestep(chunk, t)
        THEN ChunkActionAt(chunk, t)
        ELSE schedule[t]
  ]

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

C3_ExecuteNextAction ==
  /\ ScheduleLen(action_schedule) > 0
  /\ LET t == MinScheduleTimestep(action_schedule) IN
     /\ t > latest_action_timestep
     /\ latest_action_timestep' = t
     /\ action_schedule' = [action_schedule EXCEPT ![t] = NoAction]
     /\ did_execute' = TRUE
     /\ did_produce' = FALSE
     /\ UNCHANGED <<incoming_action_chunks, action_chunk_size, must_go_armed, obs_slot, last_processed_obs, predicted_obs_timesteps>>

C4_SendObservation ==
  /\ action_chunk_size > 0
  /\ ScheduleLen(action_schedule) * ChunkSizeThresholdDen <= ChunkSizeThresholdNum * action_chunk_size
  /\ LET sendT == IF latest_action_timestep < 0 THEN 0 ELSE latest_action_timestep IN
     \E payload \in Payloads :
       LET sendMustGo == must_go_armed /\ ScheduleLen(action_schedule) = 0 IN
       LET obs == [timestep |-> sendT, payload |-> payload, must_go |-> sendMustGo] IN
       /\ must_go_armed' = IF sendMustGo THEN FALSE ELSE must_go_armed
       /\ obs_slot' =
            IF obs.must_go
              THEN obs
              ELSE IF ShouldProcess(obs, predicted_obs_timesteps)
                THEN obs
                ELSE obs_slot
       /\ did_execute' = FALSE
       /\ did_produce' = FALSE
       /\ Assert(~obs.must_go \/ ScheduleLen(action_schedule) = 0,
                "must_go sent when schedule was non-empty")
       /\ UNCHANGED <<latest_action_timestep, incoming_action_chunks, action_schedule, action_chunk_size, last_processed_obs, predicted_obs_timesteps>>

(***************************************************************************
Server action
***************************************************************************)

S2_ProduceActionsForLatestObs ==
  /\ obs_slot # None
  /\ LET obs == obs_slot IN
     LET maxLen == MinInt(ActionsPerChunk, MaxTimestep - obs.timestep + 1) IN
     /\ maxLen >= 1
     /\ \E chunk \in ActionChunk :
          /\ Len(chunk) = maxLen
          /\ \A i \in 1..Len(chunk) : chunk[i].timestep = obs.timestep + (i - 1)
          /\ incoming_action_chunks' = QueueEnqueueBounded(incoming_action_chunks, chunk)
     /\ obs_slot' = None
     /\ last_processed_obs' = obs
     /\ predicted_obs_timesteps' = predicted_obs_timesteps \cup {obs.timestep}
     /\ did_execute' = FALSE
     /\ did_produce' = TRUE
     /\ UNCHANGED <<latest_action_timestep, action_schedule, action_chunk_size, must_go_armed>>

(***************************************************************************
Next-state relation + fairness
***************************************************************************)

Next ==
  C1_EnqueueIncomingChunk \/
  C2_DrainOneIncomingChunk \/
  C3_ExecuteNextAction \/
  C4_SendObservation \/
  S2_ProduceActionsForLatestObs

Fairness ==
  /\ WF_vars(C2_DrainOneIncomingChunk)
  /\ WF_vars(C3_ExecuteNextAction)
  /\ WF_vars(C4_SendObservation)
  /\ WF_vars(S2_ProduceActionsForLatestObs)

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************
Safety invariants (checked in .cfg)
***************************************************************************)
Inv_IncomingQueueBound == Len(incoming_action_chunks) <= IncomingChunkQueueMax

Inv_ObsSlotType == obs_slot = None \/ obs_slot \in TimedObservation

Inv_ScheduleOnlyFuture == \A t \in Timestep : t <= latest_action_timestep => action_schedule[t] = NoAction

Inv_ActionChunkSizeBound == action_chunk_size \in 1..ActionsPerChunk

(***************************************************************************
Liveness properties (checked in .cfg)
***************************************************************************)

Live_IfScheduleNonEmptyEventuallyExec ==
  [](ScheduleLen(action_schedule) > 0 => <> (did_execute = TRUE))

Live_MustGoObsEventuallyProducesChunk ==
  []((obs_slot # None /\ obs_slot.must_go) => <> (did_produce = TRUE))

====
