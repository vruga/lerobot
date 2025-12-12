# Async Inference Bug Report

## Test Results Summary

The protocol invariant tests have successfully exposed multiple critical bugs in the async inference system:

### 1. Multiple Chunk Execution Failure ❌
**Test:** `test_multiple_chunks_are_executed`
**Expected:** At least 3 chunks should be received and executed
**Actual:** Only 1 chunk received
**Impact:** Robot stops after first action chunk, making it unusable for continuous operation
**Related Issues:** #1500, #2120

### 2. Action Execution Failure ❌  
**Test:** `test_action_timesteps_strictly_increase`
**Expected:** >20 actions should be executed with overlapping chunks
**Actual:** 0 actions executed despite 2 chunks being received
**Impact:** Actions are received but never executed, robot remains frozen
**Evidence:** Chunks received=2, Actions executed=0

### 3. Queue Starvation Under Latency ❌
**Test:** `test_queue_never_starves_under_latency`  
**Expected:** >50 actions executed in 8 seconds even with 500ms inference latency
**Actual:** 0 actions executed, complete system freeze
**Impact:** System cannot handle inference slower than control rate
**Evidence:** 3 chunks received but 0 actions executed

## Root Causes Analysis

Based on the test failures, the likely root causes are:

1. **Chunk Generation Logic**: The server is not properly generating subsequent chunks when `must_go` is set
2. **Action Queue Processing**: Actions are being received but not properly added to the execution queue
3. **Aggregation Failure**: The `_aggregate_action_queues` method may be dropping all actions instead of merging them
4. **Thread Synchronization**: Possible race condition between receiver and executor threads

## Test Execution Commands

To reproduce these bugs, run:

```bash
# Test multiple chunk execution
uv run python -m pytest tests/async_inference/test_protocol_invariants.py::test_multiple_chunks_are_executed -xvs

# Test action ordering  
uv run python -m pytest tests/async_inference/test_protocol_invariants.py::test_action_timesteps_strictly_increase -xvs

# Test queue starvation
uv run python -m pytest tests/async_inference/test_protocol_invariants.py::test_queue_never_starves_under_latency -xvs
```

## Next Steps

1. Debug why actions aren't being added to the execution queue
2. Investigate the chunk generation logic on the server side
3. Review the aggregation logic for action dropping
4. Add logging to trace the action flow from reception to execution
5. Consider implementing the TLA+ model to formally verify the protocol

## Test Infrastructure

The test suite (`test_protocol_invariants.py`) provides:
- Deterministic reproduction of bugs
- Monitoring of execution metrics
- Mock policies for controlled testing
- Clear failure messages indicating the specific bug

These tests should be run as part of CI/CD to prevent regression once bugs are fixed.
