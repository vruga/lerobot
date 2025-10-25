# torch.compile Support for SmolVLA Policy

This document describes the torch.compile optimization work done for the SmolVLA policy.

## Overview

The SmolVLA policy has been refactored to support `torch.compile` for improved inference and training performance. The optimization focuses on:

1. **Removing graph breaks** by eliminating problematic patterns
2. **Vectorizing loops** to enable better compilation
3. **Partial compilation** of individual functions rather than the entire model

## Changes Made

### 1. Vectorized Aloha Transformations

**Before** (lines 386-411 in modeling_smolvla.py):
```python
def _pi_aloha_decode_state(self, state):
    for motor_idx in [1, 2, 8, 9]:
        state[:, motor_idx] *= -1
    for motor_idx in [6, 13]:
        state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
    return state
```

**After**:
```python
def _pi_aloha_decode_state(self, state):
    state = state.clone()  # Avoid in-place ops
    flip_indices = torch.tensor([1, 2, 8, 9], device=state.device, dtype=torch.long)
    state[:, flip_indices] *= -1
    gripper_indices = torch.tensor([6, 13], device=state.device, dtype=torch.long)
    state[:, gripper_indices] = aloha_gripper_to_angular(state[:, gripper_indices])
    return state
```

**Benefit**: Removes for loops, uses vectorized indexing which compiles well.

### 2. Fixed Iteration Denoising Loop

**Before** (lines 746-757):
```python
while time >= -dt / 2:
    expanded_time = time.expand(bsize)
    v_t = self.denoise_step(...)
    x_t += dt * v_t
    time += dt
```

**After**:
```python
def _denoising_loop(self, prefix_pad_masks, past_key_values, x_t, dt, bsize, num_steps):
    for step_idx in range(num_steps):
        time_val = 1.0 + step_idx * dt.item()
        time = torch.full((bsize,), time_val, dtype=torch.float32, device=x_t.device)
        v_t = self.denoise_step(...)
        x_t = x_t + dt * v_t
    return x_t
```

**Benefit**: Replaces dynamic while loop with fixed iteration for loop. PyTorch can unroll and optimize this.

### 3. Optimized Attention Mask Building

**Before** (lines 567, 584, 598):
```python
att_masks += [0] * (image_start_mask.shape[-1])
# ... more list concatenation
att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
```

**After**:
```python
# Build attention mask using tensor operations
att_masks = torch.cat([
    torch.zeros(num_att_mask_zeros, dtype=torch.bool, device=device),
    torch.ones(states_seq_len, dtype=torch.bool, device=device)
])
```

**Benefit**: Avoids Python list operations and uses native tensor concatenation.

### 4. Configuration Options

Added two new config options in `configuration_smolvla.py`:

```python
use_torch_compile: bool = False  # Enable torch.compile
compile_mode: str = "default"     # Compilation mode
```

Modes available:
- `"default"`: Balanced compilation
- `"reduce-overhead"`: Minimize Python overhead
- `"max-autotune"`: Maximum optimization (slower compile time)

## Usage

### Training with torch.compile

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.use_torch_compile=true \
  --policy.compile_mode=default \
  --dataset.repo_id=danaaubakirova/svla_so100_task1_v3 \
  --batch_size=64 \
  --steps=200000
```

### Inference with torch.compile

```python
from lerobot.policies.factory import make_policy_config, make_policy

config = make_policy_config(
    "smolvla",
    use_torch_compile=True,
    compile_mode="reduce-overhead"
)

policy = make_policy(config, ds_meta=ds_meta)
```

### Benchmarking

Use the provided benchmark script to measure performance:

```bash
# Run tracing to see graph breaks
python trace_graph_breaks_smolvla.py --device cuda

# Benchmark performance
python benchmark_inference_compile_lerobot.py \
  --policy smolvla \
  --device cuda \
  --output smolvla_benchmark.md
```

## What Gets Compiled

When `use_torch_compile=True`, the following functions are compiled:

**VLAFlowMatching level**:
- `denoise_step()` - Single denoising step computation
- `_denoising_loop()` - Full denoising iteration
- `embed_suffix()` - Action/timestep embedding

**SmolVLAPolicy level** (when `adapt_to_pi_aloha=True`):
- `_pi_aloha_decode_state()` - State transformation
- `_pi_aloha_encode_actions()` - Action encoding
- `_pi_aloha_encode_actions_inv()` - Inverse action encoding
- `prepare_state()` - State preparation
- `prepare_action()` - Action preparation

**Not compiled** (intentionally):
- `embed_prefix()` - Has dynamic loop over variable number of images
- `prepare_images()` - List comprehensions with variable camera count
- `select_action()` - Contains queue management (Python objects)

## Expected Performance Improvements

Based on typical torch.compile gains:

- **Inference**: 1.2-2x speedup (depending on GPU, batch size)
- **Training**: 1.1-1.5x speedup
- **First run**: Slower due to compilation overhead (~30-60s)
- **Subsequent runs**: Full speedup

## Debugging Graph Breaks

Use the tracing script to identify remaining graph breaks:

```bash
python trace_graph_breaks_smolvla.py --device cuda 2>&1 | tee trace_output.txt
```

Look for:
- Python list operations (use tensors instead)
- `.item()` calls (delays compilation)
- Dynamic control flow (if/while with data-dependent conditions)
- Python object access (deques, dicts with non-tensor values)

## Future Optimization Opportunities

1. **Batch image processing**: Stack images into a single tensor and process in parallel
2. **Compiled prepare_images**: Refactor to use fixed camera count
3. **Full graph compilation**: Eliminate remaining graph breaks in embed_prefix
4. **Custom CUDA kernels**: For specific bottleneck operations

## Testing

Always verify correctness after enabling torch.compile:

```python
# Test that outputs match
policy_eager = make_policy(config)
config.use_torch_compile = True
policy_compiled = make_policy(config)

# Compare outputs
output_eager = policy_eager.select_action(batch)
output_compiled = policy_compiled.select_action(batch)

assert torch.allclose(output_eager, output_compiled, atol=1e-5)
```

## References

- [PyTorch torch.compile documentation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [Graph breaks explanation](https://pytorch.org/docs/stable/torch.compiler_faq.html#graph-breaks)
- [SmolVLA paper](https://huggingface.co/papers/2506.01844)
