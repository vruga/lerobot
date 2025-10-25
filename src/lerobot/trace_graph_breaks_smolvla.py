#!/usr/bin/env python
"""
Trace torch.compile graph breaks for SmolVLA policy.

This script uses torch._dynamo.explain to identify all graph breaks
and provide detailed information about what's preventing compilation.

Usage:
    python trace_graph_breaks_smolvla.py --device cuda
"""

import argparse
import torch
import torch._dynamo as dynamo
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def trace_graph_breaks(policy_name="smolvla", device="cuda", repo_id="AdilZtn/grab_red_cube_test_25"):
    """Trace and report all graph breaks in the policy."""

    print(f"🔍 Tracing graph breaks for {policy_name.upper()} policy")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 80)

    # Setup
    torch.manual_seed(42)
    device = torch.device(device)

    # Load dataset metadata
    ds_meta = LeRobotDatasetMetadata(repo_id)

    # Create policy configuration
    cfg = make_policy_config(
        policy_name,
        device=str(device),
        n_obs_steps=1,
        chunk_size=50,
        n_action_steps=50
    )

    # Create policy
    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.to(device)
    policy.eval()

    # Create preprocessor
    preprocessor, _ = make_smolvla_pre_post_processors(cfg, dataset_stats=ds_meta.stats)

    # Setup dataset
    delta_timestamps = {"action": [i / ds_meta.fps for i in range(cfg.chunk_size)]}
    dataset = LeRobotDataset(repo_id, episodes=[0], delta_timestamps=delta_timestamps)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)

    # Get sample batch
    sample_batch = next(iter(dataloader))

    # Move to device
    for key in sample_batch:
        if isinstance(sample_batch[key], torch.Tensor):
            sample_batch[key] = sample_batch[key].to(device)

    # Preprocess
    sample_batch = preprocessor(sample_batch)

    print("\n" + "=" * 80)
    print("TRACING INFERENCE (select_action)")
    print("=" * 80)

    # Enable verbose mode
    dynamo.config.verbose = True
    dynamo.config.suppress_errors = False

    # Trace inference
    print("\n🔬 Analyzing select_action method...")
    try:
        explanation = dynamo.explain(policy.select_action)(sample_batch)
        print("\n📊 GRAPH BREAK ANALYSIS:")
        print(f"Total graphs: {explanation.graph_count}")

        # PyTorch 2.7+ doesn't have break_count, calculate from graph_count
        break_count = explanation.graph_count - 1 if explanation.graph_count > 0 else 0
        print(f"Graph break count: {break_count}")

        if hasattr(explanation, 'break_reasons') and explanation.break_reasons:
            print(f"\n💥 GRAPH BREAKS:")
            for i, reason in enumerate(explanation.break_reasons, 1):
                print(f"\n  {i}. {reason}")
        else:
            print(f"\n💥 GRAPH BREAKS: {break_count} breaks detected (check warnings above for details)")

        if hasattr(explanation, 'ops_per_graph'):
            print(f"\n📈 Graph ops: {explanation.ops_per_graph}")

        if hasattr(explanation, 'out_guards'):
            print(f"\n🔧 Out guards: {explanation.out_guards}")

        # Print the full output
        print("\n" + "=" * 80)
        print("DETAILED EXPLANATION:")
        print("=" * 80)
        print(explanation)

    except Exception as e:
        print(f"\n❌ Error during tracing: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TRACING TRAINING (forward)")
    print("=" * 80)

    # Reset dynamo
    dynamo.reset()

    policy.train()

    print("\n🔬 Analyzing forward method...")
    try:
        explanation = dynamo.explain(policy.forward)(sample_batch)
        print("\n📊 GRAPH BREAK ANALYSIS:")
        print(f"Total graphs: {explanation.graph_count}")

        break_count = explanation.graph_count - 1 if explanation.graph_count > 0 else 0
        print(f"Graph break count: {break_count}")

        if hasattr(explanation, 'break_reasons') and explanation.break_reasons:
            print(f"\n💥 GRAPH BREAKS:")
            for i, reason in enumerate(explanation.break_reasons, 1):
                print(f"\n  {i}. {reason}")
        else:
            print(f"\n💥 GRAPH BREAKS: {break_count} breaks detected (check warnings above for details)")

        if hasattr(explanation, 'ops_per_graph'):
            print(f"\n📈 Graph ops: {explanation.ops_per_graph}")

        if hasattr(explanation, 'out_guards'):
            print(f"\n🔧 Out guards: {explanation.out_guards}")

        # Print the full output
        print("\n" + "=" * 80)
        print("DETAILED EXPLANATION:")
        print("=" * 80)
        print(explanation)

    except Exception as e:
        print(f"\n❌ Error during tracing: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TRACING CORE MODEL (model.sample_actions)")
    print("=" * 80)

    # Reset dynamo and clear CUDA cache to avoid OOM
    dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    policy.eval()

    # Prepare inputs for model.sample_actions
    print("\n🔬 Analyzing model.sample_actions method...")
    print("⚠️  Warning: This may require significant GPU memory")
    try:
        # Get the inputs
        images, img_masks = policy.prepare_images(sample_batch)
        state = policy.prepare_state(sample_batch)
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        lang_tokens = sample_batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = sample_batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        explanation = dynamo.explain(policy.model.sample_actions)(
            images, img_masks, lang_tokens, lang_masks, state
        )
        print("\n📊 GRAPH BREAK ANALYSIS:")
        print(f"Total graphs: {explanation.graph_count}")

        break_count = explanation.graph_count - 1 if explanation.graph_count > 0 else 0
        print(f"Graph break count: {break_count}")

        if hasattr(explanation, 'break_reasons') and explanation.break_reasons:
            print(f"\n💥 GRAPH BREAKS:")
            for i, reason in enumerate(explanation.break_reasons, 1):
                print(f"\n  {i}. {reason}")
        else:
            print(f"\n💥 GRAPH BREAKS: {break_count} breaks detected (check warnings above for details)")

        if hasattr(explanation, 'ops_per_graph'):
            print(f"\n📈 Graph ops: {explanation.ops_per_graph}")

        if hasattr(explanation, 'out_guards'):
            print(f"\n🔧 Out guards: {explanation.out_guards}")

        # Print the full output
        print("\n" + "=" * 80)
        print("DETAILED EXPLANATION:")
        print("=" * 80)
        print(explanation)

    except Exception as e:
        print(f"\n❌ Error during tracing: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TRACING VLM FORWARD")
    print("=" * 80)

    # Reset dynamo and clear CUDA cache
    dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n🔬 Analyzing model.forward method...")
    try:
        # Get the inputs
        images, img_masks = policy.prepare_images(sample_batch)
        state = policy.prepare_state(sample_batch)
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, ACTION
        lang_tokens = sample_batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = sample_batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = policy.prepare_action(sample_batch)

        explanation = dynamo.explain(policy.model.forward)(
            images, img_masks, lang_tokens, lang_masks, state, actions
        )
        print("\n📊 GRAPH BREAK ANALYSIS:")
        print(f"Total graphs: {explanation.graph_count}")

        break_count = explanation.graph_count - 1 if explanation.graph_count > 0 else 0
        print(f"Graph break count: {break_count}")

        if hasattr(explanation, 'break_reasons') and explanation.break_reasons:
            print(f"\n💥 GRAPH BREAKS:")
            for i, reason in enumerate(explanation.break_reasons, 1):
                print(f"\n  {i}. {reason}")
        else:
            print(f"\n💥 GRAPH BREAKS: {break_count} breaks detected (check warnings above for details)")

        if hasattr(explanation, 'ops_per_graph'):
            print(f"\n📈 Graph ops: {explanation.ops_per_graph}")

        if hasattr(explanation, 'out_guards'):
            print(f"\n🔧 Out guards: {explanation.out_guards}")

        # Print the full output
        print("\n" + "=" * 80)
        print("DETAILED EXPLANATION:")
        print("=" * 80)
        print(explanation)

    except Exception as e:
        print(f"\n❌ Error during tracing: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ TRACING COMPLETE")
    print("=" * 80)
    print("\n💡 Next Steps:")
    print("1. Review graph breaks above")
    print("2. Identify loops, control flow, and dynamic operations")
    print("3. Refactor problematic code patterns")
    print("4. Use torch.compile on individual functions where possible")


def main():
    parser = argparse.ArgumentParser(description="Trace torch.compile graph breaks")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on"
    )
    parser.add_argument(
        "--repo-id", default="AdilZtn/grab_red_cube_test_25", help="Dataset repo ID"
    )

    args = parser.parse_args()
    trace_graph_breaks(device=args.device, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
