#!/usr/bin/env python

"""
Comparison script for ACT policy with and without torch.compile.

This script tests the performance and correctness of the ACT policy
both with and without torch.compile to validate improvements.
"""

import argparse
import time
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.utils.constants import OBS_IMAGES


class TorchCompileComparison:
    """Compare ACT policy performance with and without torch.compile."""
    
    def __init__(self, device: str = "cpu", batch_size: int = 4):
        self.device = device
        self.batch_size = batch_size
        self.policy = None
        self.sample_batch = None
        
    def setup_policy_and_data(self) -> None:
        """Setup ACT policy and synthetic test data."""
        print("🔧 Setting up ACT policy and test data...")
        
        # Create ACT configuration
        cfg = ACTConfig(
            input_features={
                "observation.images.top": PolicyFeature(
                    shape=(3, 480, 640),
                    type=FeatureType.VISUAL
                ),
                "observation.images.wrist": PolicyFeature(
                    shape=(3, 480, 640), 
                    type=FeatureType.VISUAL
                ),
                "observation.state": PolicyFeature(
                    shape=(6,),
                    type=FeatureType.STATE
                ),
                "observation.environment_state": PolicyFeature(
                    shape=(6,),
                    type=FeatureType.STATE
                )
            },
            output_features={
                "action": PolicyFeature(
                    shape=(6,),
                    type=FeatureType.ACTION
                )
            },
            chunk_size=10,
            n_action_steps=5,
            use_vae=True
        )
        
        # Create policy
        self.policy = ACTPolicy(cfg).to(self.device)
        
        # Create synthetic test data
        self.sample_batch = {
            "observation.images.top": torch.randn(
                self.batch_size, 3, 480, 640, 
                dtype=torch.float32, device=self.device
            ),
            "observation.images.wrist": torch.randn(
                self.batch_size, 3, 480, 640, 
                dtype=torch.float32, device=self.device
            ),
            "observation.state": torch.randn(
                self.batch_size, 6, 
                dtype=torch.float32, device=self.device
            ),
            "observation.environment_state": torch.randn(
                self.batch_size, 6, 
                dtype=torch.float32, device=self.device
            ),
            "action": torch.randn(
                self.batch_size, cfg.chunk_size, 6, 
                dtype=torch.float32, device=self.device
            ),
            "action_is_pad": torch.zeros(
                self.batch_size, cfg.chunk_size, 
                dtype=torch.bool, device=self.device
            )
        }
        
        print("✅ Setup complete")
    
    def benchmark_inference(self, use_compile: bool = False, num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark inference performance."""
        print(f"\n🚀 Benchmarking inference ({'with' if use_compile else 'without'} torch.compile)...")
        
        if use_compile:
            self.policy.enable_torch_compile(mode="default")
        else:
            self.policy.disable_torch_compile()
        
        # Warmup runs
        for _ in range(3):
            with torch.no_grad():
                _ = self.policy.predict_action_chunk(self.sample_batch)
        
        # Benchmark runs
        times = []
        for i in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                actions = self.policy.predict_action_chunk(self.sample_batch)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min(times),
            "max_time": max(times),
            "times": times
        }
    
    def benchmark_training(self, use_compile: bool = False, num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark training performance."""
        print(f"\n🏋️ Benchmarking training ({'with' if use_compile else 'without'} torch.compile)...")
        
        if use_compile:
            self.policy.enable_torch_compile(mode="default")
        else:
            self.policy.disable_torch_compile()
        
        # Set to training mode
        self.policy.train()
        
        # Warmup runs
        for _ in range(2):
            loss, loss_dict = self.policy.forward(self.sample_batch)
            loss.backward()
            self.policy.zero_grad()
        
        # Benchmark runs
        times = []
        for i in range(num_runs):
            start_time = time.time()
            loss, loss_dict = self.policy.forward(self.sample_batch)
            loss.backward()
            self.policy.zero_grad()
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        
        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min(times),
            "max_time": max(times),
            "times": times
        }
    
    def test_correctness(self) -> Dict[str, Any]:
        """Test that compiled and non-compiled versions produce the same results."""
        print("\n🧪 Testing correctness (compiled vs non-compiled)...")
        
        # Test without compilation
        self.policy.disable_torch_compile()
        self.policy.eval()
        
        with torch.no_grad():
            actions_no_compile = self.policy.predict_action_chunk(self.sample_batch)
            loss_no_compile, loss_dict_no_compile = self.policy.forward(self.sample_batch)
        
        # Test with compilation
        self.policy.enable_torch_compile(mode="default")
        self.policy.eval()
        
        with torch.no_grad():
            actions_with_compile = self.policy.predict_action_chunk(self.sample_batch)
            loss_with_compile, loss_dict_with_compile = self.policy.forward(self.sample_batch)
        
        # Compare results
        actions_diff = torch.abs(actions_no_compile - actions_with_compile).max().item()
        loss_diff = abs(loss_no_compile.item() - loss_with_compile.item())
        
        print(f"📊 Actions max difference: {actions_diff:.6f}")
        print(f"📊 Loss difference: {loss_diff:.6f}")
        
        return {
            "actions_diff": actions_diff,
            "loss_diff": loss_diff,
            "actions_match": actions_diff < 1e-5,
            "loss_match": loss_diff < 1e-5
        }
    
    def run_full_comparison(self) -> Dict[str, Any]:
        """Run complete comparison between compiled and non-compiled versions."""
        print("🔬 Starting comprehensive torch.compile comparison...")
        
        results = {}
        
        # Test correctness first
        results["correctness"] = self.test_correctness()
        
        # Benchmark inference
        results["inference_no_compile"] = self.benchmark_inference(use_compile=False, num_runs=20)
        results["inference_with_compile"] = self.benchmark_inference(use_compile=True, num_runs=20)
        
        # Benchmark training
        results["training_no_compile"] = self.benchmark_training(use_compile=False, num_runs=10)
        results["training_with_compile"] = self.benchmark_training(use_compile=True, num_runs=10)
        
        # Calculate speedups
        inference_speedup = (
            results["inference_no_compile"]["avg_time"] / 
            results["inference_with_compile"]["avg_time"]
        )
        training_speedup = (
            results["training_no_compile"]["avg_time"] / 
            results["training_with_compile"]["avg_time"]
        )
        
        results["speedups"] = {
            "inference": inference_speedup,
            "training": training_speedup
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print comparison results in a formatted way."""
        print("\n" + "="*80)
        print("📊 TORCH.COMPILE COMPARISON RESULTS")
        print("="*80)
        
        # Correctness
        print(f"\n🧪 CORRECTNESS:")
        print(f"   Actions match: {'✅' if results['correctness']['actions_match'] else '❌'}")
        print(f"   Loss match: {'✅' if results['correctness']['loss_match'] else '❌'}")
        print(f"   Max actions diff: {results['correctness']['actions_diff']:.2e}")
        print(f"   Loss diff: {results['correctness']['loss_diff']:.2e}")
        
        # Inference performance
        print(f"\n🚀 INFERENCE PERFORMANCE:")
        no_compile = results["inference_no_compile"]
        with_compile = results["inference_with_compile"]
        speedup = results["speedups"]["inference"]
        
        print(f"   Without compile: {no_compile['avg_time']:.4f}s ± {no_compile['std_time']:.4f}s")
        print(f"   With compile:    {with_compile['avg_time']:.4f}s ± {with_compile['std_time']:.4f}s")
        print(f"   Speedup:        {speedup:.2f}x {'🚀' if speedup > 1.0 else '🐌'}")
        
        # Training performance
        print(f"\n🏋️ TRAINING PERFORMANCE:")
        no_compile = results["training_no_compile"]
        with_compile = results["training_with_compile"]
        speedup = results["speedups"]["training"]
        
        print(f"   Without compile: {no_compile['avg_time']:.4f}s ± {no_compile['std_time']:.4f}s")
        print(f"   With compile:    {with_compile['avg_time']:.4f}s ± {with_compile['std_time']:.4f}s")
        print(f"   Speedup:        {speedup:.2f}x {'🚀' if speedup > 1.0 else '🐌'}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if results["correctness"]["actions_match"] and results["correctness"]["loss_match"]:
            print("   ✅ Correctness: PASSED")
        else:
            print("   ❌ Correctness: FAILED")
            
        if results["speedups"]["inference"] > 1.0 or results["speedups"]["training"] > 1.0:
            print("   🚀 Performance: IMPROVED")
        else:
            print("   🐌 Performance: NO IMPROVEMENT")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare ACT policy with and without torch.compile")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for testing")
    args = parser.parse_args()
    
    print(f"🔧 Running comparison on {args.device} with batch size {args.batch_size}")
    
    # Create comparison instance
    comparison = TorchCompileComparison(device=args.device, batch_size=args.batch_size)
    
    # Setup
    comparison.setup_policy_and_data()
    
    # Run comparison
    results = comparison.run_full_comparison()
    
    # Print results
    comparison.print_results(results)
    
    print("\n🎉 Comparison complete!")


if __name__ == "__main__":
    main()
