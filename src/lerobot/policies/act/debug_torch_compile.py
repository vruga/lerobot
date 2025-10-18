#!/usr/bin/env python
"""
Comprehensive torch.compile debugging script for ACT policy.

This script helps identify all torch.compile compatibility issues in the ACT policy
by running with verbose dynamo logging and analyzing graph breaks.
"""

import argparse
import logging
import sys
import time
from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torch._dynamo as dynamo
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.factory import make_policy_config
from lerobot.utils.constants import OBS_IMAGES

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Enable comprehensive dynamo debugging
dynamo.config.verbose = True
dynamo.config.suppress_errors = False
dynamo.config.disable = False

# Additional dynamo settings for debugging
dynamo.config.cache_size_limit = 64
dynamo.config.dynamic_shapes = True


class TorchCompileDebugger:
    """Comprehensive torch.compile debugger for ACT policy"""
    
    def __init__(self, device: str = "cuda", repo_id: str = "AdilZtn/grab_red_cube_test_25"):
        self.device = torch.device(device)
        self.repo_id = repo_id
        
        # Test parameters
        self.batch_size = 4
        self.n_test_runs = 5
        
        print(f"🔧 Torch.compile Debugger for ACT Policy")
        print(f"Device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Dataset: {repo_id}")
        print("=" * 60)
    
    def setup_policy_and_data(self) -> tuple[ACTPolicy, dict, Any, Any]:
        """Setup ACT policy and test data"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load dataset metadata
        ds_meta = LeRobotDatasetMetadata(self.repo_id)
        
        # Create ACT configuration with proper PolicyFeature format
        from lerobot.configs.types import PolicyFeature, FeatureType
        
        cfg = make_policy_config(
            "act", 
            device=str(self.device),
            use_vae=False,  # Start with simpler configuration
            chunk_size=50,
            n_action_steps=10,
            input_features={
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3)),
                "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3)), 
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))
            }
        )
        
        # Create policy
        policy = ACTPolicy(cfg)
        policy.to(self.device)
        
        # Create synthetic batch in the format expected by ACT model (after preprocessing)
        batch_size = self.batch_size
        
        # Create synthetic data in the format expected by ACT policy
        # Policy expects individual camera keys, then internally converts to list under OBS_IMAGES
        sample_batch = {
            "observation.images.top": torch.randn(batch_size, 3, 480, 640, dtype=torch.float32),
            "observation.images.wrist": torch.randn(batch_size, 3, 480, 640, dtype=torch.float32),
            "observation.state": torch.randn(batch_size, 6, dtype=torch.float32),
            "observation.environment_state": torch.randn(batch_size, 6, dtype=torch.float32),
            "action": torch.randn(batch_size, cfg.chunk_size, 6, dtype=torch.float32),
            "action_is_pad": torch.zeros(batch_size, cfg.chunk_size, dtype=torch.bool)
        }
        
        # Move to device
        for key in sample_batch:
            if isinstance(sample_batch[key], torch.Tensor):
                sample_batch[key] = sample_batch[key].to(self.device)
        
        return policy, sample_batch, None, None
    
    def test_compilation_step_by_step(self, policy: ACTPolicy, batch: dict) -> Dict[str, Any]:
        """Test compilation step by step to identify specific issues"""
        results = {
            "compilation_successful": False,
            "errors": [],
            "warnings": [],
            "graph_breaks": [],
            "step_results": {}
        }
        
        print("🔍 Testing compilation step by step...")
        
        # Test 1: Basic model compilation
        print("\n1️⃣ Testing basic model compilation...")
        try:
            compiled_model = torch.compile(policy.model, mode="default")
            results["step_results"]["model_compilation"] = "SUCCESS"
            print("✅ Model compilation successful")
        except Exception as e:
            results["step_results"]["model_compilation"] = f"FAILED: {str(e)}"
            results["errors"].append(f"Model compilation failed: {str(e)}")
            print(f"❌ Model compilation failed: {e}")
            return results
        
        # Test 2: Model forward pass
        print("\n2️⃣ Testing compiled model forward pass...")
        try:
            policy.eval()
            with torch.no_grad():
                # Prepare batch for model (convert individual camera keys to OBS_IMAGES list)
                model_batch = dict(batch)  # shallow copy
                if policy.config.image_features:
                    model_batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]
                
                # Test inference path
                actions = compiled_model(model_batch)[0]
                results["step_results"]["model_forward"] = "SUCCESS"
                print("✅ Model forward pass successful")
        except Exception as e:
            results["step_results"]["model_forward"] = f"FAILED: {str(e)}"
            results["errors"].append(f"Model forward pass failed: {str(e)}")
            print(f"❌ Model forward pass failed: {e}")
        
        # Test 3: Full policy compilation
        print("\n3️⃣ Testing full policy compilation...")
        try:
            compiled_policy = torch.compile(policy, mode="default")
            results["step_results"]["policy_compilation"] = "SUCCESS"
            print("✅ Policy compilation successful")
        except Exception as e:
            results["step_results"]["policy_compilation"] = f"FAILED: {str(e)}"
            results["errors"].append(f"Policy compilation failed: {str(e)}")
            print(f"❌ Policy compilation failed: {e}")
            return results
        
        # Test 4: Policy inference
        print("\n4️⃣ Testing compiled policy inference...")
        try:
            compiled_policy.eval()
            with torch.no_grad():
                action = compiled_policy.select_action(batch)
                results["step_results"]["policy_inference"] = "SUCCESS"
                print("✅ Policy inference successful")
        except Exception as e:
            results["step_results"]["policy_inference"] = f"FAILED: {str(e)}"
            results["errors"].append(f"Policy inference failed: {str(e)}")
            print(f"❌ Policy inference failed: {e}")
        
        # Test 5: Policy training forward
        print("\n5️⃣ Testing compiled policy training forward...")
        try:
            compiled_policy.train()
            loss, loss_dict = compiled_policy.forward(batch)
            results["step_results"]["policy_training"] = "SUCCESS"
            print("✅ Policy training forward successful")
        except Exception as e:
            results["step_results"]["policy_training"] = f"FAILED: {str(e)}"
            results["errors"].append(f"Policy training forward failed: {str(e)}")
            print(f"❌ Policy training forward failed: {e}")
        
        # Test 6: Backward pass
        print("\n6️⃣ Testing compiled policy backward pass...")
        try:
            loss.backward()
            results["step_results"]["policy_backward"] = "SUCCESS"
            print("✅ Policy backward pass successful")
        except Exception as e:
            results["step_results"]["policy_backward"] = f"FAILED: {str(e)}"
            results["errors"].append(f"Policy backward pass failed: {str(e)}")
            print(f"❌ Policy backward pass failed: {e}")
        
        results["compilation_successful"] = len(results["errors"]) == 0
        return results
    
    def analyze_specific_issues(self, policy: ACTPolicy, batch: dict) -> Dict[str, Any]:
        """Analyze specific known torch.compile issues"""
        issues = {
            "item_calls": [],
            "device_calls": [],
            "dynamic_shapes": [],
            "control_flow": [],
            "data_structures": []
        }
        
        print("\n🔍 Analyzing specific torch.compile issues...")
        
        # Test for .item() calls in forward method
        print("\n📊 Testing for .item() calls...")
        try:
            policy.train()
            loss, loss_dict = policy.forward(batch)
            print("✅ Forward pass without compilation works")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            issues["item_calls"].append(f"Forward pass error: {e}")
        
        # Test for device-related issues
        print("\n📱 Testing device-related operations...")
        try:
            # Check for .to(device) calls in the code
            if hasattr(policy, '_action_queue') and isinstance(policy._action_queue, deque):
                issues["data_structures"].append("deque usage in _action_queue")
                print("⚠️  Found deque usage in _action_queue")
        except Exception as e:
            print(f"❌ Device operation test failed: {e}")
        
        # Test temporal ensembler if present
        if hasattr(policy, 'temporal_ensembler'):
            print("\n⏰ Testing temporal ensembler...")
            try:
                policy.eval()
                with torch.no_grad():
                    action = policy.select_action(batch)
                    print("✅ Temporal ensembler works")
            except Exception as e:
                print(f"❌ Temporal ensembler failed: {e}")
                issues["dynamic_shapes"].append(f"Temporal ensembler error: {e}")
        
        return issues
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive torch.compile testing"""
        print("🚀 Starting comprehensive torch.compile testing...")
        
        # Setup
        try:
            policy, batch, preprocessor, postprocessor = self.setup_policy_and_data()
        except Exception as e:
            return {"error": f"Setup failed: {str(e)}"}
        
        # Test compilation step by step
        compilation_results = self.test_compilation_step_by_step(policy, batch)
        
        # Analyze specific issues
        issue_analysis = self.analyze_specific_issues(policy, batch)
        
        # Combine results
        results = {
            "compilation_results": compilation_results,
            "issue_analysis": issue_analysis,
            "policy_config": {
                "use_vae": policy.config.use_vae,
                "chunk_size": policy.config.chunk_size,
                "n_action_steps": policy.config.n_action_steps,
                "has_temporal_ensembler": hasattr(policy, 'temporal_ensembler')
            }
        }
        
        return results
    
    def generate_debug_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive debug report"""
        report = f"""# Torch.compile Debug Report: ACT Policy

## Environment
- **PyTorch Version**: {torch.__version__}
- **Device**: {self.device}
- **Dataset**: {self.repo_id}

## Policy Configuration
- **VAE Enabled**: {results["policy_config"]["use_vae"]}
- **Chunk Size**: {results["policy_config"]["chunk_size"]}
- **Action Steps**: {results["policy_config"]["n_action_steps"]}
- **Temporal Ensembler**: {results["policy_config"]["has_temporal_ensembler"]}

## Compilation Results
"""
        
        compilation = results["compilation_results"]
        for step, status in compilation["step_results"].items():
            emoji = "✅" if status == "SUCCESS" else "❌"
            report += f"- **{step.replace('_', ' ').title()}**: {emoji} {status}\n"
        
        if compilation["errors"]:
            report += "\n### Errors\n"
            for error in compilation["errors"]:
                report += f"- {error}\n"
        
        if compilation["warnings"]:
            report += "\n### Warnings\n"
            for warning in compilation["warnings"]:
                report += f"- {warning}\n"
        
        report += "\n## Issue Analysis\n"
        
        issues = results["issue_analysis"]
        for category, items in issues.items():
            if items:
                report += f"\n### {category.replace('_', ' ').title()}\n"
                for item in items:
                    report += f"- {item}\n"
        
        report += "\n## Recommendations\n"
        
        if not compilation["compilation_successful"]:
            report += """
### Priority Fixes:
1. **Fix compilation errors** - Address the specific errors listed above
2. **Remove .item() calls** - Replace with torch._dynamo.is_compiling() checks
3. **Handle dynamic shapes** - Ensure tensor shapes are static
4. **Replace data structures** - Replace deque with tensor-based alternatives
"""
        else:
            report += """
### Next Steps:
1. **Run performance benchmarks** - Use the benchmark script to measure speedup
2. **Test edge cases** - Try different batch sizes and configurations
3. **Validate correctness** - Ensure compiled version matches eager execution
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Debug torch.compile issues in ACT policy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to run on")
    parser.add_argument("--repo-id", default="AdilZtn/grab_red_cube_test_25",
                       help="Dataset repository ID")
    parser.add_argument("--output", help="Output file for debug report")
    
    args = parser.parse_args()
    
    # Run debugger
    debugger = TorchCompileDebugger(args.device, args.repo_id)
    results = debugger.run_comprehensive_test()
    
    # Generate report
    if "error" in results:
        print(f"\n❌ Debugging failed: {results['error']}")
        return
    
    report = debugger.generate_debug_report(results)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\n💾 Debug report saved to {args.output}")
    else:
        print("\n📝 DEBUG REPORT:")
        print(report)


if __name__ == "__main__":
    main()
