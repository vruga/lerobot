
"""
Specific debugging script for ACT policy torch.compile issues
Run this to identify exact graph breaks in ACT
"""

import torch
import torch._dynamo as dynamo
import logging
from typing import Dict, Any, List
import traceback

class ACTCompileDebugger:
    """Specific debugger for ACT policy compilation"""
    
    def __init__(self):
        self.graph_breaks = []
        self.act_specific_issues = {
            'vae_branching': [],
            'attention_issues': [],
            'chunk_size_problems': [],
            'transformer_issues': [],
            'other': []
        }
        
    def setup_verbose_debugging(self):
        """Setup maximum verbosity for ACT debugging"""
        
        # Reset dynamo state
        dynamo.reset()
        
        # Enable all debugging
        dynamo.config.verbose = True
        dynamo.config.log_level = logging.DEBUG
        dynamo.config.suppress_errors = True
        dynamo.config.capture_scalar_outputs = True
        
        # Custom graph break handler for ACT
        def act_graph_break_handler(guard_failure):
            break_info = {
                'reason': str(guard_failure.reason),
                'code': getattr(guard_failure, 'code_context', None),
                'user_stack': getattr(guard_failure, 'user_stack', None)
            }
            
            self.graph_breaks.append(break_info)
            self._categorize_act_issue(break_info)
            
            print(f"🔴 ACT GRAPH BREAK: {break_info['reason']}")
        
        dynamo.config.guard_fail_hook = act_graph_break_handler
    
    def _categorize_act_issue(self, break_info):
        """Categorize ACT-specific issues"""
        reason = break_info['reason'].lower()
        
        if any(keyword in reason for keyword in ['vae', 'encode', 'decode']):
            self.act_specific_issues['vae_branching'].append(break_info)
        elif any(keyword in reason for keyword in ['attention', 'attn', 'mask']):
            self.act_specific_issues['attention_issues'].append(break_info)
        elif any(keyword in reason for keyword in ['chunk', 'size', 'shape']):
            self.act_specific_issues['chunk_size_problems'].append(break_info)
        elif any(keyword in reason for keyword in ['transformer', 'layer', 'block']):
            self.act_specific_issues['transformer_issues'].append(break_info)
        else:
            self.act_specific_issues['other'].append(break_info)
    
    def debug_act_policy(self):
        """Debug ACT policy specifically"""
        print("🎯 Debugging ACT Policy for torch.compile...")
        
        try:
            # Import ACT components
            from lerobot.policies.act.policy_act import ACTPolicy
            from lerobot.policies.factory import make_policy_config
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            
            # Test both VAE and non-VAE configurations
            configs_to_test = [
                {"use_vae": False, "chunk_size": 100},
                {"use_vae": True, "chunk_size": 100}
            ]
            
            for i, config in enumerate(configs_to_test):
                print(f"\n📋 Testing ACT Configuration {i+1}: {config}")
                self._test_act_config(config)
            
        except Exception as e:
            print(f"❌ ACT Debug Error: {e}")
            traceback.print_exc()
        
        # Analyze results
        self._analyze_act_issues()
    
    def _test_act_config(self, config):
        """Test specific ACT configuration"""
        
        # Reset for each test
        self.graph_breaks = []
        for category in self.act_specific_issues:
            self.act_specific_issues[category] = []
        
        try:
            # Setup ACT policy
            ds_meta = LeRobotDatasetMetadata("AdilZtn/grab_red_cube_test_25")
            cfg = make_policy_config("act", device="cuda", **config)
            
            policy = ACTPolicy(cfg, ds_meta=ds_meta)
            policy.cuda()
            policy.eval()
            
            # Create sample inputs matching ACT's expectations
            sample_batch = self._create_act_sample_input(cfg)
            
            print(f"   📊 Input shapes:")
            for key, value in sample_batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape}")
            
            # Test uncompiled version first
            print("   🧪 Testing uncompiled...")
            with torch.no_grad():
                original_action = policy.select_action(sample_batch)
            print(f"   ✅ Uncompiled action shape: {original_action.shape}")
            
            # Setup debugging
            self.setup_verbose_debugging()
            
            # Attempt compilation
            print("   🔧 Attempting torch.compile...")
            compiled_policy = torch.compile(policy, mode="default")
            
            # Test compiled version
            print("   🧪 Testing compiled...")
            with torch.no_grad():
                compiled_action = compiled_policy.select_action(sample_batch)
            print(f"   ✅ Compiled action shape: {compiled_action.shape}")
            
            # Check correctness
            diff = torch.abs(original_action - compiled_action).max().item()
            print(f"   📊 Action difference: {diff:.2e}")
            
            if len(self.graph_breaks) == 0:
                print(f"   🎉 SUCCESS: No graph breaks for config {config}")
            else:
                print(f"   ⚠️  Found {len(self.graph_breaks)} graph breaks")
                
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
    
    def _create_act_sample_input(self, cfg):
        """Create sample input matching ACT policy expectations"""
        
        batch_size = 8
        
        # ACT typically expects these observation keys
        sample_input = {
            'observation.image': torch.randn(batch_size, 3, 224, 224).cuda(),
            'observation.state': torch.randn(batch_size, 14).cuda(),  # Typical robot state dim
        }
        
        # Add action if this is for training
        if hasattr(cfg, 'chunk_size'):
            sample_input['action'] = torch.randn(batch_size, cfg.chunk_size, 7).cuda()  # 7D actions
        
        return sample_input
    
    def _analyze_act_issues(self):
        """Provide ACT-specific analysis and fixes"""
        
        print("\n" + "="*60)
        print("🎯 ACT POLICY COMPILATION ANALYSIS")
        print("="*60)
        
        total_issues = sum(len(issues) for issues in self.act_specific_issues.values())
        
        if total_issues == 0:
            print("🎉 SUCCESS: ACT policy is torch.compile compatible!")
            return
        
        print(f"🔍 Found {total_issues} ACT-specific issues:")
        
        # Analyze each category
        for category, issues in self.act_specific_issues.items():
            if issues:
                print(f"\n🏷️  {category.upper()} ({len(issues)} issues):")
                for i, issue in enumerate(issues[:3]):  # Show first 3
                    print(f"   {i+1}. {issue['reason']}")
                if len(issues) > 3:
                    print(f"   ... and {len(issues) - 3} more")
        
        # Provide ACT-specific fixes
        self._suggest_act_fixes()
    
    def _suggest_act_fixes(self):
        """Suggest ACT-specific fixes"""
        
        print("\n🛠️  ACT-SPECIFIC FIXES:")
        
        if self.act_specific_issues['vae_branching']:
            print("\n1. VAE BRANCHING ISSUES:")
            print("   - Set use_vae as a fixed class attribute, not runtime decision")
            print("   - Create separate ACT classes for VAE vs non-VAE")
            print("   - Remove conditional VAE encoding/decoding paths")
        
        if self.act_specific_issues['attention_issues']:
            print("\n2. ATTENTION MASK ISSUES:")
            print("   - Use fixed-size attention masks with padding")
            print("   - Replace dynamic masking with vectorized operations")
            print("   - Avoid .item() calls in attention computations")
        
        if self.act_specific_issues['chunk_size_problems']:
            print("\n3. CHUNK SIZE ISSUES:")
            print("   - Fix chunk_size at class initialization")
            print("   - Always output exactly chunk_size actions")
            print("   - Use padding/truncation for variable-length sequences")
        
        if self.act_specific_issues['transformer_issues']:
            print("\n4. TRANSFORMER LAYER ISSUES:")
            print("   - Check for dynamic layer indexing")
            print("   - Ensure consistent tensor shapes through transformer")
            print("   - Fix any conditional layer skipping")
        
        print("\n📝 IMPLEMENTATION PRIORITY:")
        print("1. Fix VAE branching (highest impact)")
        print("2. Fix chunk size handling")  
        print("3. Fix attention mechanisms")
        print("4. Optimize transformer layers")

def main():
    debugger = ACTCompileDebugger()
    debugger.debug_act_policy()

if __name__ == "__main__":
    main()
