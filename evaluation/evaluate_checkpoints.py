"""
Evaluate all checkpoints to track improvement over training.
"""

import os
import sys
from pathlib import Path
import json
import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.code_tests import run_code_benchmark, save_results


def find_checkpoints(checkpoint_dir="pretraining/checkpoints"):
    """Find all checkpoint files."""
    checkpoint_dir = Path(PROJECT_ROOT) / checkpoint_dir
    
    if not checkpoint_dir.exists():
        print(f"⚠️  Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    # Find all checkpoint files
    checkpoints = []
    
    for ckpt_file in checkpoint_dir.glob("*.pt"):
        # Extract step number if present
        name = ckpt_file.stem
        
        if "step_" in name:
            try:
                step = int(name.split("step_")[-1])
            except:
                step = 0
        else:
            step = 999999  # Put non-step checkpoints at end
        
        checkpoints.append({
            'path': str(ckpt_file),
            'name': name,
            'step': step
        })
    
    # Sort by step
    checkpoints.sort(key=lambda x: x['step'])
    
    return checkpoints


def evaluate_all_checkpoints(config_path="model/config/gpt_tiny_config.json"):
    """Evaluate all available checkpoints."""
    
    print("="*70)
    print("CHECKPOINT EVALUATION")
    print("="*70)
    
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("\n⚠️  No checkpoints found to evaluate!")
        print("   Train a model first using: python pretraining/train_improved.py")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoint(s) to evaluate\n")
    
    all_results = []
    
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"\n{'='*70}")
        print(f"Evaluating {i}/{len(checkpoints)}: {ckpt['name']}")
        print(f"{'='*70}")
        
        try:
            results = run_code_benchmark(
                config_path=config_path,
                checkpoint_path=ckpt['path']
            )
            
            results['checkpoint_info'] = ckpt
            all_results.append(results)
            
            # Save individual results
            output_path = f"evaluation/results/eval_{ckpt['name']}.json"
            save_results(results, output_path)
            
        except Exception as e:
            print(f"❌ Error evaluating {ckpt['name']}: {e}")
            continue
    
    # Create comparison report
    print("\n" + "="*70)
    print("COMPARISON REPORT")
    print("="*70)
    
    if all_results:
        print(f"\n{'Checkpoint':<30} {'Overall':<10} {'Syntax':<10} {'Generation':<12} {'Perplexity'}")
        print("-"*70)
        
        for res in all_results:
            name = res['checkpoint_info']['name'][:28]
            summary = res['summary']
            ppl = res['code_perplexity']['average_perplexity']
            
            print(f"{name:<30} "
                  f"{summary['overall_score']:>8.1%}  "
                  f"{summary['syntax_score']:>8.1%}  "
                  f"{summary['generation_score']:>10.1%}  "
                  f"{ppl:>10.2f}")
        
        # Save comparison
        comparison_path = Path(PROJECT_ROOT) / "evaluation" / "results" / "checkpoint_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n💾 Comparison saved to: {comparison_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="model/config/gpt_tiny_config.json",
                        help="Model config path")
    
    args = parser.parse_args()
    
    evaluate_all_checkpoints(config_path=args.config)

