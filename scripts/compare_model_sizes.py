"""
Compare different model configurations to help choose the right size.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.architecture.model import GPTModel


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory(num_params, batch_size=8, seq_len=128):
    """
    Rough memory estimate for training.
    
    Components:
    - Model parameters (4 bytes per param for fp32)
    - Gradients (4 bytes per param)
    - Optimizer states (8 bytes per param for Adam)
    - Activations (depends on batch size and sequence length)
    """
    # Model + gradients + optimizer states
    param_memory = num_params * 4  # parameters
    grad_memory = num_params * 4   # gradients
    optim_memory = num_params * 8  # Adam states (2x momentum)
    
    # Rough activation estimate (very approximate)
    activation_memory = batch_size * seq_len * 512 * 4  # hidden_size * 4 bytes
    
    total_bytes = param_memory + grad_memory + optim_memory + activation_memory
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_mb / 1024
    
    return total_mb, total_gb


def main():
    configs = [
        ("GPT Tiny (6 layers)", "model/config/gpt_tiny_config.json"),
        ("GPT Small (12 layers)", "model/config/gpt_small_config.json"),
    ]
    
    print("="*70)
    print("MODEL SIZE COMPARISON")
    print("="*70)
    
    for name, config_path in configs:
        full_path = PROJECT_ROOT / config_path
        
        if not full_path.exists():
            print(f"\n⚠️  {name}: Config not found")
            continue
        
        # Load model
        model = GPTModel.from_config_file(str(full_path))
        total_params, trainable_params = count_parameters(model)
        
        # Memory estimates
        mem_mb_8, mem_gb_8 = estimate_memory(total_params, batch_size=8)
        mem_mb_16, mem_gb_16 = estimate_memory(total_params, batch_size=16)
        
        print(f"\n{name}")
        print(f"  Config: {config_path}")
        print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Memory estimate:")
        print(f"    Batch size 8:  ~{mem_mb_8:.0f} MB ({mem_gb_8:.2f} GB)")
        print(f"    Batch size 16: ~{mem_mb_16:.0f} MB ({mem_gb_16:.2f} GB)")
        
        # Training time estimate (very rough)
        # Assuming ~10 samples/sec on MacBook CPU
        steps_20k = 20000
        samples_per_sec = 10
        time_sec = (steps_20k * 8) / samples_per_sec
        time_min = time_sec / 60
        time_hr = time_min / 60
        
        print(f"  Estimated training time (20K steps, batch 8):")
        print(f"    ~{time_min:.0f} minutes (~{time_hr:.1f} hours)")
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("For MacBook training:")
    print("  ✓ Use GPT Tiny (6 layers) for faster iteration")
    print("  ✓ Batch size 8 is safe for most MacBooks")
    print("  ✓ Expected ~30-60 min for 20K steps on CPU")


if __name__ == "__main__":
    main()

