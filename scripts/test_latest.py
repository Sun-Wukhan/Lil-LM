"""
Test the latest checkpoint from current training.
Uses the correct config (gpt_tiny) and latest best checkpoint.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.architecture.model import GPTModel
from tokenizers import Tokenizer


def find_latest_checkpoint():
    """Find the most recent best checkpoint."""
    ckpt_dir = PROJECT_ROOT / "pretraining" / "checkpoints"
    
    # Find all best_step files
    best_ckpts = sorted(ckpt_dir.glob("best_step_*.pt"), 
                        key=lambda x: int(x.stem.split('_')[-1]))
    
    if best_ckpts:
        return best_ckpts[-1]
    
    # Fallback to any checkpoint
    all_ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime)
    if all_ckpts:
        return all_ckpts[-1]
    
    return None


def test_generation():
    """Test generation with latest checkpoint."""
    
    print("="*70)
    print("TESTING LATEST CHECKPOINT")
    print("="*70)
    
    # Config and tokenizer
    config_path = PROJECT_ROOT / "model/config/gpt_tiny_config.json"
    tokenizer_path = PROJECT_ROOT / "tokenizer/tokenizer_8k.json"
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint()
    
    if not checkpoint_path:
        print("❌ No checkpoints found!")
        return
    
    print(f"\n✓ Config: {config_path}")
    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ Tokenizer: {tokenizer_path}")
    
    # Load model
    print("\nLoading model...")
    model = GPTModel.from_config_file(str(config_path))
    
    # Load checkpoint
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
        print(f"✓ Loaded from step: {state.get('step', 'unknown')}")
        print(f"✓ Training loss: {state.get('loss', 'unknown'):.4f}" if 'loss' in state else "")
    else:
        model.load_state_dict(state)
    
    model.eval()
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    device = "cpu"
    model.to(device)
    
    # Test prompts
    prompts = [
        "Once upon a time",
        "def quick_sort(collection):",
        "import torch\nfrom torch import",
        "The quick brown",
    ]
    
    print("\n" + "="*70)
    print("GENERATION TESTS")
    print("="*70)
    
    for prompt in prompts:
        print(f"\n{'─'*70}")
        print(f"Prompt: {repr(prompt)}")
        print(f"{'─'*70}")
        
        # Encode
        encoded = tokenizer.encode(prompt).ids
        generated = encoded.copy()
        
        # Generate 30 tokens
        for _ in range(30):
            x = torch.tensor([generated], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(x)
                next_id = logits[0, -1].argmax().item()
            
            generated.append(next_id)
        
        # Decode
        output = tokenizer.decode(generated)
        print(f"Output:\n{output}")
    
    print("\n" + "="*70)
    print("✅ Testing complete!")
    print("="*70)


if __name__ == "__main__":
    test_generation()

