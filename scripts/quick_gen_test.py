"""
Quick generation test to see current model quality.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.architecture.model import GPTModel
from tokenizers import Tokenizer


def quick_test(config_path, checkpoint_path, tokenizer_path):
    """Quick generation test."""
    
    print("="*70)
    print("QUICK GENERATION TEST")
    print("="*70)
    
    # Load
    print(f"\nConfig: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Tokenizer: {tokenizer_path}")
    
    try:
        model = GPTModel.from_config_file(config_path)
        
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
                if 'step' in state:
                    print(f"Checkpoint step: {state['step']}")
                if 'loss' in state:
                    print(f"Checkpoint loss: {state['loss']:.4f}")
            else:
                model.load_state_dict(state)
        else:
            print("⚠️  No checkpoint found - using random weights")
        
        model.eval()
        
        tokenizer = Tokenizer.from_file(tokenizer_path)
        device = "cpu"
        model.to(device)
        
        # Test prompts
        prompts = [
            "Once upon a time",
            "def quick_sort(",
            "import torch",
        ]
        
        print("\n" + "="*70)
        print("GENERATIONS")
        print("="*70)
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            
            encoded = tokenizer.encode(prompt).ids
            generated = encoded.copy()
            
            for _ in range(20):
                x = torch.tensor([generated], dtype=torch.long).to(device)
                with torch.no_grad():
                    logits = model(x)
                    next_id = logits[0, -1].argmax().item()
                
                generated.append(next_id)
            
            output = tokenizer.decode(generated)
            print(f"Output: {output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    config_path = str(PROJECT_ROOT / "model/config/gpt_tiny_config.json")
    checkpoint_path = str(PROJECT_ROOT / "pretraining/checkpoints/gpt-small-ckpt.pt")
    tokenizer_path = str(PROJECT_ROOT / "tokenizer/tokenizer_8k.json")
    
    quick_test(config_path, checkpoint_path, tokenizer_path)

