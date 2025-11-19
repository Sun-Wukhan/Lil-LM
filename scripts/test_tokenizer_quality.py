"""
Test tokenizer quality on code samples to check for over-segmentation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from tokenizers import Tokenizer

def test_tokenizer():
    """Test tokenizer on various code and text samples."""
    
    tokenizer_path = project_root / "tokenizer" / "tokenizer_8k.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Test samples
    test_samples = [
        # Code samples
        "def quick_sort(collection: list) -> list:",
        "return [item for item in collection if item <= pivot]",
        "import torch",
        "from torch import nn",
        "class GPTModel(nn.Module):",
        "self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)",
        
        # Common Python keywords
        "if __name__ == '__main__':",
        "for i in range(10):",
        "while True:",
        
        # Text samples  
        "Once upon a time in Toronto,",
        "Hello world. This is a test.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    print("=" * 70)
    print("TOKENIZER QUALITY TEST")
    print("=" * 70)
    
    total_chars = 0
    total_tokens = 0
    
    for sample in test_samples:
        encoded = tokenizer.encode(sample)
        tokens = encoded.tokens
        ids = encoded.ids
        
        # Calculate compression ratio
        chars = len(sample)
        num_tokens = len(tokens)
        ratio = chars / num_tokens if num_tokens > 0 else 0
        
        total_chars += chars
        total_tokens += num_tokens
        
        print(f"\nInput: {sample}")
        print(f"Tokens ({num_tokens}): {tokens}")
        print(f"Compression ratio: {ratio:.2f} chars/token")
        
        # Decode back
        decoded = tokenizer.decode(ids)
        if decoded != sample:
            print(f"⚠️  Decode mismatch!")
            print(f"   Decoded: {decoded}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_ratio = total_chars / total_tokens
    print(f"Average compression: {avg_ratio:.2f} chars/token")
    print(f"Total tokens: {total_tokens}")
    print(f"Total chars: {total_chars}")
    
    # Guidance
    print("\n📊 Analysis:")
    if avg_ratio < 2.0:
        print("   ⚠️  LOW compression - tokenizer may be over-segmenting")
        print("   Consider retraining with larger vocab or more merge operations")
    elif avg_ratio < 3.0:
        print("   ⚠️  MODERATE compression - acceptable but could be better")
    else:
        print("   ✓ GOOD compression - tokenizer is working well")
    
    # Test specific problematic pattern from generation output
    print("\n" + "=" * 70)
    print("TESTING PROBLEMATIC PATTERN FROM GENERATION")
    print("=" * 70)
    
    problematic = "Once upon a time in Toronto,"
    encoded = tokenizer.encode(problematic)
    print(f"Input: {problematic}")
    print(f"Tokens: {encoded.tokens}")
    print(f"Decoded: {tokenizer.decode(encoded.ids)}")
    
    if any(len(t) == 1 and t.isalpha() for t in encoded.tokens):
        print("⚠️  Single character tokens detected - this explains the generation issue!")
        print("   The model is predicting spaces between characters.")

if __name__ == "__main__":
    test_tokenizer()

