"""
Improved tokenizer training with proper spacing handling.
This fixes the over-segmentation issue causing "O n ce" instead of "Once".
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
import os
from pathlib import Path

def main():
    # Use the new mixed corpus
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    corpus_path = project_root / "data" / "processed" / "mixed_corpus.txt"

    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at: {corpus_path}")
        print("Run scripts/create_mixed_corpus.py first!")
        return

    print(f"Training tokenizer on: {corpus_path}")

    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Use ByteLevel pre-tokenizer - this handles spacing properly!
    # It adds a special character (Ġ) to represent spaces, preventing the "O n ce" problem
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Set normalizer
    tokenizer.normalizer = NFKC()
    
    # Set decoder to handle byte-level encoding
    tokenizer.decoder = ByteLevelDecoder()

    # Trainer configuration - increased vocab for better code coverage
    trainer = BpeTrainer(
        vocab_size=8000,
        min_frequency=2,
        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[BOS]",
            "[EOS]"
        ],
        show_progress=True
    )

    print("Training tokenizer...")
    tokenizer.train([str(corpus_path)], trainer)

    # Save the tokenizer
    output_path = script_dir / "tokenizer_8k_improved.json"
    tokenizer.save(str(output_path))

    print(f"\n✓ Tokenizer saved to: {output_path}")
    
    # Quick test
    print("\n" + "="*60)
    print("QUICK TEST")
    print("="*60)
    
    test_samples = [
        "Once upon a time in Toronto,",
        "def quick_sort(collection: list) -> list:",
        "import torch"
    ]
    
    for sample in test_samples:
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded.ids)
        print(f"\nInput:   {sample}")
        print(f"Tokens:  {encoded.tokens[:10]}{'...' if len(encoded.tokens) > 10 else ''}")
        print(f"Decoded: {decoded}")
        print(f"Match: {'✓' if decoded == sample else '✗'}")

if __name__ == "__main__":
    main()

