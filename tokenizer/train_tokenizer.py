from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
import os

def main():
    corpus_path = "../data/raw/tokenizer_corpus.txt"

    if not os.path.exists(corpus_path):
        raise FileNotFoundError("Corpus not found at: " + corpus_path)

    # Initialise tokenizer model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()

    # Trainer configuration
    trainer = BpeTrainer(
        vocab_size=8000,
        min_frequency=2,
        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[BOS]",
            "[EOS]"
        ]
    )

    print("Training tokenizer on corpus...")
    tokenizer.train([corpus_path], trainer)

    # Save the output files
    output_path = "./tokenizer_8k.json"
    tokenizer.save(output_path)

    print(f"Tokenizer saved to {output_path}")

if __name__ == "__main__":
    main()
