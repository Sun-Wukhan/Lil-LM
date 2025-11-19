import os
import glob
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TextDataset(Dataset):
    """
    Loads text from a corpus file or directory,
    concatenates them into one giant token sequence,
    and slices it into training (x, y) pairs for GPT.

    If using corpus_file, it loads that single file.
    If using root_path, it recursively finds all .txt files.
    """

    def __init__(self, root_path=None, tokenizer_path=None, block_size=128, corpus_file=None):
        self.block_size = block_size

        # --- Load tokenizer once ---
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # --- Load text from corpus_file or directory ---
        if corpus_file is not None:
            # Load from single corpus file
            print(f"[Dataset] Loading from corpus file: {corpus_file}")
            try:
                with open(corpus_file, "r", encoding="utf-8", errors="ignore") as f:
                    full_text = f.read()
                print(f"[Dataset] Loaded corpus file successfully")
            except Exception as e:
                print(f"[Dataset] ERROR: Failed to read corpus file: {e}")
                full_text = "Hello world. This is dummy text for testing."
        else:
            # Original behavior: collect all .txt files from directory
            if root_path is None:
                raise ValueError("Either corpus_file or root_path must be provided")
            
            file_paths = glob.glob(
                os.path.join(root_path, "**/*.txt"),
                recursive=True
            )

            # === NO FILES FOUND → Dummy text fallback ===
            if len(file_paths) == 0:
                print("[Dataset] WARNING: No .txt files found. Using dummy text.")
                full_text = "Hello world. This is dummy text for testing."
            else:
                print(f"[Dataset] Found {len(file_paths)} text files.")

                # --- Read + concat all text files ---
                full_text = ""
                for fp in file_paths:
                    try:
                        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                            full_text += f.read() + "\n"
                    except Exception as e:
                        print(f"[Warning] Failed to read {fp}: {e}")

        # --- Tokenize everything at once ---
        print("[Dataset] Tokenizing corpus...")
        enc = self.tokenizer.encode(full_text)
        self.tokens = torch.tensor(enc.ids, dtype=torch.long)

        # Total number of possible (x,y) samples
        self.num_chunks = len(self.tokens) - self.block_size - 1

        print(f"[Dataset] Total tokens: {len(self.tokens)}")
        print(f"[Dataset] Training samples: {self.num_chunks}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        """
        Returns:
            x = input tokens (block_size)
            y = next-token targets (block_size)
        """
        chunk = self.tokens[idx: idx + self.block_size + 1]

        x = chunk[:-1]
        y = chunk[1:]
        return x, y
