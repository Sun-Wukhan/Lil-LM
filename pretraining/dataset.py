import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TextDataset(Dataset):
    """
    Converts a text file into token ID sequences for GPT training.
    """

    def __init__(self, path, tokenizer_path, block_size=128):
        self.block_size = block_size

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Load raw text
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Encode whole corpus → list of token IDs
        enc = self.tokenizer.encode(text)
        self.tokens = torch.tensor(enc.ids, dtype=torch.long)

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]

        x = chunk[:-1]  # input tokens
        y = chunk[1:]   # the next token the model should predict

        return x, y
