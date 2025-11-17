# --- Ensure project root is on sys.path no matter where this script is run ---
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Using PROJECT_ROOT:", PROJECT_ROOT)
# -----------------------------------------------------------------------------

import json
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from pathlib import Path

# IMPORTANT: absolute imports
from model.architecture.model import GPTModel
from pretraining.dataset import TextDataset


def main():

    # === Paths ===
    corpus_path = os.path.join(PROJECT_ROOT, "data/raw/tokenizer_corpus.txt")
    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer/tokenizer_8k.json")
    config_path = os.path.join(PROJECT_ROOT, "model/config/gpt_small_config.json")

    print("Corpus path:", corpus_path)
    print("Tokenizer path:", tokenizer_path)
    print("Config path:", config_path)

    # === Load model ===
    model = GPTModel.from_config_file(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on device:", device)
    model.to(device)

    # === Dataset ===
    dataset = TextDataset(
        path=corpus_path,
        tokenizer_path=tokenizer_path,
        block_size=128
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # === Optimizer ===
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # === Training ===
    model.train()

    max_steps = 500  # small test run

    for step, (x, y) in enumerate(loader):
        if step > max_steps:
            break

        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    # === Save checkpoint ===
    ckpt_dir = os.path.join(PROJECT_ROOT, "pretraining/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, "gpt-small-ckpt.pt")
    torch.save(model.state_dict(), ckpt_path)

    print(f"Training complete. Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
