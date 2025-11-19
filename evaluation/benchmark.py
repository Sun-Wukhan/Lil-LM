import os
import sys
import json
import math
from datetime import datetime

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Path Fix (allows running script from anywhere) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.architecture.model import GPTModel
from tokenizers import Tokenizer


# ==========================================================
#                UTILITY FUNCTIONS
# ==========================================================

def load_model_and_tokenizer(
        config_path="model/config/gpt_small_config.json",
        checkpoint_path="pretraining/checkpoints/gpt-small-ckpt.pt",
        tokenizer_path="tokenizer/tokenizer_8k.json"
    ):
    """Load model + tokenizer for evaluation."""
    model = GPTModel.from_config_file(config_path)

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    tokenizer = Tokenizer.from_file(tokenizer_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, device


def compute_perplexity(model, device, tokenizer, text):
    """Compute perplexity = exp(loss) over a given text sample."""
    encoded = tokenizer.encode(text).ids
    x = torch.tensor(encoded[:-1]).long().unsqueeze(0).to(device)
    y = torch.tensor(encoded[1:]).long().unsqueeze(0).to(device)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    ppl = torch.exp(loss)

    return float(loss.item()), float(ppl.item())


# ---------------- Benchmark Micro-Tests --------------------

def bracket_test(model, tokenizer, device):
    """Test how well the model completes brackets."""
    prompt = "("
    encoded = tokenizer.encode(prompt).ids
    x = torch.tensor([encoded], dtype=torch.long).to(device)

    logits = model(x)
    pred = logits[0, -1].argmax().item()
    out = tokenizer.decode([pred])

    return 1 if out == ")" else 0


def repeat_test(model, tokenizer, device):
    """Model sees a token twice—should predict third repetition."""
    prompt = "hello hello "
    enc = tokenizer.encode(prompt).ids
    x = torch.tensor([enc], dtype=torch.long).to(device)

    logits = model(x)
    pred = logits[0, -1].argmax().item()
    out = tokenizer.decode([pred]).strip()

    return 1 if out == "hello" else 0


def alphabet_test(model, tokenizer, device):
    """A → B, or predict next letter."""
    prompt = "a b c "
    enc = tokenizer.encode(prompt).ids
    x = torch.tensor([enc], dtype=torch.long).to(device)

    logits = model(x)
    pred = logits[0, -1].argmax().item()
    out = tokenizer.decode([pred]).strip()

    return 1 if out == "d" else 0


def sentence_completion_test(model, tokenizer, device):
    """Very simple next-word guess."""
    prompt = "Once upon a"
    enc = tokenizer.encode(prompt).ids
    x = torch.tensor([enc], dtype=torch.long).to(device)

    logits = model(x)
    pred = logits[0, -1].argmax().item()
    out = tokenizer.decode([pred]).lower()

    # Acceptable continuations
    good = ["time", "a"]

    return 1 if any(g in out for g in good) else 0


# ==========================================================
#                GENERATION SAMPLING
# ==========================================================

def generate_sample(model, tokenizer, device):
    """Generate a small sample for qualitative evaluation."""
    prompt = "In the city of Toronto,"
    enc = tokenizer.encode(prompt).ids
    x = torch.tensor([enc], dtype=torch.long).to(device)

    for _ in range(40):
        logits = model(x)
        next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
        x = torch.cat([x, next_token], dim=1)

    txt = tokenizer.decode(x[0].tolist())
    return txt


# ==========================================================
#                MAIN BENCHMARK FUNCTION
# ==========================================================

def benchmark():
    model, tokenizer, device = load_model_and_tokenizer()

    # --- Loss & Perplexity ---
    sample_text = "Hello world. This is a test sentence for evaluating perplexity."
    loss, ppl = compute_perplexity(model, device, tokenizer, sample_text)

    # --- Micro-tests ---
    results = {
        "loss": loss,
        "perplexity": ppl,
        "bracket_test": bracket_test(model, tokenizer, device),
        "repeat_test": repeat_test(model, tokenizer, device),
        "alphabet_test": alphabet_test(model, tokenizer, device),
        "sentence_test": sentence_completion_test(model, tokenizer, device),
        "sample_generation": generate_sample(model, tokenizer, device)
    }

    # --- Visualisation ---
    plot_path = "evaluation/plots/benchmark_v0.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    labels = ["Brackets", "Repeat", "Alphabet", "Sentence"]
    scores = [
        results["bracket_test"],
        results["repeat_test"],
        results["alphabet_test"],
        results["sentence_test"]
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, scores, color="skyblue")
    plt.ylim(0, 1)
    plt.title("MiniGPT v0 Benchmark Scores")
    plt.ylabel("Correct (1) / Incorrect (0)")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nSaved benchmark plot → {plot_path}")

    # --- Save JSON results ---
    json_dir = "evaluation/results"
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, "baseline_v0.json")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved JSON results → {json_path}")

    return results


if __name__ == "__main__":
    out = benchmark()
    print("\n=== Benchmark Results ===")
    for k, v in out.items():
        print(f"{k}: {v}")
