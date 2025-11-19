import os
import sys
import torch
import torch.nn.functional as F

# --- Ensure project root is on path (so imports work no matter where script is run) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---------------------------------------------------------------------------------------

from model.architecture.model import GPTModel
from tokenizers import Tokenizer


def load_model_and_tokenizer(
        config_path="model/config/gpt_small_config.json",
        checkpoint_path="pretraining/checkpoints/gpt-small-ckpt.pt",
        tokenizer_path="tokenizer/tokenizer_8k.json"
    ):
    """
    Loads:
      • model architecture (from config)
      • model weights (from checkpoint)
      • tokenizer (BPE JSON file)
    """
    print("Loading model...")
    model = GPTModel.from_config_file(config_path)

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on:", device)
    model.to(device)

    return model, tokenizer, device


def sample_next_token(logits, temperature=1.0, top_k=50, top_p=1.0):
    """
    Numerically stable sampling.
    """

    # --- Safety: replace NaNs / infinities BEFORE anything else ---
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    # --- Temperature ---
    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / max(temperature, 1e-8)

    # --- Top-K ---
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[..., -1].unsqueeze(-1)
        logits = torch.where(logits < min_val, torch.tensor(-1e10), logits)

    # --- Top-P (Nucleus Sampling) ---
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)

        cumulative = torch.cumsum(probs, dim=-1)
        mask = cumulative > top_p

        sorted_logits[mask] = -1e10

        # Scatter back to original order
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

    # --- Softmax with stabilization ---
    probs = F.softmax(logits, dim=-1)

    # --- Clean illegal values ---
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    # Rare edge case: all zeros → fallback to greedy
    if torch.sum(probs) == 0:
        return torch.argmax(logits, dim=-1)

    # --- Sample ---
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.squeeze(-1)


def generate(
        model,
        tokenizer,
        device,
        prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    ):
    """
    Autoregressive text generation loop.

    Arguments you will often tweak when working on future LLMs:

    prompt:
      The text your model starts with.

    max_new_tokens:
      How many tokens to generate AFTER the prompt.

    temperature:
      Controls randomness.
      0.0 = fully deterministic
      0.5 = conservative, safer
      1.0 = typical GPT randomness
      1.5 = chaotic philosopher on mushrooms

    top_k:
      Limits sampling to the top K tokens (helps with coherence)

    top_p:
      Nucleus sampling — only choose tokens within the top probability mass.

    Together, temperature + top_k + top_p shape your model's "personality."
    """

    # Encode prompt → token IDs
    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):

        # 1. Forward pass through the model
        logits = model(input_ids)

        # 2. Get logits of the final position (last token)
        next_logits = logits[:, -1, :]  # shape: (1, vocab_size)

        # 3. Sample a new token ID
        next_token = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        next_token = next_token.unsqueeze(0)

        # 4. Append token to the running sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode back into text
    output_text = tokenizer.decode(input_ids[0].tolist())
    return output_text


if __name__ == "__main__":
    # --- LOAD EVERYTHING ---
    model, tokenizer, device = load_model_and_tokenizer()

    # --- WRITE YOUR PROMPT HERE ---
    prompt = "Once upon a time in Toronto,"

    print("\n=== Generating ===\n")
    output = generate(
        model,
        tokenizer,
        device,
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        top_p=0.9
    )

    print(output)
