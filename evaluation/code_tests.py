"""
Code-specific evaluation tests for the mini LLM.
Tests syntax understanding, code completion, and generation capabilities.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from model.architecture.model import GPTModel
from tokenizers import Tokenizer


def load_model_and_tokenizer(
        config_path="model/config/gpt_tiny_config.json",
        checkpoint_path=None,  # Auto-find if None
        tokenizer_path="tokenizer/tokenizer_8k.json"
    ):
    """Load model and tokenizer."""
    config_path = os.path.join(PROJECT_ROOT, config_path)
    tokenizer_path = os.path.join(PROJECT_ROOT, tokenizer_path)
    
    # Auto-find latest checkpoint if not specified
    if checkpoint_path is None:
        from pathlib import Path
        ckpt_dir = Path(PROJECT_ROOT) / "pretraining" / "checkpoints"
        
        # Try to find latest best_step checkpoint
        best_ckpts = sorted(ckpt_dir.glob("best_step_*.pt"), 
                           key=lambda x: int(x.stem.split('_')[-1]))
        
        if best_ckpts:
            checkpoint_path = str(best_ckpts[-1])
            print(f"Auto-detected latest checkpoint: {best_ckpts[-1].name}")
        else:
            # Fallback to gpt-small-ckpt.pt
            checkpoint_path = os.path.join(PROJECT_ROOT, "pretraining/checkpoints/gpt-small-ckpt.pt")
    else:
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
    
    print(f"Loading model from: {config_path}")
    model = GPTModel.from_config_file(config_path)
    
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        # Handle both direct state dict and checkpoint dict
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
            print(f"Loaded checkpoint: {checkpoint_path} (step {state.get('step', '?')})")
        else:
            model.load_state_dict(state)
            print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}, using random weights")
    
    model.eval()
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model, tokenizer, device


# ============================================================
#                    SYNTAX TESTS
# ============================================================

def test_bracket_completion(model, tokenizer, device) -> Dict:
    """Test if model can complete bracket pairs."""
    tests = [
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
        ("def func(", ")"),
        ("my_list[", "]"),
    ]
    
    results = []
    for prompt, expected in tests:
        encoded = tokenizer.encode(prompt).ids
        x = torch.tensor([encoded], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(x)
            pred_id = logits[0, -1].argmax().item()
            pred_token = tokenizer.decode([pred_id])
        
        # Check if predicted token contains expected bracket
        passed = expected in pred_token
        results.append({
            'prompt': prompt,
            'expected': expected,
            'predicted': pred_token,
            'passed': passed
        })
    
    score = sum(r['passed'] for r in results) / len(results)
    return {'score': score, 'details': results}


def test_keyword_completion(model, tokenizer, device) -> Dict:
    """Test Python keyword prediction."""
    tests = [
        ("def ", ["func", "main", "test", "get", "set", "is"]),  # function name patterns
        ("import ", ["torch", "os", "sys", "numpy", "math"]),  # common imports
        ("class ", ["M", "G", "T", "C", "A"]),  # class names (often capitalized)
        ("if ", ["x", "i", "not", "True", "False"]),  # condition starters
        ("for ", ["i", "x", "item", "key", "value"]),  # loop variables
    ]
    
    results = []
    for prompt, acceptable in tests:
        encoded = tokenizer.encode(prompt).ids
        x = torch.tensor([encoded], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(x)
            pred_id = logits[0, -1].argmax().item()
            pred_token = tokenizer.decode([pred_id]).strip()
        
        # Check if prediction starts with any acceptable pattern
        passed = any(pred_token.startswith(acc) or acc in pred_token for acc in acceptable)
        results.append({
            'prompt': prompt,
            'acceptable': acceptable,
            'predicted': pred_token,
            'passed': passed
        })
    
    score = sum(r['passed'] for r in results) / len(results)
    return {'score': score, 'details': results}


def test_indentation_awareness(model, tokenizer, device) -> Dict:
    """Test if model understands indentation context."""
    tests = [
        ("def func():\n    ", "return"),  # Should predict code after indent
        ("if True:\n    ", "pass"),  # Should predict statement
        ("for i in range(10):\n    ", "print"),  # Should predict statement
    ]
    
    results = []
    for prompt, expected_pattern in tests:
        encoded = tokenizer.encode(prompt).ids
        x = torch.tensor([encoded], dtype=torch.long).to(device)
        
        with torch.no_grad():
            logits = model(x)
            pred_id = logits[0, -1].argmax().item()
            pred_token = tokenizer.decode([pred_id]).strip()
        
        # Just check if it's not dedenting (shouldn't predict closing bracket, etc.)
        passed = pred_token not in ['}', ')', ']'] and len(pred_token) > 0
        results.append({
            'prompt': prompt.replace('\n', '\\n'),
            'predicted': pred_token,
            'passed': passed
        })
    
    score = sum(r['passed'] for r in results) / len(results)
    return {'score': score, 'details': results}


# ============================================================
#                    CODE COMPLETION TESTS
# ============================================================

def test_simple_function_completion(model, tokenizer, device, max_tokens=20) -> Dict:
    """Test if model can complete simple function patterns."""
    tests = [
        {
            'prompt': 'def add(a, b):\n    return ',
            'keywords': ['a', '+', 'b'],  # Should predict something like "a + b"
        },
        {
            'prompt': 'def factorial(n):\n    if n == 0:\n        return ',
            'keywords': ['1', '0'],  # Base case should return 1 or 0
        },
        {
            'prompt': 'def is_even(x):\n    return x % ',
            'keywords': ['2', '==', '0'],  # Should involve modulo 2
        },
    ]
    
    results = []
    for test in tests:
        prompt = test['prompt']
        encoded = tokenizer.encode(prompt).ids
        x = torch.tensor([encoded], dtype=torch.long).to(device)
        
        # Generate continuation
        generated_ids = encoded.copy()
        for _ in range(max_tokens):
            x_input = torch.tensor([generated_ids], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(x_input)
                next_id = logits[0, -1].argmax().item()
            
            generated_ids.append(next_id)
            
            # Stop at newline
            token = tokenizer.decode([next_id])
            if '\n' in token:
                break
        
        completion = tokenizer.decode(generated_ids)
        
        # Check if any keywords appear in completion
        passed = any(kw in completion for kw in test['keywords'])
        results.append({
            'prompt': prompt.replace('\n', '\\n'),
            'completion': completion[len(prompt):].replace('\n', '\\n'),
            'keywords': test['keywords'],
            'passed': passed
        })
    
    score = sum(r['passed'] for r in results) / len(results)
    return {'score': score, 'details': results}


def test_docstring_completion(model, tokenizer, device, max_tokens=30) -> Dict:
    """Test if model can generate reasonable docstrings."""
    tests = [
        'def quick_sort(collection: list) -> list:\n    """',
        'def add(a, b):\n    """',
        'class GPTModel:\n    """',
    ]
    
    results = []
    for prompt in tests:
        encoded = tokenizer.encode(prompt).ids
        generated_ids = encoded.copy()
        
        for _ in range(max_tokens):
            x = torch.tensor([generated_ids], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(x)
                next_id = logits[0, -1].argmax().item()
            
            generated_ids.append(next_id)
            
            # Stop at closing docstring
            completion = tokenizer.decode(generated_ids)
            if '"""' in completion[len(prompt):]:
                break
        
        completion = tokenizer.decode(generated_ids)
        docstring = completion[len(prompt):]
        
        # Basic quality checks
        has_text = len(docstring.strip()) > 5
        no_gibberish = not any(c * 5 in docstring for c in 'abcdefghijklmnopqrstuvwxyz')
        
        passed = has_text and no_gibberish
        results.append({
            'prompt': prompt.replace('\n', '\\n'),
            'docstring': docstring.replace('\n', '\\n')[:100],
            'passed': passed
        })
    
    score = sum(r['passed'] for r in results) / len(results)
    return {'score': score, 'details': results}


# ============================================================
#                    PERPLEXITY TESTS
# ============================================================

def test_code_perplexity(model, tokenizer, device) -> Dict:
    """Measure perplexity on code samples."""
    code_samples = [
        "def add(a, b):\n    return a + b",
        "for i in range(10):\n    print(i)",
        "import torch\nfrom torch import nn",
        "class Model(nn.Module):\n    def __init__(self):\n        super().__init__()",
    ]
    
    perplexities = []
    for sample in code_samples:
        encoded = tokenizer.encode(sample).ids
        if len(encoded) < 2:
            continue
        
        x = torch.tensor(encoded[:-1]).long().unsqueeze(0).to(device)
        y = torch.tensor(encoded[1:]).long().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            ppl = torch.exp(loss).item()
        
        perplexities.append({
            'sample': sample.replace('\n', '\\n')[:50],
            'perplexity': ppl
        })
    
    avg_ppl = sum(p['perplexity'] for p in perplexities) / len(perplexities)
    
    return {
        'average_perplexity': avg_ppl,
        'samples': perplexities,
        'score': 1.0 if avg_ppl < 50 else (0.5 if avg_ppl < 100 else 0.0)
    }


# ============================================================
#                    MAIN BENCHMARK
# ============================================================

def run_code_benchmark(config_path=None, checkpoint_path=None):
    """Run full code evaluation benchmark."""
    
    print("="*70)
    print("CODE EVALUATION BENCHMARK")
    print("="*70)
    
    # Load model
    kwargs = {}
    if config_path:
        kwargs['config_path'] = config_path
    if checkpoint_path:
        kwargs['checkpoint_path'] = checkpoint_path
    
    model, tokenizer, device = load_model_and_tokenizer(**kwargs)
    
    # Run all tests
    print("\n📋 Running tests...\n")
    
    results = {}
    
    print("1️⃣  Bracket Completion Test")
    results['bracket_completion'] = test_bracket_completion(model, tokenizer, device)
    print(f"   Score: {results['bracket_completion']['score']:.2%}\n")
    
    print("2️⃣  Keyword Completion Test")
    results['keyword_completion'] = test_keyword_completion(model, tokenizer, device)
    print(f"   Score: {results['keyword_completion']['score']:.2%}\n")
    
    print("3️⃣  Indentation Awareness Test")
    results['indentation_awareness'] = test_indentation_awareness(model, tokenizer, device)
    print(f"   Score: {results['indentation_awareness']['score']:.2%}\n")
    
    print("4️⃣  Function Completion Test")
    results['function_completion'] = test_simple_function_completion(model, tokenizer, device)
    print(f"   Score: {results['function_completion']['score']:.2%}\n")
    
    print("5️⃣  Docstring Completion Test")
    results['docstring_completion'] = test_docstring_completion(model, tokenizer, device)
    print(f"   Score: {results['docstring_completion']['score']:.2%}\n")
    
    print("6️⃣  Code Perplexity Test")
    results['code_perplexity'] = test_code_perplexity(model, tokenizer, device)
    print(f"   Average Perplexity: {results['code_perplexity']['average_perplexity']:.2f}\n")
    
    # Calculate overall score
    syntax_score = (
        results['bracket_completion']['score'] +
        results['keyword_completion']['score'] +
        results['indentation_awareness']['score']
    ) / 3
    
    generation_score = (
        results['function_completion']['score'] +
        results['docstring_completion']['score']
    ) / 2
    
    perplexity_score = results['code_perplexity']['score']
    
    overall_score = (syntax_score + generation_score + perplexity_score) / 3
    
    results['summary'] = {
        'syntax_score': syntax_score,
        'generation_score': generation_score,
        'perplexity_score': perplexity_score,
        'overall_score': overall_score
    }
    
    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Syntax Understanding:  {syntax_score:.2%}")
    print(f"Code Generation:       {generation_score:.2%}")
    print(f"Perplexity Score:      {perplexity_score:.2%}")
    print(f"Overall Score:         {overall_score:.2%}")
    
    return results


def save_results(results, output_path="evaluation/results/code_eval.json"):
    """Save evaluation results to JSON."""
    output_path = os.path.join(PROJECT_ROOT, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run code evaluation tests")
    parser.add_argument("--config", type=str, help="Path to model config")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="evaluation/results/code_eval.json",
                        help="Output path for results")
    
    args = parser.parse_args()
    
    results = run_code_benchmark(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    save_results(results, args.output)

