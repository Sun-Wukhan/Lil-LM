"""
Pytest-based tests for code generation capability.
Tests if the model can generate code that actually works.
"""

import os
import sys
from pathlib import Path
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from model.architecture.model import GPTModel
from tokenizers import Tokenizer


def load_model():
    """Load the trained model."""
    config_path = os.path.join(PROJECT_ROOT, "model/config/gpt_tiny_config.json")
    checkpoint_path = os.path.join(PROJECT_ROOT, "pretraining/checkpoints/gpt-small-ckpt.pt")
    tokenizer_path = os.path.join(PROJECT_ROOT, "tokenizer/tokenizer_8k.json")
    
    model = GPTModel.from_config_file(config_path)
    
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
    
    model.eval()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return model, tokenizer, device


def generate_code(model, tokenizer, device, prompt, max_tokens=50):
    """Generate code continuation."""
    encoded = tokenizer.encode(prompt).ids
    generated_ids = encoded.copy()
    
    for _ in range(max_tokens):
        x = torch.tensor([generated_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(x)
            next_id = logits[0, -1].argmax().item()
        
        generated_ids.append(next_id)
        
        # Stop at certain patterns
        completion = tokenizer.decode(generated_ids)
        if len(completion) > len(prompt) + 5:
            # Stop at double newline or certain keywords
            recent = completion[len(prompt):]
            if '\n\n' in recent or 'def ' in recent[5:] or 'class ' in recent[5:]:
                break
    
    return tokenizer.decode(generated_ids)


# ============================================================
#                    SIMPLE FUNCTION TESTS
# ============================================================

def test_add_function():
    """Test if model can generate a working add function."""
    model, tokenizer, device = load_model()
    
    prompt = "def add(a, b):\n    return "
    generated = generate_code(model, tokenizer, device, prompt, max_tokens=10)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    
    # Extract the return statement
    try:
        # Try to execute the generated function
        full_code = generated.split('\n\n')[0]  # Take first complete block
        
        # Create a local namespace
        namespace = {}
        exec(full_code, namespace)
        
        # Test the function
        if 'add' in namespace:
            result = namespace['add'](2, 3)
            assert result == 5, f"Expected 5, got {result}"
            print(f"✓ Function works! add(2, 3) = {result}")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return False


def test_max_function():
    """Test if model can generate a working max function."""
    model, tokenizer, device = load_model()
    
    prompt = "def maximum(a, b):\n    if a > b:\n        return "
    generated = generate_code(model, tokenizer, device, prompt, max_tokens=20)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    
    # Just check if it generates 'a' as the return value
    completion = generated[len(prompt):].strip()
    return completion.startswith('a')


def test_is_even():
    """Test if model can generate an is_even function."""
    model, tokenizer, device = load_model()
    
    prompt = "def is_even(n):\n    return n % 2 == "
    generated = generate_code(model, tokenizer, device, prompt, max_tokens=5)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    
    # Check if it completes with 0
    completion = generated[len(prompt):].strip()
    return completion.startswith('0')


# ============================================================
#                    LIST OPERATION TESTS
# ============================================================

def test_list_sum():
    """Test if model can generate sum logic."""
    model, tokenizer, device = load_model()
    
    prompt = "def sum_list(items):\n    total = 0\n    for item in items:\n        total += "
    generated = generate_code(model, tokenizer, device, prompt, max_tokens=10)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    
    completion = generated[len(prompt):].strip()
    return 'item' in completion


def test_list_reverse():
    """Test if model understands list reversal."""
    model, tokenizer, device = load_model()
    
    prompt = "def reverse_list(lst):\n    return lst["
    generated = generate_code(model, tokenizer, device, prompt, max_tokens=10)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    
    completion = generated[len(prompt):].strip()
    # Check for slicing pattern
    return '::-1' in completion or '-1]' in completion


# ============================================================
#                    STRING OPERATION TESTS
# ============================================================

def test_string_upper():
    """Test if model knows string methods."""
    model, tokenizer, device = load_model()
    
    prompt = "def make_uppercase(text):\n    return text."
    generated = generate_code(model, tokenizer, device, prompt, max_tokens=10)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
    
    completion = generated[len(prompt):].strip()
    return 'upper' in completion.lower()


# ============================================================
#                    MAIN TEST RUNNER
# ============================================================

def run_all_tests():
    """Run all code generation tests."""
    
    print("="*70)
    print("PYTEST-BASED CODE GENERATION TESTS")
    print("="*70)
    
    tests = [
        ("Add Function", test_add_function),
        ("Max Function", test_max_function),
        ("Is Even Function", test_is_even),
        ("List Sum", test_list_sum),
        ("List Reverse", test_list_reverse),
        ("String Upper", test_string_upper),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print(f"{'='*70}")
        
        try:
            passed = test_func()
            results.append({
                'name': name,
                'passed': passed,
                'error': None
            })
            print(f"Result: {'✓ PASS' if passed else '✗ FAIL'}")
        except Exception as e:
            results.append({
                'name': name,
                'passed': False,
                'error': str(e)
            })
            print(f"Result: ✗ ERROR - {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total} ({passed/total:.1%})")
    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r['passed'] else "✗"
        print(f"  {status} {r['name']}")
    
    return results


if __name__ == "__main__":
    run_all_tests()

