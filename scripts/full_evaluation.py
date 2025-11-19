"""
Run complete evaluation pipeline after training.
This is your one-stop script to test everything!
"""

import os
import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


def run_script(script_path, description):
    """Run a script and report results."""
    print("\n" + "="*70)
    print(f"🔧 {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️  {description} completed with warnings")
            return False
    
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def main():
    """Run full evaluation pipeline."""
    
    print("="*70)
    print("FULL EVALUATION PIPELINE")
    print("="*70)
    print("\nThis will run all evaluation scripts in sequence.")
    print("Estimated time: 5-15 minutes depending on your hardware.\n")
    
    input("Press Enter to start evaluation...")
    
    results = {}
    
    # 1. Quick generation test
    results['quick_gen'] = run_script(
        "scripts/quick_gen_test.py",
        "Quick Generation Test"
    )
    
    # 2. Tokenizer quality check
    results['tokenizer'] = run_script(
        "scripts/test_tokenizer_quality.py",
        "Tokenizer Quality Check"
    )
    
    # 3. Code evaluation suite
    results['code_tests'] = run_script(
        "evaluation/code_tests.py",
        "Code Evaluation Suite"
    )
    
    # 4. Pytest code generation
    results['pytest'] = run_script(
        "evaluation/pytest_code_generation.py",
        "Pytest Code Generation Tests"
    )
    
    # 5. Checkpoint comparison
    results['checkpoints'] = run_script(
        "evaluation/evaluate_checkpoints.py",
        "Checkpoint Evaluation & Comparison"
    )
    
    # 6. Training summary
    results['training_summary'] = run_script(
        "scripts/monitor_training.py --summary",
        "Training Summary"
    )
    
    # Final report
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nCompleted: {passed}/{total} tasks")
    print("\nTask Results:")
    for task, success in results.items():
        status = "✅" if success else "⚠️"
        print(f"  {status} {task}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    print("""
📊 Review Results:
  - evaluation/results/code_eval.json
  - evaluation/results/checkpoint_comparison.json
  - logs/training_log.json

🧪 Test Generation:
  python pretraining/generate.py

📈 If Results Are Good (perplexity < 100):
  - Scale up to gpt_small_config.json
  - Train longer (40K steps)
  - Add more diverse code data

📉 If Results Need Improvement:
  - Check IMPROVEMENTS.md for optimization tips
  - Try adjusting learning rate
  - Increase training duration
  - Use pure code corpus

💡 For Learning & Career:
  - Read through IMPROVEMENTS.md to understand what was fixed
  - Experiment with different hyperparameters
  - Try training on your own code datasets
  - Explore SFT and RL training next
    """)
    
    print("="*70)
    print("Evaluation pipeline complete! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()

