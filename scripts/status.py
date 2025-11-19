#!/usr/bin/env python3
"""
Quick status checker for your mini LLM project.
Run this anytime to see what's happening!
"""

import os
import sys
import subprocess
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_training_running():
    """Check if training is currently running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "train_improved.py"],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except:
        return False


def get_training_progress():
    """Get current training progress from log."""
    log_path = PROJECT_ROOT / "logs" / "training_log.json"
    
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
        
        if not logs:
            return None
        
        latest = logs[-1]
        
        # Get training entries
        train_logs = [l for l in logs if 'loss' in l]
        
        if train_logs:
            first_loss = train_logs[0].get('loss', 0)
            current_loss = train_logs[-1].get('loss', 0)
            improvement = first_loss - current_loss
        else:
            improvement = 0
        
        return {
            'step': latest.get('step', 0),
            'loss': latest.get('loss'),
            'perplexity': latest.get('perplexity'),
            'val_loss': latest.get('val_loss'),
            'val_perplexity': latest.get('val_perplexity'),
            'improvement': improvement
        }
    except:
        return None


def check_checkpoints():
    """Check available checkpoints."""
    ckpt_dir = PROJECT_ROOT / "pretraining" / "checkpoints"
    
    if not ckpt_dir.exists():
        return []
    
    checkpoints = []
    for ckpt in ckpt_dir.glob("*.pt"):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(ckpt.stat().st_mtime)
        
        checkpoints.append({
            'name': ckpt.name,
            'size_mb': size_mb,
            'modified': mod_time
        })
    
    return sorted(checkpoints, key=lambda x: x['modified'], reverse=True)


def check_corpus():
    """Check corpus files."""
    mixed = PROJECT_ROOT / "data" / "processed" / "mixed_corpus.txt"
    code = PROJECT_ROOT / "data" / "processed" / "code_corpus.txt"
    
    info = {}
    
    if mixed.exists():
        with open(mixed, 'r') as f:
            content = f.read()
            info['mixed'] = {
                'lines': len(content.split('\n')),
                'words': len(content.split()),
                'chars': len(content)
            }
    
    if code.exists():
        with open(code, 'r') as f:
            content = f.read()
            info['code'] = {
                'lines': len(content.split('\n')),
                'words': len(content.split()),
                'chars': len(content)
            }
    
    return info


def main():
    """Display complete project status."""
    
    print("="*70)
    print("MINI LLM PROJECT STATUS")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Training status
    print("📊 TRAINING STATUS")
    print("-"*70)
    
    is_running = check_training_running()
    
    if is_running:
        print("✅ Training is RUNNING")
        
        progress = get_training_progress()
        if progress:
            print(f"\nProgress:")
            print(f"  Step: {progress['step']:,}/20,000 ({progress['step']/20000*100:.1f}%)")
            
            if progress['loss']:
                print(f"  Loss: {progress['loss']:.4f}")
            if progress['perplexity']:
                print(f"  Perplexity: {progress['perplexity']:.2f}")
            if progress['improvement']:
                print(f"  Improvement: {progress['improvement']:.4f}")
            
            if progress['val_loss']:
                print(f"\nValidation:")
                print(f"  Val Loss: {progress['val_loss']:.4f}")
                print(f"  Val Perplexity: {progress['val_perplexity']:.2f}")
            
            # Estimate time remaining
            if progress['step'] > 0:
                # Very rough estimate
                steps_remaining = 20000 - progress['step']
                hours_per_10k = 2  # rough estimate
                hours_remaining = (steps_remaining / 10000) * hours_per_10k
                print(f"\nEstimated time remaining: ~{hours_remaining:.1f} hours")
        else:
            print("  (Initializing - no logs yet)")
    else:
        print("⏸️  Training is NOT running")
        
        progress = get_training_progress()
        if progress and progress['step'] > 0:
            print(f"\nLast checkpoint:")
            print(f"  Step: {progress['step']:,}")
            if progress['loss']:
                print(f"  Loss: {progress['loss']:.4f}")
            if progress['perplexity']:
                print(f"  Perplexity: {progress['perplexity']:.2f}")
            
            if progress['step'] >= 20000:
                print("\n✅ Training appears COMPLETE!")
                print("   Run: python scripts/full_evaluation.py")
            else:
                print(f"\n⚠️  Training stopped at {progress['step']:,}/20,000 steps")
                print("   Resume: python pretraining/train_improved.py")
        else:
            print("  No training logs found")
            print("  Start training: python pretraining/train_improved.py")
    
    # Checkpoints
    print("\n📦 CHECKPOINTS")
    print("-"*70)
    
    checkpoints = check_checkpoints()
    
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s):\n")
        for ckpt in checkpoints[:5]:  # Show latest 5
            print(f"  • {ckpt['name']:<40} {ckpt['size_mb']:>6.1f} MB  {ckpt['modified'].strftime('%H:%M:%S')}")
        
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints)-5} more")
    else:
        print("No checkpoints found yet")
    
    # Corpus
    print("\n📚 TRAINING DATA")
    print("-"*70)
    
    corpus_info = check_corpus()
    
    if corpus_info:
        if 'mixed' in corpus_info:
            info = corpus_info['mixed']
            print(f"Mixed Corpus:")
            print(f"  Lines: {info['lines']:,}")
            print(f"  Words: {info['words']:,}")
            print(f"  Chars: {info['chars']:,}")
        
        if 'code' in corpus_info:
            info = corpus_info['code']
            print(f"\nCode Corpus:")
            print(f"  Lines: {info['lines']:,}")
            print(f"  Words: {info['words']:,}")
            print(f"  Chars: {info['chars']:,}")
    else:
        print("⚠️  Corpus files not found")
        print("   Run: python scripts/create_mixed_corpus.py")
    
    # Tokenizer
    print("\n🔤 TOKENIZER")
    print("-"*70)
    
    tokenizer_path = PROJECT_ROOT / "tokenizer" / "tokenizer_8k.json"
    
    if tokenizer_path.exists():
        size_mb = tokenizer_path.stat().st_size / (1024 * 1024)
        print(f"✅ Fixed tokenizer: tokenizer_8k.json ({size_mb:.1f} MB)")
        print("   Test: python scripts/test_tokenizer_quality.py")
    else:
        print("⚠️  Tokenizer not found")
    
    # Quick actions
    print("\n⚡ QUICK ACTIONS")
    print("-"*70)
    
    if is_running:
        print("  python scripts/monitor_training.py     # Watch training live")
        print("  python scripts/monitor_training.py --summary  # Show summary")
    elif progress and progress['step'] >= 20000:
        print("  python scripts/full_evaluation.py      # Run all evaluations")
        print("  python scripts/quick_gen_test.py       # Test generation")
        print("  python evaluation/code_tests.py        # Code evaluation")
    else:
        print("  python pretraining/train_improved.py   # Start/resume training")
        print("  python scripts/monitor_training.py     # Monitor training")
    
    print("\n💡 DOCUMENTATION")
    print("-"*70)
    print("  IMPLEMENTATION_COMPLETE.md   # Overview of everything done")
    print("  IMPROVEMENTS.md              # Technical details")
    print("  USAGE_GUIDE.md               # How to use everything")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

