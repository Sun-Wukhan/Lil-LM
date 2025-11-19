"""
Monitor training progress by reading the log file.
"""

import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def monitor_training(log_path="logs/training_log.json", refresh_interval=10):
    """
    Monitor training progress from log file.
    
    Args:
        log_path: Path to training log JSON
        refresh_interval: Seconds between refreshes
    """
    log_path = PROJECT_ROOT / log_path
    
    print("="*70)
    print("TRAINING MONITOR")
    print("="*70)
    print(f"Monitoring: {log_path}")
    print(f"Press Ctrl+C to stop monitoring\n")
    
    last_size = 0
    
    try:
        while True:
            if log_path.exists():
                # Check if file has been updated
                current_size = log_path.stat().st_size
                
                if current_size != last_size:
                    try:
                        with open(log_path, 'r') as f:
                            logs = json.load(f)
                        
                        if logs:
                            latest = logs[-1]
                            
                            print(f"\r[{time.strftime('%H:%M:%S')}] ", end='')
                            
                            if 'step' in latest:
                                print(f"Step {latest['step']:>6} | ", end='')
                            
                            if 'loss' in latest:
                                print(f"Loss: {latest['loss']:.4f} | ", end='')
                            
                            if 'perplexity' in latest:
                                print(f"PPL: {latest['perplexity']:>6.2f} | ", end='')
                            
                            if 'val_loss' in latest:
                                print(f"Val Loss: {latest['val_loss']:.4f} | ", end='')
                            
                            if 'val_perplexity' in latest:
                                print(f"Val PPL: {latest['val_perplexity']:>6.2f} | ", end='')
                            
                            if 'lr' in latest:
                                print(f"LR: {latest['lr']:.2e}", end='')
                            
                            print(flush=True)
                        
                        last_size = current_size
                    
                    except json.JSONDecodeError:
                        # File might be being written
                        pass
                    except Exception as e:
                        print(f"\nError reading log: {e}")
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] Waiting for training to start...", end='', flush=True)
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def show_training_summary():
    """Show summary of training progress."""
    log_path = PROJECT_ROOT / "logs" / "training_log.json"
    
    if not log_path.exists():
        print("No training log found yet.")
        return
    
    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    if not logs:
        print("Training log is empty.")
        return
    
    print("="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    # Filter train vs val entries
    train_logs = [l for l in logs if 'loss' in l]
    val_logs = [l for l in logs if 'val_loss' in l]
    
    if train_logs:
        first_train = train_logs[0]
        latest_train = train_logs[-1]
        
        print(f"\nTraining Progress:")
        print(f"  Steps completed: {latest_train.get('step', 'N/A')}")
        print(f"  Initial loss: {first_train.get('loss', 'N/A'):.4f}")
        print(f"  Current loss: {latest_train.get('loss', 'N/A'):.4f}")
        print(f"  Initial perplexity: {first_train.get('perplexity', 'N/A'):.2f}")
        print(f"  Current perplexity: {latest_train.get('perplexity', 'N/A'):.2f}")
        
        if len(train_logs) > 1:
            loss_improvement = first_train.get('loss', 0) - latest_train.get('loss', 0)
            print(f"  Loss improvement: {loss_improvement:.4f}")
    
    if val_logs:
        print(f"\nValidation Results:")
        latest_val = val_logs[-1]
        print(f"  Latest val loss: {latest_val.get('val_loss', 'N/A'):.4f}")
        print(f"  Latest val perplexity: {latest_val.get('val_perplexity', 'N/A'):.2f}")
        
        if len(val_logs) > 1:
            best_val = min(val_logs, key=lambda x: x.get('val_loss', float('inf')))
            print(f"  Best val loss: {best_val.get('val_loss', 'N/A'):.4f}")
            print(f"  Best val step: {best_val.get('step', 'N/A')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Show summary instead of monitoring")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    if args.summary:
        show_training_summary()
    else:
        monitor_training(refresh_interval=args.interval)

