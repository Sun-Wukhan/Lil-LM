"""
Improved training script with:
- Learning rate scheduler (cosine decay)
- Gradient clipping
- Validation tracking
- Regular checkpointing
- Better logging and monitoring
"""

import os
import sys
import json
import math
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm

from model.architecture.model import GPTModel
from pretraining.dataset import TextDataset


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Creates a learning rate scheduler with warmup and cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of steps for linear warmup
        num_training_steps: Total training steps
        min_lr_ratio: Minimum learning rate as ratio of initial lr
    """
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, dataloader, device, max_batches=50):
    """
    Evaluate model on validation set.
    
    Args:
        model: GPT model
        dataloader: Validation dataloader
        device: torch device
        max_batches: Maximum batches to evaluate (for speed)
    
    Returns:
        avg_loss: Average validation loss
        perplexity: Validation perplexity
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            if num_batches >= max_batches:
                break
            
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(min(avg_loss, 20))  # Cap at 20 to prevent overflow
    
    model.train()
    return avg_loss, perplexity


def save_checkpoint(model, optimizer, scheduler, step, loss, checkpoint_dir, name="checkpoint"):
    """Save model checkpoint with metadata."""
    checkpoint_path = checkpoint_dir / f"{name}_step_{step}.pt"
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, checkpoint_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def main():
    print("="*70)
    print("IMPROVED TRAINING SCRIPT")
    print("="*70)
    
    # === Configuration ===
    config = {
        'corpus_file': 'data/processed/mixed_corpus.txt',
        'tokenizer_path': 'tokenizer/tokenizer_8k.json',
        'config_path': 'model/config/gpt_tiny_config.json',  # Using tiny for faster training
        'checkpoint_dir': 'pretraining/checkpoints',
        'log_dir': 'logs',
        
        # Training hyperparameters
        'max_steps': 20000,
        'batch_size': 8,  # Reduced for MacBook
        'block_size': 128,
        'learning_rate': 1e-4,  # More conservative than 3e-4
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'grad_clip_norm': 1.0,
        
        # Logging and checkpointing
        'log_interval': 20,
        'eval_interval': 500,
        'checkpoint_interval': 2000,
        'val_batch_size': 8,
        'val_split': 0.05,  # 5% for validation
    }
    
    print("\n📋 Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # === Setup directories ===
    checkpoint_dir = Path(PROJECT_ROOT) / config['checkpoint_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(PROJECT_ROOT) / config['log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # === Device ===
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Training on: {device}")
    
    # === Load Model ===
    print("\n📦 Loading model...")
    config_path = os.path.join(PROJECT_ROOT, config['config_path'])
    model = GPTModel.from_config_file(config_path)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # === Dataset ===
    print("\n📚 Loading dataset...")
    corpus_path = os.path.join(PROJECT_ROOT, config['corpus_file'])
    full_dataset = TextDataset(
        corpus_file=corpus_path,
        tokenizer_path=os.path.join(PROJECT_ROOT, config['tokenizer_path']),
        block_size=config['block_size']
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # MacBook optimization
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # === Optimizer and Scheduler ===
    print("\n⚙️  Setting up optimizer and scheduler...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)  # Better for transformers
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['max_steps'],
        min_lr_ratio=0.1
    )
    
    # === Training Loop ===
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    model.train()
    global_step = 0
    running_loss = 0
    best_val_loss = float('inf')
    
    # Training log
    training_log = []
    
    epoch = 0
    while global_step < config['max_steps']:
        epoch += 1
        print(f"\n📖 Epoch {epoch}")
        
        pbar = tqdm(train_loader, desc=f"Training")
        for batch_idx, (x, y) in enumerate(pbar):
            if global_step >= config['max_steps']:
                break
            
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['grad_clip_norm']
            )
            
            optimizer.step()
            scheduler.step()
            
            # Track loss
            running_loss += loss.item()
            global_step += 1
            
            # Logging
            if global_step % config['log_interval'] == 0:
                avg_loss = running_loss / config['log_interval']
                current_lr = scheduler.get_last_lr()[0]
                ppl = math.exp(min(avg_loss, 20))
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{ppl:.2f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                training_log.append({
                    'step': global_step,
                    'loss': avg_loss,
                    'perplexity': ppl,
                    'lr': current_lr
                })
                
                running_loss = 0
            
            # Evaluation
            if global_step % config['eval_interval'] == 0:
                print(f"\n📊 Evaluation at step {global_step}...")
                val_loss, val_ppl = evaluate(model, val_loader, device)
                print(f"   Validation loss: {val_loss:.4f}")
                print(f"   Validation perplexity: {val_ppl:.2f}")
                
                training_log.append({
                    'step': global_step,
                    'val_loss': val_loss,
                    'val_perplexity': val_ppl
                })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, scheduler,
                        global_step, val_loss,
                        checkpoint_dir, name="best"
                    )
            
            # Regular checkpointing
            if global_step % config['checkpoint_interval'] == 0:
                save_checkpoint(
                    model, optimizer, scheduler,
                    global_step, loss.item(),
                    checkpoint_dir, name="checkpoint"
                )
    
    # === Final Save ===
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    final_path = checkpoint_dir / "gpt-small-ckpt.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n💾 Final model saved: {final_path}")
    
    # Save training log
    log_path = log_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"📊 Training log saved: {log_path}")
    
    print(f"\n✓ Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

