# Mini LLM - Code Generation Model

A from-scratch GPT-style language model that learns to generate code and text. This project demonstrates the complete pipeline of training a small language model, from data preparation to evaluation.

## 🎯 What This Actually Does

This project trains a **27 million parameter transformer model** (mini GPT) that can:
- Generate coherent text and Python code
- Complete code snippets
- Understand programming patterns
- Learn from a mixture of code and books

**Current Status**: Model can generate proper words and follow basic patterns. Training takes ~4 hours on a MacBook.

## 🚀 Quick Start - Get It Running

### 1. Check What's Happening Right Now

```bash
# See project status (training progress, checkpoints, etc.)
python scripts/status.py
```

### 2. Start Training (if not already running)

```bash
# This trains the model for 20,000 steps (~4 hours on MacBook)
python pretraining/train_improved.py
```

**What happens during training:**
- Loads 4,882 lines of code + books
- Trains a 27M parameter model
- Saves checkpoints every 2,000 steps
- Tracks loss, perplexity, and learning rate
- Validates every 500 steps

### 3. Monitor Training Progress

```bash
# Watch training in real-time
python scripts/monitor_training.py

# Or check summary
python scripts/monitor_training.py --summary
```

### 4. After Training: Test the Model

```bash
# Quick generation test
python scripts/quick_gen_test.py

# Full evaluation suite (tests syntax, code completion, etc.)
python scripts/full_evaluation.py

# Or test generation interactively
python pretraining/generate.py
```

## 📊 What to Expect

### Before Training
```
Output: "O n ce u p on a t i me in T or on to , C h in"
Perplexity: ~248 (completely confused)
Tests Passed: 0%
```

### After Training
```
Output: "Once upon a time in Toronto, there was a small"
Perplexity: <100 (ideally <50)
Tests Passed: >50% syntax understanding, >30% code generation
```

## 🏗️ Project Architecture

```
ML-Gym/
├── data/
│   ├── raw/                      # Original data
│   │   ├── code/                 # Python code examples
│   │   └── books/                # Text books for language
│   └── processed/
│       ├── code_corpus.txt       # All code combined (~284 lines)
│       └── mixed_corpus.txt      # Training data: 3:1 code:books (~4,882 lines)
│
├── model/
│   ├── architecture/
│   │   ├── model.py             # Main GPT model
│   │   ├── layers.py            # Attention & feed-forward layers
│   │   └── rotary.py            # Positional embeddings
│   └── config/
│       ├── gpt_tiny_config.json    # 27M params (recommended for MacBook)
│       └── gpt_small_config.json   # 46M params (slower but better)
│
├── pretraining/
│   ├── train_improved.py        # Main training script ⭐
│   ├── dataset.py               # Data loader
│   ├── generate.py              # Text generation
│   └── checkpoints/             # Saved model weights
│
├── tokenizer/
│   ├── tokenizer_8k.json           # BPE tokenizer (fixed version)
│   ├── train_tokenizer_improved.py # Retrain tokenizer if needed
│   └── verify_token.py             # Test tokenizer
│
├── evaluation/
│   ├── benchmark.py                # YOUR original evaluation ⭐ (keep this!)
│   ├── results/
│   │   └── baseline_v0.json        # YOUR baseline results (before fixes)
│   ├── plots/
│   │   └── benchmark_v0.png        # YOUR baseline plot (before fixes)
│   ├── code_tests.py               # New: Code-specific tests
│   ├── evaluate_checkpoints.py     # New: Compare all checkpoints
│   └── pytest_code_generation.py   # New: Test if code actually runs
│
└── scripts/
    ├── status.py                   # Check project status ⭐
    ├── monitor_training.py         # Watch training progress ⭐
    ├── full_evaluation.py          # Run all tests ⭐
    ├── create_mixed_corpus.py      # Regenerate training data
    ├── test_tokenizer_quality.py   # Check tokenizer
    └── compare_model_sizes.py      # Compare configs
```

## 🎛️ How to Improve the Model

### Option 1: Get More/Better Training Data

**Add more code files:**
```bash
# 1. Put Python files in data/raw/code/
cp ~/my_project/*.py data/raw/code/

# 2. Regenerate corpus
python scripts/create_mixed_corpus.py

# 3. Retrain
python pretraining/train_improved.py
```

**Adjust code-to-text ratio:**
```python
# Edit scripts/create_mixed_corpus.py
# Change: create_mixed_corpus(code_weight=3)
# To:     create_mixed_corpus(code_weight=5)  # More code focus

python scripts/create_mixed_corpus.py
```

### Option 2: Train Longer

```python
# Edit pretraining/train_improved.py line 124
# Change: 'max_steps': 20000
# To:     'max_steps': 40000

# Then train
python pretraining/train_improved.py
```

### Option 3: Adjust Learning Rate

```python
# Edit pretraining/train_improved.py line 127
# If loss is jumpy/not decreasing:
'learning_rate': 1e-4    # Current
# Change to:
'learning_rate': 5e-5    # More conservative

# If loss decreasing too slowly:
'learning_rate': 3e-4    # More aggressive
```

### Option 4: Use Larger Model

```python
# Edit pretraining/train_improved.py line 119
# Change: 'config_path': 'model/config/gpt_tiny_config.json'
# To:     'config_path': 'model/config/gpt_small_config.json'

# Note: 46M params, ~2x slower training
```

### Option 5: Change Data Mix

**Try pure code (no books):**
```python
# Edit pretraining/train_improved.py line 117
# Change: 'corpus_file': 'data/processed/mixed_corpus.txt'
# To:     'corpus_file': 'data/processed/code_corpus.txt'
```

**Add your own text file:**
```python
# Put file in data/processed/my_corpus.txt
# Edit pretraining/train_improved.py line 117
'corpus_file': 'data/processed/my_corpus.txt'
```

### Option 6: Adjust Generation Quality

```python
# Edit pretraining/generate.py lines 169-177
output = generate(
    model, tokenizer, device,
    prompt="def fibonacci(n):",
    max_new_tokens=100,
    temperature=0.8,    # Lower = more conservative (0.5)
                        # Higher = more creative (1.2)
    top_k=40,           # Higher = more diverse (100)
    top_p=0.9           # Lower = more focused (0.8)
)
```

## 🔬 Evaluation & Testing

### Your Original Baseline (What You Created)

```bash
# Your original evaluation system
python evaluation/benchmark.py
```

**Your baseline results** (`evaluation/results/baseline_v0.json`):
- **Perplexity**: 247.96 (with broken tokenizer)
- **All tests**: Failed (0/4 passed)
- **Output**: "In the c it y of T or on to , b in ent i re system..."
- **Visualization**: `evaluation/plots/benchmark_v0.png`

This is your "before" benchmark - keep it to compare improvements!

### New Evaluation Tools (What I Added)

```bash
# Comprehensive code evaluation
python evaluation/code_tests.py

# Test if generated code actually runs
python evaluation/pytest_code_generation.py

# Compare all saved checkpoints
python evaluation/evaluate_checkpoints.py

# Check tokenizer quality
python scripts/test_tokenizer_quality.py

# Run everything + create comparison visualizations
python scripts/full_evaluation.py
```

### Visualizing Improvements

**Compare before/after:**
```bash
# 1. Run new evaluation
python evaluation/code_tests.py

# 2. Compare results
# Your baseline: evaluation/results/baseline_v0.json
# New results:   evaluation/results/code_eval.json

# 3. View plots
# Your baseline plot: evaluation/plots/benchmark_v0.png
# Training progress:  logs/training_log.json (can be plotted)
```

**Create comparison visualization:**
```python
# Quick script to compare your baseline vs new model
import json
import matplotlib.pyplot as plt

# Load your original baseline
with open('evaluation/results/baseline_v0.json') as f:
    baseline = json.load(f)

# Load new results (after training with improvements)
with open('evaluation/results/code_eval.json') as f:
    improved = json.load(f)

# Compare perplexity
print(f"Baseline perplexity: {baseline['perplexity']:.2f}")
print(f"Improved perplexity: {improved['code_perplexity']['average_perplexity']:.2f}")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Perplexity comparison
ax1.bar(['Baseline\n(broken tokenizer)', 'Improved\n(fixed tokenizer)'], 
        [baseline['perplexity'], improved['code_perplexity']['average_perplexity']])
ax1.set_ylabel('Perplexity (lower is better)')
ax1.set_title('Perplexity Improvement')

# Test scores comparison
baseline_tests = [baseline['bracket_test'], baseline['repeat_test'], 
                  baseline['alphabet_test'], baseline['sentence_test']]
improved_tests = [improved['bracket_completion']['score'],
                  improved['keyword_completion']['score'],
                  improved['indentation_awareness']['score'],
                  improved['function_completion']['score']]

x = range(4)
ax2.bar([i-0.2 for i in x], baseline_tests, 0.4, label='Baseline', alpha=0.7)
ax2.bar([i+0.2 for i in x], improved_tests, 0.4, label='Improved', alpha=0.7)
ax2.set_ylabel('Pass Rate')
ax2.set_title('Test Scores Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(['Brackets', 'Keywords', 'Indentation', 'Functions'], rotation=45)
ax2.legend()
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('evaluation/plots/before_after_comparison.png', dpi=200)
print("\nSaved comparison plot to: evaluation/plots/before_after_comparison.png")
```

### Understanding Results

**Perplexity** (main metric):
- `<50` = Excellent (coherent, confident predictions)
- `50-100` = Good (reasonable text generation)
- `100-200` = Okay (needs improvement)
- `>200` = Poor (like your baseline with broken tokenizer)

**Test Scores**:
- Syntax tests: `>50%` is good
- Generation tests: `>30%` is good
- Code execution: Any passing tests is progress!

**Your Baseline vs Target:**
| Metric | Your Baseline | Target After Training |
|--------|---------------|----------------------|
| Perplexity | 247.96 | <100 (ideally <50) |
| Bracket Test | 0% | >50% |
| Code Generation | Gibberish | Coherent code |
| Output Quality | "O n ce" spacing | Proper words |

## 📝 Key Files You'll Edit

### Most Common Edits

**1. Training Configuration** (`pretraining/train_improved.py`)
```python
Line 117: corpus_file        # What data to train on
Line 119: config_path        # Model size
Line 124: max_steps          # How long to train
Line 125: batch_size         # Memory usage
Line 127: learning_rate      # Training speed/stability
Line 133: eval_interval      # How often to validate
Line 134: checkpoint_interval # How often to save
```

**2. Model Size** (`model/config/gpt_tiny_config.json`)
```json
{
  "n_layers": 6,           // More = smarter but slower
  "n_heads": 8,            // Usually keep this
  "hidden_size": 512,      // Bigger = more capacity
  "max_seq_len": 1024,     // Longer context (memory²)
  "dropout": 0.1           // Regularization
}
```

**3. Generation Settings** (`pretraining/generate.py`)
```python
Line 166: prompt             # What to generate from
Line 174: max_new_tokens     # Length of generation
Line 175: temperature        # Randomness
Line 176: top_k              # Diversity
Line 177: top_p              # Quality filter
```

## 🐛 Troubleshooting

### "Model outputs gibberish"
```bash
# 1. Check tokenizer
python scripts/test_tokenizer_quality.py
# Should show good compression (~3-4 chars/token)

# 2. Check if training happened
python scripts/status.py
# Should show checkpoints and steps completed

# 3. Verify checkpoint loaded
python scripts/quick_gen_test.py
# Should show checkpoint step and loss
```

### "Training not starting"
```bash
# Check if already running
python scripts/status.py

# Check data exists
ls -lh data/processed/mixed_corpus.txt

# Try running directly
python pretraining/train_improved.py
```

### "Loss not decreasing"
```python
# Try these in order:
1. Lower learning rate by 3x
   # Edit train_improved.py: 1e-4 → 3e-5

2. Check data loaded correctly
   # Should see "[Dataset] Loading from corpus file" message

3. Reduce model size
   # Use gpt_tiny_config.json instead of gpt_small

4. Increase gradient clipping
   # Edit train_improved.py line 129: 1.0 → 0.5
```

### "Training too slow"
```python
# Speed up:
1. Reduce batch_size: 8 → 4
   # Less memory, slightly slower per step

2. Reduce block_size: 128 → 64
   # Shorter sequences = 4x faster

3. Use smaller model
   # gpt_tiny (6 layers) vs gpt_small (12 layers)

4. Reduce max_steps: 20000 → 10000
   # Less training, but faster to test
```

### "Out of memory"
```python
# Reduce memory usage:
1. batch_size: 8 → 4 or 2
2. block_size: 128 → 64
3. Use gpt_tiny_config.json
4. Close other applications
```

## 💡 Experiments to Try

### Beginner Experiments
```bash
# 1. Train on different data mixes
python scripts/create_mixed_corpus.py  # Change code_weight
python pretraining/train_improved.py

# 2. Try different temperatures
# Edit generate.py, test with 0.5, 0.8, 1.2

# 3. Compare checkpoints
python evaluation/evaluate_checkpoints.py
# Which training step was best?
```

### Intermediate Experiments
```python
# 1. Implement early stopping
# Stop training when validation loss increases

# 2. Try different LR schedules
# Linear decay instead of cosine

# 3. Add data augmentation
# Random dropout of tokens, paraphrasing

# 4. Fine-tune on specific task
# Load checkpoint, train on only function definitions
```

### Advanced Experiments
```python
# 1. Implement LoRA (parameter-efficient fine-tuning)
# Add adapter layers instead of training all weights

# 2. Add reinforcement learning
# Reward model when generated code passes tests

# 3. Multi-task training
# Train on code + docs + conversations simultaneously

# 4. Implement Flash Attention
# Faster attention mechanism for longer contexts
```

## 📦 Your Original Work (Preserved)

Your baseline evaluation is **important** - it shows the "before" state and demonstrates the improvement!

**What you created:**
```
evaluation/
├── benchmark.py              # Your evaluation script
├── results/
│   └── baseline_v0.json      # Perplexity: 247.96, all tests failed
└── plots/
    └── benchmark_v0.png      # Visual of test results
```

**Why it's valuable:**
- Shows model performance with broken tokenizer
- Demonstrates the "O n ce" spacing problem
- Provides baseline for comparison
- Documents the debugging journey

**Keep running it:**
```bash
# You can still run your original benchmark anytime
python evaluation/benchmark.py

# It will test the current model state
# Compare results to baseline_v0.json
```

**Use it for comparison:**
```bash
# After training, create visual comparison
python scripts/compare_baseline.py

# This uses your baseline_v0.json as the "before"
# And shows improvement with the fixed model
```

## 📚 Understanding the Output

### During Training
```
Step 1000: loss = 3.5, ppl = 33.1, lr = 9.8e-05
          │        │      │           │
          │        │      │           └─ Current learning rate
          │        │      └─ Perplexity (lower is better)
          │        └─ Loss (lower is better)
          └─ Training step (0-20,000)
```

### In Generated Text
```python
# Good signs:
✓ Complete words (not "O n ce")
✓ Proper spacing
✓ Code syntax looks right (indentation, colons, etc.)
✓ Stays on topic

# Warning signs:
⚠ Repeats same phrase over and over
⚠ Random characters or spacing
⚠ Completely off-topic
⚠ Syntax errors in every line
```

## 🎓 What You're Learning

By working with this project, you're learning:

**ML Engineering:**
- Data pipeline creation
- Training loop design
- Hyperparameter tuning
- Model evaluation
- Debugging ML systems

**LLM-Specific:**
- Tokenization (BPE, ByteLevel encoding)
- Transformer architecture
- Attention mechanisms
- Generation strategies
- Perplexity metrics

**Practical Skills:**
- Systematic experimentation
- Performance optimization
- Code organization
- Documentation

## 🚀 Next Steps

**After your first successful training:**

1. **Analyze results**
   ```bash
   python scripts/full_evaluation.py
   ```

2. **Try improvements** (pick one):
   - Add more code data
   - Train longer (40K steps)
   - Adjust learning rate
   - Scale up model

3. **Experiment with generation**
   ```bash
   python pretraining/generate.py
   # Try different prompts and settings
   ```

4. **Share your results**
   - Document what worked/didn't
   - Compare different approaches
   - Build portfolio piece

## 📞 Quick Reference

```bash
# Essential commands
python scripts/status.py                    # Check everything
python pretraining/train_improved.py        # Train model
python scripts/monitor_training.py          # Watch training
python scripts/full_evaluation.py           # Test model
python pretraining/generate.py              # Generate text

# Your original evaluation (keep this!)
python evaluation/benchmark.py              # Your baseline benchmark
# Results: evaluation/results/baseline_v0.json
# Plot:    evaluation/plots/benchmark_v0.png

# Comparison & visualization
python scripts/compare_baseline.py          # Compare before/after (VISUAL!)
python evaluation/code_tests.py             # New comprehensive tests

# Debugging
python scripts/test_tokenizer_quality.py    # Check tokenizer
python scripts/quick_gen_test.py            # Quick test

# Utilities
python scripts/create_mixed_corpus.py       # Regenerate data
python scripts/compare_model_sizes.py       # Compare configs
python evaluation/evaluate_checkpoints.py   # Compare checkpoints
```

## 📊 Visualizing Your Progress

After training completes, create a visual comparison:

```bash
# This creates a before/after visualization
python scripts/compare_baseline.py
```

This will generate `evaluation/plots/before_after_comparison.png` showing:
- Perplexity improvement (247.96 → <100)
- Test score improvements (0% → >50%)
- Generation quality comparison
- Overall performance metrics

**Your files to keep:**
- `evaluation/results/baseline_v0.json` - Your original results
- `evaluation/plots/benchmark_v0.png` - Your original plot
- These show "before" state with broken tokenizer

## 🤝 Contributing

This is a learning project! Experiment, break things, and learn. Common improvements:
- Add more diverse code data
- Implement new evaluation metrics
- Try different architectures
- Optimize training speed
- Better generation strategies

## 📄 License

Educational project for learning ML/LLM engineering.

---

**Current Status**: Check with `python scripts/status.py`

**Need Help?** Review the troubleshooting section or check your training logs in `logs/training_log.json`
