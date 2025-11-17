# Model Configuration Explained

This document explains each parameter in your `gpt_small_config.json` and how it shapes your custom LLM. Treat this as a readable companion to the configuration file.

---

## **vocab_size**

This defines how many unique tokens exist in your tokenizer. Each token has a corresponding embedding vector inside the model. A lower vocabulary keeps memory usage small and training fast.

---

## **n_layers**

This is the number of Transformer blocks stacked on top of each other. More layers allow deeper reasoning and richer representations, at the cost of training time.

---

## **n_heads**

Attention is split into multiple "heads". Each head learns to look at different relationships in the input sequence. More heads allow more parallel pattern recognition.

---

## **hidden_size**

The width of the model’s internal representations. This is the dimension of:

* Token embeddings
* Attention projections
* Residual streams

Larger hidden sizes increase intelligence but require more compute.

---

## **intermediate_size**

Inside each Transformer block is a feed‑forward network (an MLP). This setting controls the width of that MLP. It is typically 3–4× the hidden size. It allows the model to perform richer non‑linear transformations.

---

## **max_seq_len**

The maximum number of tokens the model can process in a single forward pass. A higher value supports longer inputs such as bug reports, code snippets, and diffs.

---

## **rotary_pct**

Controls what percentage of the attention dimensions use RoPE (rotary positional embeddings). RoPE helps preserve relative token positions and improves long‑range attention. A value of 1.0 applies RoPE to all heads.

---

## **rotary_emb_base**

A scaling constant used by the RoPE algorithm. Higher values improve stability when handling longer sequences. Most modern code models use 10,000.

---

## **dropout**

Dropout probability used to prevent overfitting by randomly zeroing parts of the network during training. Small models benefit from modest dropout.

---

## **bias**

Indicates whether linear layers include bias terms. Many modern efficient models disable biases to reduce parameter count and memory usage while maintaining performance.

---

This document should give you a clear grasp of what each configuration parameter does and why it matters when defining the brain of your LLM.
