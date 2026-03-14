"""Lesson 4: The Transformer (Vaswani et al., 2017).

In 2017, "Attention Is All You Need" introduced self-attention as
a replacement for recurrence, enabling the architecture behind
modern large language models.

Run: uv run python lessons/04_transformer.py
"""

import os

from modelwerk.primitives.random import create_rng
from modelwerk.data.text import SHAKESPEARE_SONNETS, build_vocab, prepare_sequences
from modelwerk.models.transformer import (
    create_transformer_lm, train, generate, transformer_forward,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modelwerk.viz.attention_maps import plot_attention_weights


def _count_transformer_params(model):
    """Count total parameters in the transformer."""
    total = 0
    # Token embedding table
    total += len(model.token_emb.table) * len(model.token_emb.table[0])
    # Attention projections: W_q, W_k, W_v, W_o
    for W in [model.attention.W_q, model.attention.W_k, model.attention.W_v, model.attention.W_o]:
        total += len(W) * len(W[0])
    # Feed-forward
    total += len(model.ff_expand.weights) * len(model.ff_expand.weights[0]) + len(model.ff_expand.biases)
    total += len(model.ff_contract.weights) * len(model.ff_contract.weights[0]) + len(model.ff_contract.biases)
    # Output projection
    total += len(model.output_proj.weights) * len(model.output_proj.weights[0]) + len(model.output_proj.biases)
    return total


def main():
    print("=" * 60)
    print("  LESSON 4: THE TRANSFORMER")
    print("  Vaswani, Ashish, et al., 2017")
    print("=" * 60)

    print("""
In Lesson 3 the CNN learned to recognize digits by sliding a fixed
5x5 window across the image. Each position sees exactly 25 pixels —
the filter's receptive field (the region of the input it can "see")
is local and fixed. A vertical edge at pixel (5,5) and a curve at
pixel (20,20) are processed the same way, but the filter can never
look at both simultaneously. For images this is fine: nearby pixels
matter most.

But what about sequences? Consider predicting the next word:

    "The cat sat on the ___"

The answer ("mat", "floor", etc.) depends on "cat" and "sat" —
words at variable distances. A 5x5 window sized for local context
would miss these dependencies. You could stack many layers to grow
the receptive field, but that's indirect: the signal has to pass
through every intermediate layer in the network.

Before transformers, the dominant approach for sequences was the
recurrent neural network (RNN). An RNN processes tokens one at a
time, maintaining a hidden state that carries information forward.
A token here means a single unit of input — in our case, one
character. (In production systems, tokens are often subword pieces:
"attention" might become "atten" + "tion". For this lesson, one
character = one token.)

    h_0 -> h_1 -> h_2 -> h_3 -> ... -> h_n

This is a sequential bottleneck: token 50 must wait for tokens 1-49
to be processed. Worse, gradients flowing backward through many
time steps shrink exponentially — the same vanishing gradient
problem from Lesson 2, but now across sequence positions instead of
network layers. By the time the gradient reaches token 1, the signal
from token 50 has nearly vanished. Long Short-Term Memory networks
(LSTMs) and Gated Recurrent Units (GRUs) added gating mechanisms to
help, but the fundamental sequential bottleneck remained.

What if, instead of processing tokens sequentially and hoping
information survives the journey, each token could directly look
at every other token and decide what's relevant?

The transformer's insight is self-attention. To explain what that
means, we need one new concept: a position. In our previous
networks, an input was a single vector: pixel values, logic gate inputs.
In a transformer, the input is a sequence of vectors, one per token.
Each vector in that sequence sits at a numbered position: position 0
is the first token, position 1 is the second, and so on. When we
say "position 3 attends to position 1," we mean the token at index 3
is pulling information from the token at index 1.

Self-attention lets each position look at every other position and
learn which ones matter, with the pattern changing based on the
input. Instead of a fixed window or sequential processing, each
position directly computes a relevance score against every other
position. "Attending" simply means "deciding how much to focus on"
— position 3 might attend strongly to position 1 (high score,
pulls lots of information) and barely attend to position 2 (low
score, mostly ignores it).

Here's how it works. Each token is represented as a vector (its
embedding). Self-attention transforms each token's embedding by
computing three vectors from it:

    Query (Q): what this position is looking for in other positions
    Key (K):   what this position offers to other positions
    Value (V): the actual information this position carries

The names come from a database analogy: a query is matched against
keys to find relevant entries, then the corresponding values are
retrieved. Here the "lookup" is soft — instead of finding one exact
match, every position gets a weighted blend of all values, with
weights determined by how well each query matches each key.

For each position, its query vector is compared (via dot product)
against the key vector of every other position. High dot product
means "these two positions are relevant to each other." The scores
are normalized with softmax (so they sum to 1), then used as weights
to compute a weighted average of the value vectors. The result: each
position's output is a blend of information from across the sequence,
weighted by learned relevance.

Concretely, for tokens ["t", "h", "e", " "] with d_model=3
(d_model is the size of each token's vector — how many numbers
represent each character):

Step 1 — Embeddings (looked up from a table, 4x3):
    t: [0.12, -0.34,  0.56]
    h: [0.78,  0.23, -0.11]
    e: [-0.45, 0.67,  0.33]
    " ": [0.01, -0.89, 0.44]

Step 2 — Project to Q, K, V via weight matrices:
    Q = embeddings @ W_q    (each position gets a query vector)
    K = embeddings @ W_k    (each position gets a key vector)
    V = embeddings @ W_v    (each position gets a value vector)

Step 3 — Attention scores (Q @ K^T / sqrt(d_k)):
    score[i][j] = dot(Q[i], K[j]) / sqrt(3)

    This gives a 4x4 matrix — every position scored against every
    other. Dividing by sqrt(d_k) prevents scores from growing too
    large, which would push softmax into near-zero gradient regions.

Step 4 — Causal mask (so the model can't cheat by seeing the future):
    When generating text, the model predicts one token at a time. It
    shouldn't see the answer before guessing. We enforce this by
    setting score[i][j] = -infinity for j > i, so each position can
    only attend to itself and earlier positions. After softmax, the
    masked entries become zero:

          t     h     e     _
    t  [  .   -inf  -inf  -inf ]   t can only see t
    h  [  .     .   -inf  -inf ]   h can see t, h
    e  [  .     .     .   -inf ]   e can see t, h, e
    _  [  .     .     .     .  ]   _ can see all

Step 5 — Softmax (row-wise):
    weights[i] = softmax(scores[i])

    For position e (row 2), suppose after masking:
    scores = [1.2, 0.8, 0.5]  (only first 3 visible)
    weights = [0.45, 0.30, 0.25]  — "e" attends 45% to "t"

Step 6 — Weighted sum of values:
    output[i] = sum_j(weights[i][j] * V[j])

    For position e: 0.45*V[t] + 0.30*V[h] + 0.25*V[e]
    "Position 2 attended 45% to position 0 — it learned that
    the 't' is the most relevant context for predicting what
    follows 'the'."

That's the core mechanism. In our actual model, d_model=32 (not 3),
so the vectors are larger but the process is identical.

Multi-head attention runs this entire Q/K/V process multiple times
in parallel with different weight matrices. With 2 heads, each head
works on d_k = d_model/2 = 16 dimensions. One head might learn to
attend to nearby characters (local patterns like "th" followed by
"e"), while another learns longer-range dependencies (matching
quotes, repeating phrases). The head outputs are concatenated and
projected back to d_model dimensions.

Positional encoding: since attention compares every position to
every other position regardless of order (it's a set operation),
the model has no way to know that position 0 comes before
position 1. We fix this by adding sinusoidal position signals to
the embeddings before they enter attention:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Each position gets a unique pattern. The sinusoidal form lets the
model learn to attend to relative positions: PE(pos+k) can be
expressed as a linear function of PE(pos).

This lesson has more moving parts than the previous three. The core
idea is attention — everything else (positional encoding, layer norm,
residuals) is supporting infrastructure that makes it trainable.
Here's how it all fits together:

The full architecture:

    Input         Embed +     ┌─── Transformer Block ───────────┐     Output
    (tokens)      Pos Enc     │                                 │
                              │  LayerNorm → Attention → + Res  │
    "t" ─┐      ┌───────┐     │  LayerNorm → FF(dense) → + Res  │     ┌─────┐
    "h" ─┤─────>│ embed │───> │                                 │ ──> │Dense│──> softmax
    "e" ─┤      │  + PE │     │  (attention lets each position  │     │ 32  │    probs
    " " ─┘      └───────┘     │   look at all prior positions)  │     │->48 │
                              └─────────────────────────────────┘     └─────┘

    seq_len=32    d_model=32    2 attn heads     FF: 32->64->32         vocab=48

Reading left to right:

  Input: a sequence of 32 character tokens, each represented as
  an integer ID (e.g., 'a'=0, 'b'=1, ...).

  Embedding + positional encoding: each token ID is looked up in a
  table to get a 32-dimensional vector, then a sinusoidal position
  signal is added so the model knows where each token sits.

  Transformer block: the core of the architecture. Two sublayers:
  (1) multi-head self-attention — each position computes relevance
  scores against all prior positions and blends their information;
  (2) feed-forward — the same dense layers from Lesson 2, applied
  independently at each position. These use ReLU instead of tanh —
  a simpler activation that avoids the vanishing gradient problem
  (ReLU's derivative is 1 for positive inputs, vs tanh's derivative
  that shrinks toward 0). Each sublayer has a residual
  connection (adding the input back, so the layer only needs to
  learn the change) and layer normalization (scaling to zero mean
  and unit variance to stabilize training).

  Output: a dense layer projects from d_model=32 to vocab_size=48
  (one score per character), then softmax converts to probabilities.

We use pre-norm (normalize before each sublayer), which is easier
to train than the original paper's post-norm. Our layer norm omits
learnable scale/shift parameters — the following dense layers
absorb any needed rescaling.

Simplifications vs the original "Attention Is All You Need":
  - 1 block instead of 6 (keeps training fast in pure Python)
  - Pre-norm instead of post-norm (modern practice)
  - No learnable layer norm params (gamma/beta omitted)
  - SGD instead of Adam (matches prior lessons)
  - Decoder-only (GPT-style) instead of encoder-decoder
  - Character-level instead of subword tokenization

Let's see it learn Shakespeare, one character at a time.
""")

    # --- PART 1: THE DATA ---
    print(f"{'='*60}")
    print("  PART 1: THE DATA")
    print(f"{'='*60}")

    corpus = SHAKESPEARE_SONNETS
    char_to_id, id_to_char = build_vocab(corpus)
    vocab_size = len(char_to_id)

    print(f"""
  Corpus: 4 Shakespeare sonnets (18, 29, 73, 130)
  Characters: {len(corpus)}
  Vocabulary: {vocab_size} unique characters

  First 200 characters:
  {repr(corpus[:200])}

  Character vocabulary:""")
    chars_display = ""
    for ch in sorted(char_to_id.keys()):
        display = repr(ch) if ch in ("\n", " ", "'") else ch
        chars_display += f" {display}"
    print(f"  {chars_display}")

    seq_len = 32
    inputs, targets = prepare_sequences(corpus, char_to_id, seq_len)
    print(f"""
  Sequence length: {seq_len} characters
  Training sequences: {len(inputs)} (sliding window, stride 1)

  Example input:  "{corpus[:seq_len]}"
  Example target: "{corpus[1:seq_len+1]}"
  (target is input shifted right by one character)
""")

    # --- PART 2: THE ARCHITECTURE ---
    print(f"{'='*60}")
    print("  PART 2: THE ARCHITECTURE")
    print(f"{'='*60}")

    rng = create_rng(42)
    d_model = 32
    num_heads = 2
    d_k = d_model // num_heads
    ff_dim = 64

    model = create_transformer_lm(
        rng, vocab_size=vocab_size, d_model=d_model,
        num_heads=num_heads, ff_dim=ff_dim, seq_len=seq_len,
    )
    total_params = _count_transformer_params(model)

    emb_params = vocab_size * d_model
    attn_params = 4 * d_model * d_model
    ff_params = d_model * ff_dim + ff_dim + ff_dim * d_model + d_model
    out_params = d_model * vocab_size + vocab_size

    print(f"""
  Model architecture (decoder-only transformer):
    Token embedding ({vocab_size}x{d_model})        {emb_params:>6} params
    Attention (Q,K,V,O: {d_model}x{d_model})    {attn_params:>6} params
      {num_heads} heads, d_k={d_k}
    Feed-forward ({d_model}->{ff_dim}->{d_model})        {ff_params:>6} params
    Output projection ({d_model}->{vocab_size})       {out_params:>6} params
    Total:                          {total_params:>6} params

  Compare: LeNet-5 had ~3,700 params for 10-class images.
  This transformer has {total_params:,} params for {vocab_size}-class
  character prediction — a similar scale.
""")

    # --- PART 3: TRAINING ---
    print(f"{'='*60}")
    print("  PART 3: TRAINING")
    print(f"{'='*60}")

    # Subsample sequences for tractable training
    # Take every Nth sequence to cover the corpus while keeping training short
    stride = max(1, len(inputs) // 200)
    train_inputs = inputs[::stride]
    train_targets = targets[::stride]

    learning_rate = 0.01
    epochs = 30
    print(f"""
  Training: lr={learning_rate}, epochs={epochs}
  Sequences: {len(train_inputs)} (subsampled from {len(inputs)})
  Each sequence: {seq_len} characters predicting next character

  Training (this takes a few minutes in pure Python)...
""")

    loss_history = train(
        model, train_inputs, train_targets,
        vocab_size=vocab_size,
        learning_rate=learning_rate, epochs=epochs,
    )

    print(f"\n  Training results:")
    for i in range(0, len(loss_history), max(1, len(loss_history) // 5)):
        print(f"    Epoch {i+1:>3}: loss={loss_history[i]:.4f}")
    print(f"    Epoch {len(loss_history):>3}: loss={loss_history[-1]:.4f}")

    # Show a few predictions
    print(f"\n  Sample predictions (input -> predicted next char):")
    sample_indices = [0, len(train_inputs) // 4, len(train_inputs) // 2]
    for idx in sample_indices:
        seq = train_inputs[idx]
        probs, _ = transformer_forward(model, seq)
        text_in = "".join(id_to_char[t] for t in seq)
        # Show prediction at last position
        pred_id = max(range(vocab_size), key=lambda i: probs[-1][i])
        actual_id = train_targets[idx][-1]
        pred_ch = repr(id_to_char[pred_id])
        actual_ch = repr(id_to_char[actual_id])
        mark = "correct" if pred_id == actual_id else f"expected {actual_ch}"
        print(f'    "...{text_in[-20:]}" -> {pred_ch}  ({mark})')

    # --- PART 4: GENERATION ---
    print(f"\n{'='*60}")
    print("  PART 4: GENERATION")
    print(f"{'='*60}")

    prompt = "Shall I compare"
    prompt_ids = [char_to_id[ch] for ch in prompt if ch in char_to_id]

    print(f'\n  Prompt: "{prompt}"')
    print(f"  Generating 100 characters...\n")

    gen_rng = create_rng(123)
    generated_text, last_attn_weights = generate(
        model, prompt_ids, length=100, id_to_char=id_to_char,
        temperature=0.8, rng=gen_rng,
    )

    print(f"  Generated text:")
    # Show in a box
    print(f"  ┌{'─'*56}┐")
    # Word-wrap at 54 chars
    text = generated_text
    while text:
        line = text[:54]
        text = text[54:]
        print(f"  │ {line:<54} │")
    print(f"  └{'─'*56}┘")

    print("""
  The output won't be coherent Shakespeare — this is a tiny model
  trained on ~1.8KB of text. But you should see it learning character-
  level patterns: common letter sequences, spacing, punctuation,
  and fragments of words from the sonnets.
""")

    # --- Plots ---
    os.makedirs("output", exist_ok=True)

    # Training loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, epochs + 1), loss_history, color="#5CB8B2", linewidth=2, marker="o")
    ax.set_title("Transformer Training Loss (Shakespeare Sonnets)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    plt.tight_layout()
    fig.savefig("output/04_transformer_training.png", dpi=150)
    plt.close(fig)
    print("  Training plot saved to ./output/04_transformer_training.png")

    # Attention heatmap from the last generation step
    if last_attn_weights:
        # Use the last seq_len tokens of the generated text for labels
        gen_ids = [char_to_id[ch] for ch in generated_text if ch in char_to_id]
        context_ids = gen_ids[-model.seq_len:]
        context_tokens = [id_to_char[tid] for tid in context_ids]
        fig = plot_attention_weights(
            last_attn_weights, context_tokens,
            title="Attention Weights (Last Generation Step)",
            filepath="output/04_transformer_attention.png",
        )
        plt.close(fig)
        print("  Attention map saved to ./output/04_transformer_attention.png")

    # --- WHAT CHANGED ---
    print(f"\n{'='*60}")
    print("  WHAT CHANGED")
    print(f"{'='*60}")

    print(f"""
The CNN looked through a fixed 5x5 window — a local, static
receptive field. The transformer learns which positions to attend
to, with the pattern changing based on input. This is the key
shift: from fixed spatial structure to dynamic, data-dependent
context.

What attention gives us:

  Global receptive field: every position can attend to every other
  position in a single layer. The CNN needed stacked layers to
  grow its receptive field.

  Interpretability: attention weights show what the model focuses
  on. In the heatmap above, brighter cells show stronger attention.
  You can see which characters the model considered when predicting
  the next one. CNNs and MLPs don't have such transparent internal
  states.

  Parallelism: unlike RNNs, attention processes all positions
  simultaneously. No sequential bottleneck. This is why transformers
  train efficiently on GPUs — and why they scaled to billions of
  parameters while RNNs did not.

The cost: attention computes a score for every pair of positions,
giving O(n^2) complexity in sequence length. For our 32-character
sequences, that's a 32x32 = 1,024-entry attention matrix. For
GPT-4's ~128,000-token context, that would be ~16 billion entries
per layer per head. The same quadratic cost that makes attention
powerful also makes it expensive — this is why techniques like
sliding window attention, sparse attention, and linear attention
are active research areas.

Backpropagation still works — same chain rule from Lesson 2.
The forward pass has more steps than LeNet, but each step uses
primitives you've seen: dense layers, softmax, addition. The
backward pass is where the complexity shows up — gradients flowing
through residual connections split into two paths (one through the
sublayer, one through the skip connection), and the softmax backward
involves all outputs simultaneously rather than independently. If
the backward code is hard to follow, focus on the forward pass and
treat the backward as "the chain rule applied to each step, working
right to left." The softmax + cross-entropy gradient simplification
(probs - target) that we used in Lesson 3 works here too.

The original paper used this architecture for machine translation
(encoder-decoder). Modern LLMs (GPT, Claude) use decoder-only
variants like the one here, scaled up enormously: billions of
parameters, terabytes of text, thousands of GPUs.

Our {total_params:,}-parameter model on 4 sonnets is the same
algorithm. The difference is scale.

Next: we've now built a perceptron, an MLP, a CNN, and a transformer
— each one introduced a new architectural idea. In Lesson 5 we'll
step back and look at the training process itself: how modern
optimizers, regularization, and scaling techniques turn these
building blocks into systems that actually work at scale.
""")

    print(f"{'='*60}")
    print("  END OF LESSON 4")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
