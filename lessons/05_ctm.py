"""Lesson 5: The Continuous Thought Machine (Darlow et al., 2025).

Every network we've built so far runs one forward pass per input.
The CTM introduces an internal time dimension — "thought steps" —
where the network iterates, refining its answer over multiple ticks.

Run: uv run python lessons/05_ctm.py
"""

import os
import math

from modelwerk.primitives.random import create_rng
from modelwerk.primitives import vector
from modelwerk.data.generators import parity
from modelwerk.data.utils import one_hot
from modelwerk.models.ctm import (
    create_ctm, ctm_forward, ctm_loss, train, predict,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _count_ctm_params(model):
    """Count total parameters in the CTM."""
    total = 0
    # Synapse model
    total += len(model.synapse.W1) * len(model.synapse.W1[0]) + len(model.synapse.b1)
    total += len(model.synapse.W2) * len(model.synapse.W2[0]) + len(model.synapse.b2)
    # NLMs
    for d in range(model.d_model):
        total += len(model.nlm.W1[d]) * len(model.nlm.W1[d][0]) + len(model.nlm.b1[d])
        total += len(model.nlm.W2[d]) * len(model.nlm.W2[d][0]) + len(model.nlm.b2[d])
    # Synchronization decay rates
    total += model.pairs_out.n_pairs + model.pairs_action.n_pairs
    # Output projection
    total += len(model.W_out) * len(model.W_out[0]) + len(model.b_out)
    # Query projection
    total += len(model.W_query) * len(model.W_query[0])
    # Embeddings
    total += len(model.W_embed) * len(model.W_embed[0])
    # KV projection
    total += len(model.W_kv) * len(model.W_kv[0]) + len(model.b_kv)
    # Attention K, V
    total += len(model.W_attn_k) * len(model.W_attn_k[0])
    total += len(model.W_attn_v) * len(model.W_attn_v[0])
    # z_init
    total += len(model.z_init)
    return total


def main():
    print("=" * 60)
    print("  LESSON 5: THE CONTINUOUS THOUGHT MACHINE")
    print("  Darlow, et al., 2025")
    print("=" * 60)

    print("""
Every network we've built so far has the same limitation: given an
input, it runs one forward pass and produces one answer. A simple
input and a hard input get exactly the same amount of computation.
But humans don't work this way—we think longer about harder
problems. What if a network could do the same?

The Continuous Thought Machine gives the network an internal time
dimension. Instead of one forward pass, it runs T "thought steps."
At each tick, it:

    1. Produces pre-activations through a synapse model (an MLP)
    2. Processes each neuron's activation history through private
       per-neuron MLPs (neuron-level models, or NLMs)
    3. Measures temporal correlations between neurons' activities
       (neural synchronization)
    4. Uses synchronization to make predictions AND to decide what
       in the input to attend to next

The key innovations:

  Neuron-level models: each neuron has its own private weights that
  process a rolling history of its past activations. This gives
  every neuron its own temporal dynamics—like giving each neuron
  a small memory and a small brain to process that memory.

  Neural synchronization: instead of using neuron activations
  directly, the CTM measures how pairs of neurons' activities
  correlate over time. These correlations—computed as a recursive
  weighted dot product—become the representation the network
  uses for everything: prediction, attention queries, decision-making.

  Certainty-based loss: the network predicts at every tick, but the
  loss only uses two ticks—the one with lowest loss and the one
  with highest certainty. This naturally gives adaptive compute:
  the network learns to become certain sooner on easy inputs and
  to keep thinking on hard ones.
""")

    # --- PART 1: THE DATA ---
    print(f"{'='*60}")
    print("  PART 1: THE DATA")
    print(f"{'='*60}")

    rng_data = create_rng(42)
    seq_len = 16
    n_train = 200
    n_test = 20
    inputs_train, targets_train = parity(rng_data, seq_len=seq_len, n_samples=n_train)
    inputs_test, targets_test = parity(rng_data, seq_len=seq_len, n_samples=n_test)

    print(f"""
  Task: Parity: predict the cumulative parity of ±1 sequences.

  Why parity? It's a task where the answer depends on the entire input —
  every ±1 value affects the running product. A feedforward network sees
  the whole sequence at once but has no way to iteratively track the
  running sign. The CTM's tick loop gives it exactly that: each thinking
  step can refine its running estimate. Difficulty also scales naturally
  with sequence length, exercising the adaptive compute mechanism.
  Darlow et al. use parity as a benchmark for these reasons.

  Each input is a sequence of +1 or -1 values.
  The target at each position is 1 if the running product is
  positive, 0 otherwise.

  Example:
    Input:   {inputs_train[0][:8]}
    Target:  {targets_train[0][:8]}
    (product flips sign with each -1)

  Training samples: {n_train}
  Test samples:     {n_test}
  Sequence length:  {seq_len}
  Output classes:   2 (parity 0 or 1 at the last position)
""")

    # --- PART 2: THE ARCHITECTURE ---
    print(f"{'='*60}")
    print("  PART 2: THE ARCHITECTURE")
    print(f"{'='*60}")

    rng = create_rng(123)
    d_model = 32
    d_input = 16
    d_embed = 8
    d_hidden_syn = 32
    d_hidden_nlm = 4
    M = 7
    T = 20
    num_classes = 2
    J_out = 8
    J_action = 8

    model = create_ctm(
        rng,
        d_model=d_model,
        d_input=d_input,
        d_embed=d_embed,
        d_hidden_syn=d_hidden_syn,
        d_hidden_nlm=d_hidden_nlm,
        M=M,
        T=T,
        num_classes=num_classes,
        seq_len=seq_len,
        J_out=J_out,
        J_action=J_action,
    )
    total_params = _count_ctm_params(model)

    # Count sub-components
    syn_params = (len(model.synapse.W1) * len(model.synapse.W1[0]) + len(model.synapse.b1)
                  + len(model.synapse.W2) * len(model.synapse.W2[0]) + len(model.synapse.b2))
    nlm_params = sum(
        len(model.nlm.W1[d]) * len(model.nlm.W1[d][0]) + len(model.nlm.b1[d])
        + len(model.nlm.W2[d]) * len(model.nlm.W2[d][0]) + len(model.nlm.b2[d])
        for d in range(d_model)
    )
    sync_params = model.pairs_out.n_pairs + model.pairs_action.n_pairs
    out_params = len(model.W_out) * len(model.W_out[0]) + len(model.b_out)
    attn_params = (len(model.W_query) * len(model.W_query[0])
                   + len(model.W_attn_k) * len(model.W_attn_k[0])
                   + len(model.W_attn_v) * len(model.W_attn_v[0]))
    embed_params = (len(model.W_embed) * len(model.W_embed[0])
                    + len(model.W_kv) * len(model.W_kv[0]) + len(model.b_kv))

    print(f"""
  Model architecture (Continuous Thought Machine):

    ┌─────── Internal Tick Loop (T={T} ticks) ────────┐
    │                                                   │
    │  Synapse: concat(z,o) → SiLU → a^t                │
    │    MLP: ({d_model}+{d_input}) → {d_hidden_syn} → {d_model}          {syn_params:>6} params│
    │                                                   │
    │  NLMs: per-neuron history → z^{{t+1}}               │
    │    {d_model} neurons × (M={M} → {d_hidden_nlm} → 1)         {nlm_params:>6} params│
    │                                                   │
    │  Sync: recursive α,β → S vectors                  │
    │    J_out={J_out}, J_action={J_action} (semi-dense)       {sync_params:>6} params│
    │                                                   │
    │  Output: S_out → logits → softmax         {out_params:>6} params│
    │  Attention: S_action → query, cross-attn   {attn_params:>6} params│
    └───────────────────────────────────────────────────┘

    M: memory window, each NLM sees its last {M} pre-activations
    J_out, J_action: neurons per sync group ({J_out}×{J_out}={J_out*J_out} pairs each)

    Embeddings & projections:                   {embed_params:>6} params
    Initial state (z_init):                     {d_model:>6} params
    Total:                                      {total_params:>6} params

  Compare: the transformer (Lesson 4) had 11,408 params.
  The CTM is comparable, but spends its parameters differently —
  much of the capacity is in the {d_model} independent NLMs.
""")

    print(f"""
  The CTM loop (simplified):

  ┌──────────────────────────────────────────────────────────┐
  │  for each tick t = 1..T:                                 │
  │    a^t  = Synapse(z^t, o^t)       ← pre-activations     │
  │    z^{{t+1}} = NLM_d(history of a)   ← per-neuron MLPs    │
  │    S^t  = Sync(z^{{t+1}})            ← temporal corr.     │
  │    y^t  = W_out · S^t_out          ← predict each tick   │
  │    q^t  = W_query · S^t_action     ← attention query     │
  │    o^t  = CrossAttention(q^t, KV)  ← re-read input       │
  └──────────────────────────────────────────────────────────┘

  Key insight: the output y^t is produced at EVERY tick, but the
  loss only selects two ticks: the best-loss tick (t1) and the
  most-certain tick (t2). The model learns when to "commit."
""")

    # --- PART 3: TRAINING ---
    print(f"{'='*60}")
    print("  PART 3: TRAINING")
    print(f"{'='*60}")

    learning_rate = 0.00005
    epochs = 200

    print(f"""
  Training: lr={learning_rate:.5f}, epochs={epochs}
  Samples: {n_train} sequences of length {seq_len}
  Internal ticks: {T} per sample
  Loss: certainty-based (min-loss tick + max-certainty tick)

  The learning rate is tiny because gradients flow through {T} ticks of
  the inner loop — similar to a {T}-layer-deep network. Without a very
  small step size the accumulated gradients blow up and training diverges.

  Optimizer: AdamW (adaptive learning rates per parameter)

  Lessons 1-4 used SGD — one learning rate for everything. That works
  when all parameters play similar roles. But the CTM has fundamentally
  different parameter types: decay rates (scalars controlling temporal
  memory), NLM weights ({d_model} private neural nets), synapse weights (shared
  MLP), and attention projections. SGD can't balance all of these — it
  plateaus at loss ~0.6 because the learning rate that works for one
  group is wrong for another.

  AdamW solves this by tracking each parameter's gradient history and
  adapting the step size automatically. Parameters with large gradients
  get smaller steps; parameters with small gradients get larger steps.

  Gradient clipping: disabled (max_norm=999). The CTM's tick loop creates
  recurrent gradient flow that can explode, but AdamW's per-parameter
  scaling handles this naturally.

  Training (this takes a few minutes in pure Python)...
""")

    epoch_losses = train(
        model, inputs_train, targets_train,
        learning_rate=learning_rate, epochs=epochs, max_norm=999.0,
        optimizer="adamw",
    )

    print(f"""
  Training results:""")
    for i, loss in enumerate(epoch_losses):
        if i == 0 or (i + 1) % 5 == 0 or i == len(epoch_losses) - 1:
            print(f"    Epoch {i+1:>3}: loss={loss:.4f}")

    # --- PART 4: EVALUATION ---
    print(f"""
{'='*60}
  PART 4: EVALUATION
{'='*60}""")

    # Evaluate on test set
    correct = 0
    total = 0
    tick_details = []
    for i in range(len(inputs_test)):
        target_class = int(targets_test[i][-1])
        target_vec = one_hot(target_class, num_classes)

        # Reset sync
        model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
        model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
        model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
        model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)

        per_tick_probs, cache = ctm_forward(model, inputs_test[i])
        loss, t1, t2, losses, certs = ctm_loss(per_tick_probs, target_vec)

        # Find most certain tick
        best_t = certs.index(max(certs))
        pred_probs = per_tick_probs[best_t]
        pred = pred_probs.index(max(pred_probs))
        if pred == target_class:
            correct += 1
        total += 1
        tick_details.append((t1, t2, best_t, certs))

    accuracy = correct / total * 100

    print(f"""
  Test accuracy: {correct}/{total} ({accuracy:.1f}%)

  (Training for 300 epochs pushes accuracy to ~95%. The extra 100
  epochs let AdamW settle into a lower basin — loss drops from ~0.35
  to ~0.28. Try it: set epochs=300 above.)

  Sample predictions (test set):""")
    for i in range(min(5, len(inputs_test))):
        target_class = int(targets_test[i][-1])
        t1, t2, best_t, certs = tick_details[i]
        print(f"    seq {i}: target={target_class}, t1={t1}, t2={t2}, "
              f"certainty@best={certs[best_t]:.3f}")

    # --- PART 5: ADAPTIVE COMPUTE ---
    print(f"""
{'='*60}
  PART 5: ADAPTIVE COMPUTE
{'='*60}

  The certainty-based loss means the model naturally develops
  adaptive compute — becoming certain sooner on easier inputs.

  Certainty over ticks (selected test samples):""")

    for i in range(min(3, len(inputs_test))):
        _, _, _, certs = tick_details[i]
        cert_str = "  ".join(f"{c:.2f}" for c in certs)
        print(f"    seq {i}: [{cert_str}]")

    # --- PLOTS ---
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Training loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CTM Training Loss (Parity Task)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(output_dir, "05_ctm_training.png")
    fig.savefig(loss_path, dpi=100)
    plt.close(fig)
    print(f"\n  Saved training loss plot: {loss_path}")

    # Certainty over ticks heatmap
    if tick_details:
        n_show = min(10, len(tick_details))
        cert_matrix = [tick_details[i][3] for i in range(n_show)]

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(cert_matrix, aspect="auto", cmap="viridis")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Test Sample")
        ax.set_title("Certainty Over Internal Ticks")
        fig.colorbar(im, ax=ax, label="Certainty")
        fig.tight_layout()
        cert_path = os.path.join(output_dir, "05_ctm_certainty.png")
        fig.savefig(cert_path, dpi=100)
        plt.close(fig)
        print(f"  Saved certainty heatmap: {cert_path}")

    # --- CLOSING ---
    print(f"""
{'='*60}
  WHAT CHANGED
{'='*60}

The transformer (Lesson 4) processes a sequence in one pass: each
position attends to other positions, but the network's depth is
fixed. The CTM adds a second kind of depth — internal time. The
network iterates T times, and each iteration can change what it
attends to and refine its answer.

What the CTM introduces:

  Internal time: the network "thinks" for T steps, each building
  on the previous. The synapse-NLM-sync loop is the core mechanism.

  Neuron-level models: instead of uniform activations, each of
  the {d_model} neurons has its own private MLP processing its own
  activation history. This means neurons can develop specialized
  temporal behaviors.

  Neural synchronization: the representation isn't neuron
  activations but their temporal correlations. Two neurons that
  fire in sync carry different information than two that alternate.
  This is computed recursively via weighted dot products with
  learnable decay rates.

  Adaptive compute: because the loss selects the best tick (not
  the last one), the model naturally learns to commit early on
  easy inputs and keep thinking on hard ones. No explicit halting
  mechanism needed — it emerges from the certainty-based loss.

The CTM composes everything from prior lessons:
  - The synapse model is an MLP (Lesson 2)
  - Cross-attention reads the input (Lesson 4)
  - The tick loop is the new idea

Our {total_params:,}-parameter model on parity is a minimal version of
the architecture. The paper uses 1024-dimensional models on tasks
ranging from image classification to maze navigation. The difference
is scale — and the same backpropagation through time (BPTT) we use
here scales to those larger settings.
""")

    print(f"{'='*60}")
    print("  END OF LESSON 5")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
