"""Lesson 5: Mamba — Selective State Space Models (Gu & Dao, 2023).

The transformer attends to all positions at every layer — O(L^2) in
sequence length. Mamba replaces attention with a selective state space
model that processes the sequence in O(L), deciding at each position
what to remember and what to ignore.

Run: uv run python lessons/05_mamba.py
"""

import os
import math

from modelwerk.primitives.random import create_rng
from modelwerk.data.generators import selective_copying
from modelwerk.models.mamba import (
    create_mamba_lm, mamba_forward, train, predict,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _count_params(model):
    """Count total parameters in the Mamba LM."""
    total = 0
    # Embedding
    total += len(model.embedding) * len(model.embedding[0])
    # Input projection
    total += len(model.block.in_proj) * len(model.block.in_proj[0])
    # Conv
    total += len(model.block.conv_weight) * len(model.block.conv_weight[0])
    total += len(model.block.conv_bias)
    # SSM
    total += len(model.block.ssm.A_log) * len(model.block.ssm.A_log[0])
    total += len(model.block.ssm.B_proj) * len(model.block.ssm.B_proj[0])
    total += len(model.block.ssm.C_proj) * len(model.block.ssm.C_proj[0])
    total += len(model.block.ssm.dt_proj_down) * len(model.block.ssm.dt_proj_down[0])
    total += len(model.block.ssm.dt_proj_up) * len(model.block.ssm.dt_proj_up[0])
    total += len(model.block.ssm.dt_bias)
    total += len(model.block.ssm.D)
    # Output projection
    total += len(model.block.out_proj) * len(model.block.out_proj[0])
    total += len(model.block.out_proj_bias)
    # LM head
    total += len(model.head) * len(model.head[0])
    total += len(model.head_bias)
    return total


def main():
    print("=" * 60)
    print("  LESSON 5: MAMBA (SELECTIVE STATE SPACE MODELS)")
    print("  Gu & Dao, 2023")
    print("=" * 60)

    print("""
The transformer (Lesson 4) looks at every position to process each
token — that's what self-attention does. It works brilliantly, but
the cost grows as O(L^2) with sequence length. Double the sequence,
quadruple the work.

What if a model could process sequences in O(L) — linear time —
while still deciding what to remember? That's Mamba's core idea.

Instead of attention, Mamba maintains a hidden state that gets
updated at each position. The key innovation: the update rule is
input-dependent. Three parameters control the update:

  B (input selection) — decides what features of the current token
    get written into the hidden state
  C (output read) — decides what gets read back from the hidden state
  Delta (step size) — controls HOW MUCH to write: the volume knob
    between "remember everything" and "ignore completely"

All three are computed from the current input, so the model can:

  - Spike Delta high at important tokens → write them into state
  - Keep Delta low at irrelevant tokens → let them pass through
  - Adjust B and C to control what gets stored and what gets read

This "selection mechanism" is what separates Mamba from classical
state space models (which use fixed B, C and can only learn fixed
patterns like convolutions).
""")

    # --- PART 1: THE DATA ---
    print(f"{'='*60}")
    print("  PART 1: THE DATA")
    print(f"{'='*60}")

    rng_data = create_rng(42)
    vocab_size = 8
    seq_len = 32
    n_copy = 4
    n_train = 300
    n_test = 50

    inputs_train, targets_train = selective_copying(
        rng_data, seq_len=seq_len, n_copy=n_copy,
        vocab_size=vocab_size, n_samples=n_train,
    )
    inputs_test, targets_test = selective_copying(
        rng_data, seq_len=seq_len, n_copy=n_copy,
        vocab_size=vocab_size, n_samples=n_test,
    )

    print(f"""
  Task: Selective Copying (from the Mamba paper, Figure 2)

  The model sees a sequence with data tokens scattered at random
  positions among blanks. After a COPY_MARKER, it must reproduce
  the data tokens in order. The catch: the positions are random,
  so the model can't memorize a fixed pattern — it must inspect
  each token to decide whether to remember it.

  Vocabulary: {{0: BLANK, 1: COPY_MARKER, 2..{vocab_size-1}: data tokens}}

  Example:
    Input:  {inputs_train[0]}
    Target: {targets_train[0]}

  The data tokens appear at random positions. After the marker (1),
  the target contains those same tokens in their original order.

  This is the Mamba paper's motivating task. A standard (non-selective)
  state space model with fixed B, C can solve regular copying (fixed
  spacing) by learning a convolution kernel that matches the offsets.
  But randomized spacing breaks this — the model MUST look at each
  token's content to decide whether to store it. That's exactly what
  Mamba's selection mechanism provides.

  Training samples: {n_train}
  Test samples:     {n_test}
  Sequence length:  {seq_len}
  Tokens to copy:   {n_copy}
""")

    # --- PART 2: THE ARCHITECTURE ---
    print(f"{'='*60}")
    print("  PART 2: THE ARCHITECTURE")
    print(f"{'='*60}")

    rng = create_rng(123)
    d_model = 24
    d_inner = 48
    d_state = 8
    d_conv = 4
    dt_rank = 6

    model = create_mamba_lm(
        rng, vocab_size=vocab_size, d_model=d_model, d_inner=d_inner,
        d_state=d_state, d_conv=d_conv, dt_rank=dt_rank, seq_len=seq_len,
    )
    total_params = _count_params(model)

    # Sub-component counts
    embed_p = vocab_size * d_model
    in_proj_p = 2 * d_inner * d_model
    conv_p = d_inner * d_conv + d_inner
    a_log_p = d_inner * d_state
    bc_proj_p = 2 * d_state * d_inner
    dt_p = dt_rank * d_inner + d_inner * dt_rank + d_inner
    d_skip_p = d_inner
    out_proj_p = d_model * d_inner + d_model
    head_p = vocab_size * d_model + vocab_size

    print(f"""
  Single Mamba block (Figure 3 of the paper):

    d_model={d_model}, d_inner={d_inner} (expansion E=2),
    d_state={d_state} (N), d_conv={d_conv}, dt_rank={dt_rank}

    +----- Mamba Block ----------------------------------+
    |                                                     |
    |  Input projection: embedded -> [x_branch, z_branch] |
    |    W_in: ({2*d_inner}, {d_model})                  {in_proj_p:>6} params|
    |                                                     |
    |  x_branch path:                                     |
    |    Conv1d (depthwise, causal, k={d_conv})           {conv_p:>6} params|
    |    SiLU                                             |
    |    +-- Selective SSM --------------------------+    |
    |    |  B = B_proj @ x        (d_state,)        |    |
    |    |  C = C_proj @ x        (d_state,)        |    |
    |    |  Delta = softplus(up(down(x)) + bias)    |    |
    |    |  A_bar = exp(Delta * A)   discretize     |    |
    |    |  B_bar = Delta * B        Euler approx   |    |
    |    |  h[t] = A_bar*h[t-1] + B_bar*x[t]       |    |
    |    |  y[t] = C.h[t] + D*x[t]                 |    |
    |    +------------------------------------------+    |
    |    SSM params (A,B,C,Delta,D):           {a_log_p + bc_proj_p + dt_p + d_skip_p:>6} params|
    |                                                     |
    |  z_branch: SiLU(z_branch) -> gate                   |
    |  gated = ssm_out * gate                             |
    |                                                     |
    |  Output projection: gated -> projected      {out_proj_p:>6} params|
    +-----------------------------------------------------+

    Embedding: ({vocab_size}, {d_model})                        {embed_p:>6} params
    LM head: ({vocab_size}, {d_model}) + bias                   {head_p:>6} params
    Total:                                        {total_params:>6} params

  Reading the diagram:
    A (via A_log) — state decay: how fast the model forgets older tokens
    B (via B_proj) — input selection: what features get written to state
    C (via C_proj) — output read: what features get read from state
    Delta (via dt_proj) — step size: how much of this token to remember
    D — skip connection: lets the raw input bypass the state entirely
    h[t] — hidden state: a fixed-size summary of all tokens so far

    "B_proj @ x" means "multiply the B projection by the current input" —
    this is how B becomes input-dependent (computed fresh at each position).
    "exp(Delta * A)" converts the continuous decay rate A into a per-step
    multiplier — larger Delta means faster decay of old state and more room
    for the new input.

    The Conv1d gives each channel a small (k={d_conv}) window of local
    context — "look at your immediate neighbors" — before the SSM processes
    long-range dependencies across the whole sequence.

  Compare with the transformer (Lesson 4):
    - Transformer: O(L^2) per layer (attention over all pairs)
    - Mamba: O(L) per layer (single pass through the sequence)

  The selective SSM is the key: at each position t, the model computes
  fresh B, C, and Delta from the input. Delta controls the "gate" —
  how much of the current input to write into the hidden state.
  When Delta is large, the discretized B_bar is large and the input
  gets written strongly. When Delta is small, the state mostly
  carries forward unchanged.
""")

    # --- PART 3: TRAINING ---
    print(f"{'='*60}")
    print("  PART 3: TRAINING")
    print(f"{'='*60}")

    learning_rate = 0.001
    epochs = 150

    print(f"""
  Training: AdamW, lr={learning_rate}, epochs={epochs}
  Weight decay: 0.01, gradient clipping: max_norm=1.0
  Loss: cross-entropy over all positions (including blanks)

  The model must learn to output BLANK at non-output positions and
  the correct data tokens after the marker. Cross-entropy over all
  positions means getting blanks right matters too — the model can't
  just focus on the copy slots.

  AdamW (Lesson 6) adapts per-parameter learning rates, which helps
  because Mamba has diverse parameter types: the A_log values control
  state decay timescales, Delta biases control gating sensitivity,
  and the projections are standard linear layers.

  Training (this takes a few minutes in pure Python)...
""")

    epoch_losses = train(
        model, inputs_train, targets_train,
        learning_rate=learning_rate, epochs=epochs,
        max_norm=1.0, weight_decay=0.01,
    )

    print(f"""
  Training results:""")
    for i, loss in enumerate(epoch_losses):
        if i == 0 or (i + 1) % 25 == 0 or i == len(epoch_losses) - 1:
            print(f"    Epoch {i+1:>3}: loss={loss:.4f}")

    # --- PART 4: EVALUATION ---
    print(f"""
{'='*60}
  PART 4: EVALUATION
{'='*60}""")

    marker_pos = seq_len - n_copy - 1
    correct_seqs = 0
    correct_tokens = 0
    total_copy_tokens = 0

    for i in range(len(inputs_test)):
        pred = predict(model, inputs_test[i])
        # Check the copy region
        copy_correct = True
        for j in range(marker_pos + 1, seq_len):
            total_copy_tokens += 1
            if pred[j] == targets_test[i][j]:
                correct_tokens += 1
            else:
                copy_correct = False
        if copy_correct:
            correct_seqs += 1

    seq_acc = correct_seqs / len(inputs_test) * 100
    tok_acc = correct_tokens / total_copy_tokens * 100

    print(f"""
  Test results:
    Sequence accuracy (all {n_copy} tokens correct): {correct_seqs}/{len(inputs_test)} ({seq_acc:.1f}%)
    Token accuracy (individual copy slots):     {correct_tokens}/{total_copy_tokens} ({tok_acc:.1f}%)

  Sample predictions (test set):""")
    for i in range(min(5, len(inputs_test))):
        pred = predict(model, inputs_test[i])
        data_in = [inputs_test[i][j] for j in range(marker_pos) if inputs_test[i][j] >= 2]
        pred_out = pred[marker_pos + 1:]
        tgt_out = targets_test[i][marker_pos + 1:]
        match = "OK" if pred_out == tgt_out else "MISS"
        print(f"    [{match}] data={data_in} -> target={tgt_out}, predicted={pred_out}")

    # --- PART 5: SELECTION IN ACTION ---
    print(f"""
{'='*60}
  PART 5: SELECTION IN ACTION
{'='*60}

  The selection mechanism works through Delta — the discretization
  step size. When Delta is large at a position, the model writes
  that input strongly into state. When Delta is small, the state
  carries forward mostly unchanged.

  If Mamba learns the task correctly, Delta should spike at data
  token positions and stay low at blanks. Let's check:""")

    # Get delta values for a few test samples
    delta_samples = []
    for i in range(min(5, len(inputs_test))):
        _, cache = mamba_forward(model, inputs_test[i])
        # Average Delta across d_inner channels
        avg_delta = []
        for t in range(seq_len):
            avg_delta.append(sum(cache.delta[t]) / d_inner)
        delta_samples.append(avg_delta)

    for i in range(min(3, len(delta_samples))):
        inp = inputs_test[i]
        deltas = delta_samples[i]
        print(f"\n  Sample {i}:")
        print(f"    Input:  {inp}")
        tokens_str = ""
        deltas_str = ""
        for t in range(seq_len):
            if inp[t] >= 2:
                tokens_str += f"  [{inp[t]}]"
                deltas_str += f" {deltas[t]:4.2f}"
            elif inp[t] == 1:
                tokens_str += "  [M]"
                deltas_str += f" {deltas[t]:4.2f}"
            else:
                tokens_str += "    ."
                deltas_str += f" {deltas[t]:4.2f}"
        print(f"    Tokens: {tokens_str}")
        print(f"    Delta:  {deltas_str}")

    # --- PLOTS ---
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Training loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, "b-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Mamba Training Loss (Selective Copying)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(output_dir, "05_mamba_training.png")
    fig.savefig(loss_path, dpi=100)
    plt.close(fig)
    print(f"\n  Saved training loss plot: {loss_path}")

    # Delta heatmap
    if delta_samples:
        n_show = min(10, len(inputs_test))
        delta_matrix = []
        input_labels = []
        for i in range(n_show):
            _, cache = mamba_forward(model, inputs_test[i])
            avg_delta = [sum(cache.delta[t]) / d_inner for t in range(seq_len)]
            delta_matrix.append(avg_delta)
            input_labels.append(f"Sample {i}")

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(delta_matrix, aspect="auto", cmap="hot")
        ax.set_xlabel("Position")
        ax.set_ylabel("Test Sample")
        ax.set_title("Delta (Discretization Step) Heatmap — Selection in Action")
        fig.colorbar(im, ax=ax, label="Avg Delta")

        # Mark data token positions on the first sample
        for j in range(seq_len):
            if inputs_test[0][j] >= 2:
                ax.plot(j, 0, "cv", markersize=6)
            elif inputs_test[0][j] == 1:
                ax.plot(j, 0, "g^", markersize=6)

        fig.tight_layout()
        delta_path = os.path.join(output_dir, "05_mamba_delta.png")
        fig.savefig(delta_path, dpi=100)
        plt.close(fig)
        print(f"  Saved delta heatmap: {delta_path}")

    # Predictions visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for i, ax in enumerate(axes):
        if i >= len(inputs_test):
            break
        inp = inputs_test[i]
        tgt = targets_test[i]
        pred = predict(model, inp)

        positions = list(range(seq_len))
        ax.bar([p - 0.2 for p in positions], tgt, width=0.4, alpha=0.6, label="Target", color="blue")
        ax.bar([p + 0.2 for p in positions], pred, width=0.4, alpha=0.6, label="Predicted", color="orange")
        # Mark data positions in input
        for j in range(seq_len):
            if inp[j] >= 2:
                ax.axvline(x=j, color="green", alpha=0.2, linewidth=2)
        ax.axvline(x=marker_pos, color="red", alpha=0.3, linewidth=2, linestyle="--", label="Marker")
        ax.set_ylabel(f"Sample {i}")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(-0.5, vocab_size)

    axes[-1].set_xlabel("Position")
    fig.suptitle("Selective Copying: Target vs Predicted", fontsize=14)
    fig.tight_layout()
    pred_path = os.path.join(output_dir, "05_mamba_predictions.png")
    fig.savefig(pred_path, dpi=100)
    plt.close(fig)
    print(f"  Saved predictions plot: {pred_path}")

    # --- CLOSING ---
    print(f"""
{'='*60}
  WHAT CHANGED
{'='*60}

The progression so far:

  Lesson 1 — Perceptron (1958): one neuron, one decision boundary
  Lesson 2 — MLP (1986): hidden layers + backprop solve XOR
  Lesson 3 — LeNet-5 (1998): convolutions learn spatial features
  Lesson 4 — Transformer (2017): attention lets every position
              see every other position — powerful but O(L^2)
  Lesson 5 — Mamba (2023): selective state spaces — O(L) with
              input-dependent gating
  Lesson 6 — CTM (2025): internal time + neural synchronization

What Mamba introduces:

  State space models: instead of attention (compare all pairs),
  maintain a running hidden state h[t] that summarizes the
  sequence so far. Process positions one at a time in O(L).

  Selection mechanism: classical SSMs use fixed B, C matrices —
  they're linear time-invariant (LTI) and can only learn patterns
  with fixed spacing (equivalent to convolutions). Mamba makes B,
  C, and Delta functions of the input, breaking LTI. The model
  can now decide PER TOKEN what to remember.

  Delta as a gate: the discretization step Delta controls how much
  of the current input gets written into state. High Delta = "this
  is important, remember it." Low Delta = "this is noise, skip it."
  The delta heatmap above shows this directly.

  Linear scaling: the recurrence h[t] = A*h[t-1] + B*x[t] is O(1)
  per position, so the full sequence costs O(L). For long sequences,
  this is a fundamental advantage over O(L^2) attention.

The selective copying task demonstrates exactly WHY selection matters:
random spacing means the model can't rely on fixed convolution
patterns. It must inspect each token, decide if it's data or blank,
and selectively write data tokens into its state for later recall.

Our {total_params:,}-parameter model shows the mechanism working in
miniature. The paper scales to billions of parameters and matches
or exceeds transformers on language modeling, DNA modeling, and audio.
""")

    print(f"{'='*60}")
    print("  END OF LESSON 5")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
