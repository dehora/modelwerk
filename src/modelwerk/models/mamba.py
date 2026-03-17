"""Mamba: Selective State Space Model (Gu & Dao, 2023).

A sequence model that replaces transformer attention with input-dependent
state space updates. Instead of attending to all positions (O(L²)), Mamba
maintains a compressed running state that gets selectively updated at each
position (O(L)).

Architecture (single Mamba block):
    Input projection: embedded → [x_branch, z_branch]
    x_branch: Conv1d (depthwise, causal) → SiLU → Selective SSM
    z_branch: SiLU gate
    Output: (ssm_out ⊙ gate) → output projection → LM head

Key innovation: B, C, and Δ are functions of the input, so the model
decides at each timestep what to remember and what to ignore. This is
what makes it "selective" — an LTI (linear time-invariant) SSM uses
fixed B, C and cannot solve tasks where spacing varies.
"""

import math
from dataclasses import dataclass

from modelwerk.primitives import scalar, vector, matrix
from modelwerk.primitives.activations import (
    silu, silu_derivative, sigmoid, softmax, softplus, softplus_derivative,
)
from modelwerk.primitives.losses import cross_entropy
from modelwerk.primitives.progress import progress_bar, progress_done
from modelwerk.primitives.random import xavier_init, random_vector, uniform
from modelwerk.data.utils import one_hot

Vector = list[float]
Matrix = list[list[float]]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MambaSSM:
    """Selective state space model parameters.

    A state space model maintains a hidden state h that summarizes the sequence
    seen so far, updated at each position by the recurrence:

        h[t+1] = A_bar * h[t] + B_bar * x[t]    (state update)
        y[t]   = C[t] · h[t+1] + D * x[t]       (output read + skip)

    Classical SSMs use fixed A, B, C — they treat every token the same way,
    which limits them to fixed-spacing patterns (equivalent to convolutions).

    Mamba's key innovation: B, C, and the step size Delta are computed FROM
    the input at each position. This lets the model decide per-token what to
    write into state, what to read out, and how strongly to gate the update.
    This is what "selective" means — the model selects what to remember.
    """
    A_log: Matrix        # (d_inner, d_state) — controls state decay: how fast
                         # the model forgets older tokens. Stored as log(n+1) so
                         # A = -exp(A_log) stays negative (stable, decaying).
    B_proj: Matrix       # (d_state, d_inner) — input selection ("write gate"):
                         # projects the current input to B, controlling what
                         # features get written into hidden state.
    C_proj: Matrix       # (d_state, d_inner) — output read ("read head"):
                         # projects the current input to C, controlling what
                         # features get read back from hidden state.
    dt_proj_down: Matrix # (dt_rank, d_inner) — first half of the Delta bottleneck.
                         # Delta is the discretization step size — the "volume knob"
                         # controlling how much of the current input to write into
                         # state. Large Delta = "remember this token." Small Delta =
                         # "skip it." The bottleneck (d_inner -> dt_rank -> d_inner)
                         # reduces parameter count.
    dt_proj_up: Matrix   # (d_inner, dt_rank) — second half of the Delta bottleneck.
    dt_bias: Vector      # (d_inner,) — bias added before softplus, sets the
                         # baseline gating sensitivity for each channel.
    D: Vector            # (d_inner,) — skip connection: lets the current input
                         # contribute directly to the output, bypassing the state.


@dataclass
class MambaBlock:
    """Single Mamba block: two branches, local convolution, SSM, and gating.

    The input is projected into two parallel branches of width d_inner:

      x_branch: Conv1d → SiLU → Selective SSM  (content processing)
      z_branch: SiLU                            (learned gate)

    The output is their element-wise product: SSM_output * SiLU(z_branch).
    Gating lets the network suppress or amplify individual channels — the
    z_branch sees the same input but skips the SSM, so it learns a
    complementary signal about which channels are useful for each token.

    The depthwise Conv1d gives each channel a small window (k=4) of local
    context — "look at your immediate neighbors" — before the SSM processes
    long-range dependencies across the whole sequence.
    """
    in_proj: Matrix      # (2*d_inner, d_model) — projects input into both
                         # branches at once. First d_inner rows = x_branch,
                         # remaining d_inner rows = z_branch.
    conv_weight: Matrix  # (d_inner, d_conv) — depthwise conv kernels: each
                         # channel has its own small kernel for local context.
    conv_bias: Vector    # (d_inner,)
    ssm: MambaSSM
    out_proj: Matrix     # (d_model, d_inner) — projects gated output back
                         # to d_model for the LM head.
    out_proj_bias: Vector


@dataclass
class MambaLM:
    """Complete Mamba language model.

    Pipeline: token embedding → single Mamba block → LM head (logits → softmax).
    Production Mamba stacks many blocks with layer norms between them; we use
    one block to keep the implementation transparent.
    """
    embedding: Matrix    # (vocab_size, d_model)
    block: MambaBlock
    head: Matrix         # (vocab_size, d_model)
    head_bias: Vector
    vocab_size: int
    d_model: int
    d_inner: int
    d_state: int
    d_conv: int
    dt_rank: int
    seq_len: int


@dataclass
class MambaCache:
    """All intermediates needed for backprop."""
    token_ids: list[int]
    embedded: list[Vector]      # (L, d_model)
    xz: list[Vector]            # (L, 2*d_inner)
    x_branch: list[Vector]      # (L, d_inner)
    z_branch: list[Vector]      # (L, d_inner)
    z_gate: list[Vector]        # (L, d_inner) — SiLU(z_branch)
    conv_input: list[Vector]    # (L, d_inner) — pre-conv (with padding context)
    conv_out: list[Vector]      # (L, d_inner)
    conv_silu_out: list[Vector] # (L, d_inner)
    B_t: list[Vector]           # (L, d_state)
    C_t: list[Vector]           # (L, d_state)
    dt_down: list[Vector]       # (L, dt_rank) — dt_proj_down @ x
    dt_pre_softplus: list[Vector]  # (L, d_inner)
    delta: list[Vector]         # (L, d_inner)
    A_bar: list[Matrix]         # (L, d_inner, d_state)
    B_bar: list[Matrix]         # (L, d_inner, d_state)
    h_states: list[Matrix]      # (L+1, d_inner, d_state) — h[0] is zeros
    ssm_out: list[Vector]       # (L, d_inner)
    gated_out: list[Vector]     # (L, d_inner)
    projected: list[Vector]     # (L, d_model)
    logits: list[Vector]        # (L, vocab_size)
    probs: list[Vector]         # (L, vocab_size)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def create_mamba_lm(
    rng,
    vocab_size: int = 8,
    d_model: int = 24,
    d_inner: int = 48,
    d_state: int = 8,
    d_conv: int = 4,
    dt_rank: int = 6,
    seq_len: int = 32,
) -> MambaLM:
    """Create a Mamba language model with S4D-Real initialization."""
    # Embedding
    embedding = xavier_init(rng, d_model, vocab_size)  # (vocab_size, d_model)

    # Input projection: d_model → 2*d_inner
    in_proj = xavier_init(rng, d_model, 2 * d_inner)

    # Depthwise conv1d: each of d_inner channels has its own kernel of size d_conv
    conv_limit = math.sqrt(6.0 / (1 + d_conv))
    conv_weight = [[rng.uniform(-conv_limit, conv_limit) for _ in range(d_conv)]
                   for _ in range(d_inner)]
    conv_bias = [0.0] * d_inner

    # SSM parameters
    # A_log: S4D-Real initialization — A[d][n] = -(n+1), stored as log(n+1)
    A_log = [[math.log(n + 1) for n in range(d_state)] for _ in range(d_inner)]

    # B and C projections
    B_proj = xavier_init(rng, d_inner, d_state)   # (d_state, d_inner)
    C_proj = xavier_init(rng, d_inner, d_state)   # (d_state, d_inner)

    # Delta projection: bottleneck d_inner → dt_rank → d_inner
    dt_proj_down = xavier_init(rng, d_inner, dt_rank)   # (dt_rank, d_inner)
    dt_proj_up = xavier_init(rng, dt_rank, d_inner)     # (d_inner, dt_rank)

    # dt_bias: inverse_softplus(uniform(0.001, 0.1))
    dt_bias = []
    for _ in range(d_inner):
        val = uniform(rng, 0.001, 0.1)
        # inverse softplus: log(exp(x) - 1)
        dt_bias.append(math.log(math.exp(val) - 1.0))

    # D: skip connection, initialized to ones
    D = [1.0] * d_inner

    ssm = MambaSSM(A_log, B_proj, C_proj, dt_proj_down, dt_proj_up, dt_bias, D)

    # Output projection: d_inner → d_model
    out_proj = xavier_init(rng, d_inner, d_model)  # (d_model, d_inner)
    out_proj_bias = [0.0] * d_model

    block = MambaBlock(in_proj, conv_weight, conv_bias, ssm, out_proj, out_proj_bias)

    # LM head: d_model → vocab_size
    head = xavier_init(rng, d_model, vocab_size)  # (vocab_size, d_model)
    head_bias = [0.0] * vocab_size

    return MambaLM(
        embedding, block, head, head_bias,
        vocab_size, d_model, d_inner, d_state, d_conv, dt_rank, seq_len,
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def mamba_forward(model: MambaLM, token_ids: list[int]) -> tuple[list[Vector], MambaCache]:
    """Forward pass through the Mamba LM.

    Returns (probs, cache) where probs[t] is the probability distribution
    over vocab at position t.
    """
    L = len(token_ids)
    d_inner = model.d_inner
    d_state = model.d_state
    d_conv = model.d_conv

    # 1. Embed tokens
    embedded = [model.embedding[tid][:] for tid in token_ids]

    # 2. Input projection: embedded → xz, split into x_branch and z_branch
    xz_list = []
    x_branch_list = []
    z_branch_list = []
    for t in range(L):
        xz = matrix.mat_vec(model.block.in_proj, embedded[t])
        xz_list.append(xz)
        x_branch_list.append(xz[:d_inner])
        z_branch_list.append(xz[d_inner:])

    # 3. Causal depthwise conv1d on x_branch
    # Left-pad with zeros for causality
    conv_input_list = []
    conv_out_list = []
    for t in range(L):
        out = [0.0] * d_inner
        for d in range(d_inner):
            val = model.block.conv_bias[d]
            for k in range(d_conv):
                src_t = t - (d_conv - 1 - k)
                if src_t >= 0:
                    val += model.block.conv_weight[d][k] * x_branch_list[src_t][d]
            out[d] = val
        conv_input_list.append(x_branch_list[t][:])
        conv_out_list.append(out)

    # 4. SiLU on conv output
    conv_silu_list = []
    for t in range(L):
        conv_silu_list.append([silu(conv_out_list[t][d]) for d in range(d_inner)])

    # 5. Input-dependent SSM parameters — Mamba's key innovation.
    # Classical SSMs use fixed B, C — every token is processed the same way.
    # Mamba computes B, C, and Delta FROM the current input x[t]:
    #   B_t = B_proj @ x[t]  -> what to write depends on what we're seeing
    #   C_t = C_proj @ x[t]  -> what to read depends on what we're seeing
    #   Delta_t = softplus(up(down(x[t])) + bias) -> how much to write
    # This breaks time-invariance: the model can treat data tokens differently
    # from blanks, even though it has no explicit "if data_token" logic.
    B_t_list = []
    C_t_list = []
    dt_down_list = []
    dt_pre_list = []
    delta_list = []
    for t in range(L):
        x = conv_silu_list[t]
        B_t = matrix.mat_vec(model.block.ssm.B_proj, x)
        C_t = matrix.mat_vec(model.block.ssm.C_proj, x)
        dt_down = matrix.mat_vec(model.block.ssm.dt_proj_down, x)
        dt_up = matrix.mat_vec(model.block.ssm.dt_proj_up, dt_down)
        dt_pre = [dt_up[d] + model.block.ssm.dt_bias[d] for d in range(d_inner)]
        delta = [softplus(dt_pre[d]) for d in range(d_inner)]
        B_t_list.append(B_t)
        C_t_list.append(C_t)
        dt_down_list.append(dt_down)
        dt_pre_list.append(dt_pre)
        delta_list.append(delta)

    # 6. Discretize and 7. Selective scan
    #
    # Discretization converts the continuous-time SSM (dh/dt = A*h + B*x) into
    # discrete step-by-step updates we can run on a sequence of tokens:
    #   A_bar = exp(Delta * A)  — state decay per step. When Delta is small,
    #     A_bar ≈ 1 and the state barely changes. When Delta is large, A_bar
    #     shrinks toward 0 and old state is forgotten, making room for new input.
    #   B_bar = Delta * B  — input gate, scaled by step size. When Delta is
    #     large, B_bar is large and the input writes strongly into state.
    #
    # The scan then runs the recurrence forward, building up h — a fixed-size
    # compressed summary of all tokens seen so far. Unlike a transformer's KV
    # cache (which grows with sequence length), h stays the same size regardless
    # of how long the sequence is: d_inner * d_state values.
    A_bar_list = []
    B_bar_list = []
    h_states = [matrix.zeros(d_inner, d_state)]  # h[0] = zeros
    ssm_out_list = []

    for t in range(L):
        x = conv_silu_list[t]
        delta_t = delta_list[t]
        B_t = B_t_list[t]

        # Discretize: A = -exp(A_log) is the continuous decay rate.
        # A_bar = exp(Delta * A) converts it to a per-step multiplier.
        # B_bar = Delta * B scales the input gate by the step size.
        A_bar_t = [[0.0] * d_state for _ in range(d_inner)]
        B_bar_t = [[0.0] * d_state for _ in range(d_inner)]
        for d in range(d_inner):
            for n in range(d_state):
                a_val = -math.exp(model.block.ssm.A_log[d][n])
                A_bar_t[d][n] = math.exp(delta_t[d] * a_val)
                B_bar_t[d][n] = delta_t[d] * B_t[n]

        A_bar_list.append(A_bar_t)
        B_bar_list.append(B_bar_t)

        # Selective scan recurrence: build up h, the running state summary.
        # h[t+1] = A_bar * h[t]  +  B_bar * x[t]
        #          ^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^
        #          old state,        new input,
        #          decayed           gated in
        # Each (d, n) pair is an independent memory channel.
        h_prev = h_states[t]
        h_new = [[0.0] * d_state for _ in range(d_inner)]
        for d in range(d_inner):
            for n in range(d_state):
                h_new[d][n] = A_bar_t[d][n] * h_prev[d][n] + B_bar_t[d][n] * x[d]
        h_states.append(h_new)

        # Read from state: C selects what to extract, D provides a direct shortcut.
        # C was computed from x[t] (step 5), so what gets read is also selective.
        C_t = C_t_list[t]
        y = [0.0] * d_inner
        for d in range(d_inner):
            c_dot_h = 0.0
            for n in range(d_state):
                c_dot_h += C_t[n] * h_new[d][n]
            y[d] = c_dot_h + model.block.ssm.D[d] * x[d]
        ssm_out_list.append(y)

    # 8. Gate: the z_branch learns to suppress or amplify channels.
    # z_branch saw the same input as x_branch but took a different path (no
    # conv, no SSM). It learns which channels of the SSM output are useful
    # for the current input. SiLU allows both positive and negative gating.
    z_gate_list = []
    gated_list = []
    for t in range(L):
        z_gate = [silu(z_branch_list[t][d]) for d in range(d_inner)]
        gated = [ssm_out_list[t][d] * z_gate[d] for d in range(d_inner)]
        z_gate_list.append(z_gate)
        gated_list.append(gated)

    # 9. Output projection
    projected_list = []
    for t in range(L):
        proj = matrix.mat_vec(model.block.out_proj, gated_list[t])
        proj = [proj[d] + model.block.out_proj_bias[d] for d in range(model.d_model)]
        projected_list.append(proj)

    # 10. LM head → softmax
    logits_list = []
    probs_list = []
    for t in range(L):
        logit = matrix.mat_vec(model.head, projected_list[t])
        logit = [logit[v] + model.head_bias[v] for v in range(model.vocab_size)]
        logits_list.append(logit)
        probs_list.append(softmax(logit))

    cache = MambaCache(
        token_ids=token_ids,
        embedded=embedded,
        xz=xz_list,
        x_branch=x_branch_list,
        z_branch=z_branch_list,
        z_gate=z_gate_list,
        conv_input=conv_input_list,
        conv_out=conv_out_list,
        conv_silu_out=conv_silu_list,
        B_t=B_t_list,
        C_t=C_t_list,
        dt_down=dt_down_list,
        dt_pre_softplus=dt_pre_list,
        delta=delta_list,
        A_bar=A_bar_list,
        B_bar=B_bar_list,
        h_states=h_states,
        ssm_out=ssm_out_list,
        gated_out=gated_list,
        projected=projected_list,
        logits=logits_list,
        probs=probs_list,
    )
    return probs_list, cache


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def mamba_backward(model: MambaLM, cache: MambaCache, targets: list[Vector]) -> dict:
    """Backward pass through the Mamba LM.

    targets: list of one-hot vectors, one per position.
    Returns gradient dict keyed by parameter name.

    The backward pass reverses the forward computation in three stages:

    Stage 1 (per-timestep, forward order): loss → LM head → output projection →
      gating → SSM output equation. Computes d_ssm_out and d_z_branch per
      timestep, and accumulates gradients for C_proj, D, and the contribution
      of x through the output equation.

    Stage 2 (SSM recurrence, REVERSE time order — the hard part): walks backward
      through time propagating d_h, the gradient of loss w.r.t. hidden state.
      At each timestep: (a) add the output gradient's contribution to d_h,
      (b) compute discretization gradients (d_A_log, d_delta, d_B),
      (c) propagate d_h backward: d_h[t] = d_h[t+1] * A_bar[t].
      This mirrors how the forward scan propagates h forward.

    Stage 3 (per-timestep, forward order): Delta projection → softplus →
      conv1d → input projection → embedding. Standard chain rule through
      linear layers.
    """
    L = len(cache.token_ids)
    d_model = model.d_model
    d_inner = model.d_inner
    d_state = model.d_state
    d_conv = model.d_conv
    dt_rank = model.dt_rank
    V = model.vocab_size

    # Gradient accumulators for all parameters
    d_embedding = matrix.zeros(V, d_model)
    d_head = matrix.zeros(V, d_model)
    d_head_bias = vector.zeros(V)
    d_out_proj = matrix.zeros(d_model, d_inner)
    d_out_proj_bias = vector.zeros(d_model)
    d_in_proj = matrix.zeros(2 * d_inner, d_model)
    d_conv_weight = matrix.zeros(d_inner, d_conv)
    d_conv_bias = vector.zeros(d_inner)
    d_A_log = matrix.zeros(d_inner, d_state)
    d_B_proj = matrix.zeros(d_state, d_inner)
    d_C_proj = matrix.zeros(d_state, d_inner)
    d_dt_proj_down = matrix.zeros(dt_rank, d_inner)
    d_dt_proj_up = matrix.zeros(d_inner, dt_rank)
    d_dt_bias = vector.zeros(d_inner)
    d_D = vector.zeros(d_inner)

    # Step 1: Compute per-timestep gradients from loss down to SSM output.
    # Store d_ssm_out and d_z_branch per timestep for later use.
    d_ssm_out_list = []
    d_z_branch_list = []
    d_conv_silu = [vector.zeros(d_inner) for _ in range(L)]
    loss_scale = 1.0 / L  # loss is averaged over sequence positions

    for t in range(L):
        # 10. Loss → probs: softmax + cross-entropy gradient = (probs - target) / L
        d_logit = [(cache.probs[t][v] - targets[t][v]) * loss_scale for v in range(V)]

        # d_head_bias
        for v in range(V):
            d_head_bias[v] += d_logit[v]

        # d_head: outer(d_logit, projected)
        for v in range(V):
            for d in range(d_model):
                d_head[v][d] += d_logit[v] * cache.projected[t][d]

        # d_projected = head^T @ d_logit
        d_proj = [0.0] * d_model
        for d in range(d_model):
            for v in range(V):
                d_proj[d] += model.head[v][d] * d_logit[v]

        # 9. Output projection backward
        for d in range(d_model):
            d_out_proj_bias[d] += d_proj[d]

        # d_out_proj: outer(d_proj, gated)
        for r in range(d_model):
            for c in range(d_inner):
                d_out_proj[r][c] += d_proj[r] * cache.gated_out[t][c]

        # d_gated = out_proj^T @ d_proj
        d_gated = [0.0] * d_inner
        for d in range(d_inner):
            for r in range(d_model):
                d_gated[d] += model.block.out_proj[r][d] * d_proj[r]

        # 8. Gate backward: gated = ssm_out * z_gate
        d_ssm_out = [d_gated[d] * cache.z_gate[t][d] for d in range(d_inner)]
        d_z_gate = [d_gated[d] * cache.ssm_out[t][d] for d in range(d_inner)]

        # d_z_branch through SiLU
        d_z_branch = [d_z_gate[d] * silu_derivative(cache.z_branch[t][d]) for d in range(d_inner)]

        d_ssm_out_list.append(d_ssm_out)
        d_z_branch_list.append(d_z_branch)

        # SSM output: y[d] = C·h[d] + D[d]*x[d]
        # Accumulate D and C-projection gradients (not d_h — that's done in the backward loop)
        h_tp1 = cache.h_states[t + 1]
        C_t = cache.C_t[t]
        x_t = cache.conv_silu_out[t]

        d_C_t = [0.0] * d_state
        for d in range(d_inner):
            # d_D
            d_D[d] += d_ssm_out[d] * x_t[d]
            # d_x from D*x term
            d_conv_silu[t][d] += d_ssm_out[d] * model.block.ssm.D[d]
            # d_C from C·h term
            for n in range(d_state):
                d_C_t[n] += d_ssm_out[d] * h_tp1[d][n]

        # d_C_proj: outer(d_C_t, x_t)
        for n in range(d_state):
            for d in range(d_inner):
                d_C_proj[n][d] += d_C_t[n] * x_t[d]
        # d_x from C projection
        for d in range(d_inner):
            for n in range(d_state):
                d_conv_silu[t][d] += model.block.ssm.C_proj[n][d] * d_C_t[n]

    # Step 2: Backward through SSM recurrence (reverse time).
    #
    # This mirrors the forward scan but runs backward. d_h tracks how the loss
    # changes when the hidden state changes. At each timestep we:
    #   (a) Add the output equation's contribution: since y = C·h + D*x,
    #       changing h affects y, so d_h += d_y * C.
    #   (b) Use d_h to compute gradients for A_bar, B_bar, and delta
    #       (the discretization parameters that shaped this timestep's update).
    #   (c) Propagate d_h backward: since h[t+1] = A_bar*h[t] + B_bar*x[t],
    #       the gradient flows through A_bar: d_h[t] = d_h[t+1] * A_bar[t].
    #       Because A_bar < 1, gradients decay at the same rate as the forward
    #       signal — this naturally prevents gradient explosion over long sequences.
    d_h = matrix.zeros(d_inner, d_state)
    d_delta_list = [vector.zeros(d_inner) for _ in range(L)]

    for t in range(L - 1, -1, -1):
        x_t = cache.conv_silu_out[t]
        h_prev = cache.h_states[t]
        A_bar_t = cache.A_bar[t]
        B_bar_t = cache.B_bar[t]
        C_t = cache.C_t[t]
        d_ssm_out = d_ssm_out_list[t]

        # (a) Output gradient contribution to d_h:
        # y[t] = C[t] · h[t+1] + D*x[t], so d_h[t+1] += d_y[t] * C[t]
        for d in range(d_inner):
            for n in range(d_state):
                d_h[d][n] += d_ssm_out[d] * C_t[n]

        # (b) Discretization gradients — reverse of the exp() and multiplication
        # in forward step 6. We need d_A_bar, d_B_bar to compute d_delta and d_A_log.
        d_B_t = [0.0] * d_state
        for d in range(d_inner):
            for n in range(d_state):
                # From h[t+1] = A_bar*h[t] + B_bar*x[t]:
                d_A_bar_dn = d_h[d][n] * h_prev[d][n]
                d_B_bar_dn = d_h[d][n] * x_t[d]
                d_conv_silu[t][d] += d_h[d][n] * B_bar_t[d][n]

                # Chain through discretization:
                # A_bar = exp(Delta * A) → d_Delta += d_A_bar * A_bar * A
                a_val = -math.exp(model.block.ssm.A_log[d][n])
                d_delta_list[t][d] += d_A_bar_dn * A_bar_t[d][n] * a_val
                # d_A_log: chain through A = -exp(A_log)
                d_A_log[d][n] += d_A_bar_dn * A_bar_t[d][n] * cache.delta[t][d] * (-math.exp(model.block.ssm.A_log[d][n]))

                # B_bar = Delta * B → d_Delta += d_B_bar * B, d_B += d_B_bar * Delta
                d_delta_list[t][d] += d_B_bar_dn * cache.B_t[t][n]
                d_B_t[n] += d_B_bar_dn * cache.delta[t][d]

            # (c) Propagate d_h backward through the recurrence:
            # d_h[t] = d_h[t+1] * A_bar[t] — gradient decays like the forward signal.
            for n in range(d_state):
                d_h[d][n] = d_h[d][n] * A_bar_t[d][n]

        # d_B_proj: outer(d_B_t, x_t)
        for n in range(d_state):
            for d in range(d_inner):
                d_B_proj[n][d] += d_B_t[n] * x_t[d]
        # d_x from B projection
        for d in range(d_inner):
            for n in range(d_state):
                d_conv_silu[t][d] += model.block.ssm.B_proj[n][d] * d_B_t[n]

    # Step 3: Delta backward — Δ = softplus(dt_proj_up @ dt_proj_down @ x + dt_bias)
    for t in range(L):
        d_pre = [d_delta_list[t][d] * softplus_derivative(cache.dt_pre_softplus[t][d])
                 for d in range(d_inner)]

        for d in range(d_inner):
            d_dt_bias[d] += d_pre[d]

        # d_dt_proj_up: outer(d_pre, dt_down)
        for d in range(d_inner):
            for r in range(dt_rank):
                d_dt_proj_up[d][r] += d_pre[d] * cache.dt_down[t][r]

        # d_dt_down = dt_proj_up^T @ d_pre
        d_dt_down = [0.0] * dt_rank
        for r in range(dt_rank):
            for d in range(d_inner):
                d_dt_down[r] += model.block.ssm.dt_proj_up[d][r] * d_pre[d]

        # d_dt_proj_down: outer(d_dt_down, x_t)
        x_t = cache.conv_silu_out[t]
        for r in range(dt_rank):
            for d in range(d_inner):
                d_dt_proj_down[r][d] += d_dt_down[r] * x_t[d]

        # d_x from dt projection
        for d in range(d_inner):
            for r in range(dt_rank):
                d_conv_silu[t][d] += model.block.ssm.dt_proj_down[r][d] * d_dt_down[r]

    # SiLU backward on conv output
    d_conv_out = [vector.zeros(d_inner) for _ in range(L)]
    for t in range(L):
        for d in range(d_inner):
            d_conv_out[t][d] = d_conv_silu[t][d] * silu_derivative(cache.conv_out[t][d])

    # Conv1d backward (depthwise, causal)
    d_x_branch = [vector.zeros(d_inner) for _ in range(L)]
    for t in range(L):
        for d in range(d_inner):
            # d_conv_bias
            d_conv_bias[d] += d_conv_out[t][d]
            for k in range(d_conv):
                src_t = t - (d_conv - 1 - k)
                if src_t >= 0:
                    # d_conv_weight
                    d_conv_weight[d][k] += d_conv_out[t][d] * cache.x_branch[src_t][d]
                    # d_x_branch
                    d_x_branch[src_t][d] += d_conv_out[t][d] * model.block.conv_weight[d][k]

    # Step 5: Input projection backward
    # xz = in_proj @ embedded, split into [x_branch, z_branch]
    for t in range(L):
        d_xz = d_x_branch[t] + d_z_branch_list[t]  # concatenation

        # d_in_proj: outer(d_xz, embedded[t])
        for r in range(2 * d_inner):
            for c in range(d_model):
                d_in_proj[r][c] += d_xz[r] * cache.embedded[t][c]

        # d_embedded = in_proj^T @ d_xz
        d_emb = [0.0] * d_model
        for c in range(d_model):
            for r in range(2 * d_inner):
                d_emb[c] += model.block.in_proj[r][c] * d_xz[r]

        # d_embedding: accumulate for each token
        tid = cache.token_ids[t]
        for d in range(d_model):
            d_embedding[tid][d] += d_emb[d]

    return {
        "d_embedding": d_embedding,
        "d_head": d_head,
        "d_head_bias": d_head_bias,
        "d_out_proj": d_out_proj,
        "d_out_proj_bias": d_out_proj_bias,
        "d_in_proj": d_in_proj,
        "d_conv_weight": d_conv_weight,
        "d_conv_bias": d_conv_bias,
        "d_A_log": d_A_log,
        "d_B_proj": d_B_proj,
        "d_C_proj": d_C_proj,
        "d_dt_proj_down": d_dt_proj_down,
        "d_dt_proj_up": d_dt_proj_up,
        "d_dt_bias": d_dt_bias,
        "d_D": d_D,
    }


# ---------------------------------------------------------------------------
# AdamW optimizer (self-contained, following CTM pattern)
# ---------------------------------------------------------------------------

def _adamw_update_scalar(
    param: float, grad: float, m: float, v: float,
    lr: float, beta1: float, beta2: float, eps: float, wd: float,
    bc1: float, bc2: float,
) -> tuple[float, float, float]:
    """AdamW update for a single scalar parameter."""
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * grad * grad
    m_hat = m / bc1
    v_hat = v / bc2
    param = param * (1.0 - lr * wd) - lr * m_hat / (scalar.power(v_hat, 0.5) + eps)
    return param, m, v


def _create_adamw_state(grads: dict) -> tuple[dict, dict]:
    """Create zeroed moment dicts matching grad shapes."""
    m: dict = {}
    v: dict = {}
    for key, val in grads.items():
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], list):
                m[key] = [[0.0] * len(row) for row in val]
                v[key] = [[0.0] * len(row) for row in val]
            else:
                m[key] = [0.0] * len(val)
                v[key] = [0.0] * len(val)
    return m, v


def _clip_grad_norm(grads: dict, max_norm: float) -> float:
    """Clip gradients by global L2 norm."""
    sum_sq = 0.0
    for val in grads.values():
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], list):
                for row in val:
                    for x in row:
                        sum_sq += x * x
            else:
                for x in val:
                    sum_sq += x * x

    global_norm = math.sqrt(sum_sq)
    if global_norm > max_norm:
        scale = max_norm / global_norm
        for val in grads.values():
            if isinstance(val, list) and len(val) > 0:
                if isinstance(val[0], list):
                    for row in val:
                        for j in range(len(row)):
                            row[j] *= scale
                else:
                    for j in range(len(val)):
                        val[j] *= scale
    return global_norm


def _mamba_adamw_update(
    model: MambaLM, grads: dict, m_state: dict, v_state: dict,
    lr: float, step: int, beta1: float = 0.9, beta2: float = 0.999,
    eps: float = 1e-8, weight_decay: float = 0.01,
) -> None:
    """Update all Mamba parameters using AdamW."""
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    def update_matrix(M, key):
        dM = grads[key]; mM = m_state[key]; vM = v_state[key]
        for r in range(len(M)):
            for c in range(len(M[0])):
                M[r][c], mM[r][c], vM[r][c] = _adamw_update_scalar(
                    M[r][c], dM[r][c], mM[r][c], vM[r][c],
                    lr, beta1, beta2, eps, weight_decay, bc1, bc2)

    def update_vector(vec, key):
        dv = grads[key]; mv = m_state[key]; vv = v_state[key]
        for i in range(len(vec)):
            vec[i], mv[i], vv[i] = _adamw_update_scalar(
                vec[i], dv[i], mv[i], vv[i],
                lr, beta1, beta2, eps, weight_decay, bc1, bc2)

    update_matrix(model.embedding, "d_embedding")
    update_matrix(model.head, "d_head")
    update_vector(model.head_bias, "d_head_bias")
    update_matrix(model.block.out_proj, "d_out_proj")
    update_vector(model.block.out_proj_bias, "d_out_proj_bias")
    update_matrix(model.block.in_proj, "d_in_proj")
    update_matrix(model.block.conv_weight, "d_conv_weight")
    update_vector(model.block.conv_bias, "d_conv_bias")
    update_matrix(model.block.ssm.A_log, "d_A_log")
    update_matrix(model.block.ssm.B_proj, "d_B_proj")
    update_matrix(model.block.ssm.C_proj, "d_C_proj")
    update_matrix(model.block.ssm.dt_proj_down, "d_dt_proj_down")
    update_matrix(model.block.ssm.dt_proj_up, "d_dt_proj_up")
    update_vector(model.block.ssm.dt_bias, "d_dt_bias")
    update_vector(model.block.ssm.D, "d_D")


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------

def predict(model: MambaLM, token_ids: list[int]) -> list[int]:
    """Predict output tokens for an input sequence.

    Returns list of predicted token ids (argmax at each position).
    """
    probs, _ = mamba_forward(model, token_ids)
    return [max(range(model.vocab_size), key=lambda v: probs[t][v]) for t in range(len(token_ids))]


def train(
    model: MambaLM,
    inputs: list[list[int]],
    targets: list[list[int]],
    learning_rate: float = 0.001,
    epochs: int = 150,
    max_norm: float = 1.0,
    weight_decay: float = 0.01,
) -> list[float]:
    """Train the Mamba LM on selective copying data.

    Returns list of average losses per epoch.
    """
    epoch_losses: list[float] = []
    m_state: dict = {}
    v_state: dict = {}
    adamw_initialized = False
    step = 0

    for epoch in range(epochs):
        total_loss = 0.0
        n_samples = len(inputs)

        for sample_idx in range(n_samples):
            inp = inputs[sample_idx]
            tgt = targets[sample_idx]
            target_onehots = [one_hot(t, model.vocab_size) for t in tgt]

            # Forward
            probs, cache = mamba_forward(model, inp)
            loss = sum(cross_entropy(probs[t], target_onehots[t])
                       for t in range(len(inp))) / len(inp)
            total_loss += loss

            # Backward
            grads = mamba_backward(model, cache, target_onehots)
            _clip_grad_norm(grads, max_norm)

            # Update
            step += 1
            if not adamw_initialized:
                m_state, v_state = _create_adamw_state(grads)
                adamw_initialized = True
            _mamba_adamw_update(
                model, grads, m_state, v_state,
                learning_rate, step, weight_decay=weight_decay,
            )

        avg_loss = total_loss / n_samples
        epoch_losses.append(avg_loss)
        progress_bar(epoch + 1, epochs, avg_loss)

    progress_done()
    return epoch_losses
