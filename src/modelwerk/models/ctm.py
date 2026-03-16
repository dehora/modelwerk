"""Continuous Thought Machine (Darlow et al., 2025).

A neural network with an internal time dimension — "thought steps" — where
each neuron has its own temporal dynamics via private weights (neuron-level
models), and neural synchronization (temporal correlations between neurons)
serves as the latent representation.

Architecture (per internal tick t):
    Synapse model: MLP producing pre-activations from z^t and o^t
    NLMs: per-neuron MLPs processing rolling history of pre-activations
    Synchronization: recursive dot product of post-activation histories
    Cross-attention: queries from synchronization, keys/values from input data
    Output: projection from synchronization to class predictions

Key innovations vs prior lessons:
    - Internal tick loop: the network "thinks" for T steps, refining its answer
    - Neuron-level models: each neuron has private weights over its history
    - Neural synchronization: temporal correlations become the representation
    - Certainty-based loss: naturally gives adaptive compute
"""

import math
from dataclasses import dataclass, field

from modelwerk.primitives import scalar, vector, matrix
from modelwerk.primitives.activations import (
    silu, silu_derivative, sigmoid, softmax, layer_norm, layer_norm_backward,
)
from modelwerk.primitives.losses import cross_entropy
from modelwerk.primitives.progress import progress_bar, progress_done
from modelwerk.primitives.random import xavier_init, random_matrix, random_vector
from modelwerk.data.utils import one_hot

Vector = list[float]
Matrix = list[list[float]]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SynapseModel:
    """MLP that produces pre-activations from concatenated z^t and o^t.

    Input:  concat(z^t, o^t) of size (d_model + d_input)
    Output: pre-activations a^t of size d_model

    The synapse is the only shared computation in the CTM — every neuron
    sees the same pre-activations before the NLMs branch into per-neuron
    processing.  Two-layer MLP with SiLU activation between layers.

    W1: first-layer weights, projects the concatenated input down to the
        hidden dimension.  Shape (d_hidden_syn, d_model + d_input).
    b1: first-layer biases.  Shape (d_hidden_syn,).
    W2: second-layer weights, projects hidden back to model dimension to
        produce one pre-activation per neuron.  Shape (d_model, d_hidden_syn).
    b2: second-layer biases.  Shape (d_model,).
    """
    W1: Matrix
    b1: Vector
    W2: Matrix
    b2: Vector


@dataclass
class NLMParams:
    """Per-neuron MLPs processing pre-activation histories.

    Each neuron d has its own MLP: R^M -> R^d_hidden -> R^1
    We store all D neurons' weights together for clarity.
    """
    # W1[d]: (d_hidden, M) — first layer weights for neuron d
    # b1[d]: (d_hidden,)   — first layer biases for neuron d
    # W2[d]: (1, d_hidden) — second layer weights for neuron d
    # b2[d]: (1,)          — second layer biases for neuron d
    W1: list[Matrix]
    b1: list[Vector]
    W2: list[Matrix]
    b2: list[Vector]


@dataclass
class SyncPairs:
    """Neuron pairs for synchronization subsampling.

    Computing sync for all O(d_model²) pairs is expensive. Instead, we split
    neurons into two groups of J each and only compute cross-group pairs,
    giving J² pairs — a ~4x reduction for typical configurations.
    """
    left_indices: list[int]    # J1 neuron indices
    right_indices: list[int]   # J2 neuron indices
    n_pairs: int               # len(left) * len(right)


@dataclass
class SyncState:
    """Running state for recursive synchronization computation.

    Tracks how pairs of neurons correlate over thinking steps:
      α accumulates z_i * z_j products (how similarly two neurons fire)
      β counts effective time steps (normalizer, prevents α from growing forever)
      S = α / sqrt(β) is the final synchronization value

    Each tick updates recursively:
      α_{t+1} = e^{-r} · α_t + z_i · z_j   (old correlations decay, new one added)
      β_{t+1} = e^{-r} · β_t + 1            (old counts decay, new count added)

    The decay rate r is learnable — the network chooses how far back to remember.
    """
    alpha: list[float]         # (n_pairs,) — weighted dot product accumulator
    beta: list[float]          # (n_pairs,) — decay normalization accumulator
    decay_rates: list[float]   # (n_pairs,) — learnable r_ij >= 0


@dataclass
class CTM:
    """Continuous Thought Machine.

    Two synchronization pathways serve different purposes:
      sync_out  → drives class prediction (output projection)
      sync_action → drives attention query (what to look at next in the input)
    """
    synapse: SynapseModel
    nlm: NLMParams
    sync_out: SyncState         # synchronization for output projection
    sync_action: SyncState      # synchronization for attention query
    pairs_out: SyncPairs        # neuron pairs for output sync
    pairs_action: SyncPairs     # neuron pairs for action sync
    W_out: Matrix               # (num_classes, n_pairs_out) — output projection
    b_out: Vector               # (num_classes,)
    W_query: Matrix             # (d_input, n_pairs_action) — query projection
    # Input processing: embedding ±1 values + positional encoding
    W_embed: Matrix             # (d_embed, 2) — embed ±1 as learned vectors
    W_kv: Matrix                # (d_input, d_embed) — project embeddings to KV space
    b_kv: Vector                # (d_input,)
    # Attention weights
    W_attn_k: Matrix            # (d_input, d_input) — key projection
    W_attn_v: Matrix            # (d_input, d_input) — value projection
    # Learnable initial state
    z_init: Vector              # (d_model,) — initial post-activations
    # Hyperparameters
    d_model: int                # number of neurons (width of internal state z)
    d_input: int                # dimension of attention output / KV space
    d_embed: int                # dimension of input embeddings (before KV projection)
    d_hidden_syn: int           # synapse MLP hidden layer width
    d_hidden_nlm: int           # per-neuron NLM hidden layer width
    M: int                      # memory window — each NLM sees its last M pre-activations
    T: int                      # number of internal thinking ticks per forward pass
    num_classes: int            # output classes (2 for parity)
    seq_len: int                # input sequence length


@dataclass
class TickCache:
    """Cache for a single internal tick's computation."""
    # Synapse
    syn_input: Vector           # concat(z^t, o^t)
    syn_h: Vector               # hidden layer activations
    syn_pre_h: Vector           # pre-activation of hidden layer
    a: Vector                   # pre-activations output

    # NLM
    nlm_inputs: list[Vector]    # (D,) each is (M,) history slice
    nlm_h: list[Vector]         # (D,) hidden activations per neuron
    nlm_pre_h: list[Vector]     # (D,) pre-activation of hidden per neuron
    z_new: Vector               # (D,) post-activations = NLM output

    # Synchronization (we store the alpha/beta BEFORE this tick's update)
    alpha_out_prev: list[float]     # previous tick's correlation accumulator (output path)
    beta_out_prev: list[float]      # previous tick's normalization accumulator (output path)
    alpha_action_prev: list[float]  # previous tick's correlation accumulator (action path)
    beta_action_prev: list[float]   # previous tick's normalization accumulator (action path)
    z_left_out: list[float]         # z values at left neuron indices for output pairs
    z_right_out: list[float]        # z values at right neuron indices for output pairs
    z_left_action: list[float]      # z values at left neuron indices for action pairs
    z_right_action: list[float]     # z values at right neuron indices for action pairs
    S_out: list[float]              # α/√β synchronization vector → output projection
    S_action: list[float]           # α/√β synchronization vector → attention query

    # Output
    logits: Vector              # (num_classes,) raw logits
    probs: Vector               # (num_classes,) softmax probabilities

    # Attention
    query: Vector               # (d_input,) projected from S_action
    attn_scores: list[float]    # (seq_len,) attention scores
    attn_weights: list[float]   # (seq_len,) softmax weights
    attn_output: Vector         # (d_input,) weighted sum of values
    K: Matrix                   # (seq_len, d_input) keys
    V: Matrix                   # (seq_len, d_input) values


@dataclass
class CTMCache:
    """Full cache for backward pass."""
    tick_caches: list[TickCache]
    pre_act_history: list[Vector]   # all pre-activations across ticks
    post_act_history: list[Vector]  # all post-activations across ticks
    embedded_input: Matrix          # (seq_len, d_embed) embedded input sequence
    kv_input: Matrix                # (seq_len, d_input) projected for KV (post layer-norm)
    kv_pre_norm: Matrix             # (seq_len, d_input) pre-layer-norm KV projections
    input_indices: list[int]        # (seq_len,) which embedding row each input used (0 or 1)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def _create_sync_pairs(rng, d_model: int, J: int) -> SyncPairs:
    """Create semi-dense neuron pairing: two disjoint subsets of size J."""
    indices = list(range(d_model))
    rng.shuffle(indices)
    left = sorted(indices[:J])
    right = sorted(indices[J:2 * J])
    n_pairs = J * J
    return SyncPairs(
        left_indices=left,
        right_indices=right,
        n_pairs=n_pairs,
    )


def _create_sync_state(n_pairs: int) -> SyncState:
    """Initialize synchronization state with zero accumulators and zero decay."""
    return SyncState(
        alpha=vector.zeros(n_pairs),
        beta=vector.zeros(n_pairs),
        decay_rates=vector.zeros(n_pairs),  # r_ij=0 means no decay initially
    )


def create_ctm(
    rng,
    d_model: int = 64,
    d_input: int = 32,
    d_embed: int = 16,
    d_hidden_syn: int = 64,
    d_hidden_nlm: int = 4,
    M: int = 5,
    T: int = 10,
    num_classes: int = 2,
    seq_len: int = 16,
    J_out: int = 16,
    J_action: int = 16,
) -> CTM:
    """Create a CTM with all parameters Xavier-initialized.

    rng:            random number generator (from create_rng)
    d_model:        number of neurons — width of internal state z
    d_input:        dimension of attention output and KV space
    d_embed:        dimension of input embeddings (before KV projection)
    d_hidden_syn:   synapse MLP hidden layer width
    d_hidden_nlm:   per-neuron NLM hidden layer width
    M:              memory window — each NLM sees its last M pre-activations
    T:              number of internal thinking ticks per forward pass
    num_classes:    output classes (2 for parity)
    seq_len:        input sequence length
    J_out:          neurons per side in the output sync pairing — produces
                    J_out² pairs that drive class prediction
    J_action:       neurons per side in the action sync pairing — produces
                    J_action² pairs that drive the attention query
    """
    # Synapse model: concat(z, o) -> hidden -> pre-activations
    syn_input_dim = d_model + d_input
    synapse = SynapseModel(
        W1=xavier_init(rng, syn_input_dim, d_hidden_syn),
        b1=vector.zeros(d_hidden_syn),
        W2=xavier_init(rng, d_hidden_syn, d_model),
        b2=vector.zeros(d_model),
    )

    # NLMs: one MLP per neuron, M -> d_hidden_nlm -> 1
    nlm_W1 = [xavier_init(rng, M, d_hidden_nlm) for _ in range(d_model)]
    nlm_b1 = [vector.zeros(d_hidden_nlm) for _ in range(d_model)]
    nlm_W2 = [xavier_init(rng, d_hidden_nlm, 1) for _ in range(d_model)]
    nlm_b2 = [vector.zeros(1) for _ in range(d_model)]
    nlm = NLMParams(
        W1=nlm_W1,
        b1=nlm_b1,
        W2=nlm_W2,
        b2=nlm_b2,
    )

    # Synchronization pairs and state
    pairs_out = _create_sync_pairs(rng, d_model, J_out)
    pairs_action = _create_sync_pairs(rng, d_model, J_action)
    sync_out = _create_sync_state(pairs_out.n_pairs)
    sync_action = _create_sync_state(pairs_action.n_pairs)

    # Output projection from sync
    limit_out = math.sqrt(6.0 / (num_classes + pairs_out.n_pairs))
    W_out = random_matrix(rng, num_classes, pairs_out.n_pairs, -limit_out, limit_out)
    b_out = vector.zeros(num_classes)

    # Query projection from action sync
    limit_q = math.sqrt(6.0 / (d_input + pairs_action.n_pairs))
    W_query = random_matrix(rng, d_input, pairs_action.n_pairs, -limit_q, limit_q)

    # Input embeddings: ±1 mapped to index 0 or 1
    W_embed = random_matrix(rng, 2, d_embed, -0.1, 0.1)

    # KV projection
    W_kv = xavier_init(rng, d_embed, d_input)
    b_kv = vector.zeros(d_input)

    # Attention key/value projections
    W_attn_k = xavier_init(rng, d_input, d_input)
    W_attn_v = xavier_init(rng, d_input, d_input)

    # Learnable initial z
    z_init = random_vector(rng, d_model, -0.1, 0.1)

    return CTM(
        synapse=synapse,
        nlm=nlm,
        sync_out=sync_out,
        sync_action=sync_action,
        pairs_out=pairs_out,
        pairs_action=pairs_action,
        W_out=W_out,
        b_out=b_out,
        W_query=W_query,
        W_embed=W_embed,
        W_kv=W_kv,
        b_kv=b_kv,
        W_attn_k=W_attn_k,
        W_attn_v=W_attn_v,
        z_init=z_init,
        d_model=d_model,
        d_input=d_input,
        d_embed=d_embed,
        d_hidden_syn=d_hidden_syn,
        d_hidden_nlm=d_hidden_nlm,
        M=M,
        T=T,
        num_classes=num_classes,
        seq_len=seq_len,
    )


# ---------------------------------------------------------------------------
# Forward pass components
# ---------------------------------------------------------------------------

def _synapse_forward(syn: SynapseModel, z: Vector, o: Vector) -> tuple[Vector, Vector, Vector]:
    """Synapse model: concat(z, o) -> SiLU hidden -> linear output.

    z: current internal state — post-activations from the previous tick (d_model,)
    o: observation vector from cross-attention over the input (d_input,)

    Returns: (a, h, pre_h) where a = pre-activations fed to NLMs.
    """
    inp = vector.concat(z, o)
    # Hidden layer
    pre_h = vector.add(matrix.mat_vec(syn.W1, inp), syn.b1)
    h = vector.apply(silu, pre_h)
    # Output layer (linear)
    a = vector.add(matrix.mat_vec(syn.W2, h), syn.b2)
    return a, h, pre_h


def _nlm_forward(
    nlm: NLMParams, pre_act_history: list[Vector], d_model: int, M: int
) -> tuple[Vector, list[Vector], list[Vector]]:
    """Apply each neuron's private MLP to its recent pre-activation history.

    Each of the d_model neurons has its own 2-layer MLP. It looks at the
    last M pre-activations it received (padded with zeros if fewer than M
    ticks have passed) and produces a single output — the neuron's new
    post-activation value.

    This is the key CTM innovation: neurons aren't just dot-product-plus-ReLU.
    Each one has a private model that processes its own history, giving it
    a kind of temporal memory. A standard neuron is a stateless function
    of its current input; an NLM neuron is a function of its recent past,
    letting it track patterns across thinking steps.

    nlm:              weights for all d_model private MLPs (W1, b1, W2, b2 per neuron)
    pre_act_history:  list of pre-activation vectors from previous ticks, oldest first.
                      Each entry is (d_model,) — the synapse output at that tick.
                      Only the most recent M entries are used; earlier ticks are ignored.
    d_model:          number of neurons (each gets its own MLP)
    M:                memory window — how many past pre-activations each neuron sees

    Returns:
      z_new:          (d_model,) — new post-activation state, one value per neuron
      all_h:          (d_model,) list of (d_hidden_nlm,) — hidden activations per neuron
                      (retained for backprop)
      all_pre_h:      (d_model,) list of (d_hidden_nlm,) — pre-activation of hidden layer
                      per neuron (retained for backprop)
    """
    # Build history matrix: pad with zeros if fewer than M entries
    n_hist = len(pre_act_history)
    z_new: Vector = []
    all_h: list[Vector] = []
    all_pre_h: list[Vector] = []

    for d in range(d_model):
        # Extract this neuron's history (last M values)
        hist: Vector = []
        for t_idx in range(M):
            source_idx = n_hist - M + t_idx
            if source_idx >= 0:
                hist.append(pre_act_history[source_idx][d])
            else:
                hist.append(0.0)

        # MLP: hist (M,) -> hidden (d_hidden,) -> output (1,)
        pre_h = vector.add(matrix.mat_vec(nlm.W1[d], hist), nlm.b1[d])
        h = vector.apply(silu, pre_h)
        out = vector.add(matrix.mat_vec(nlm.W2[d], h), nlm.b2[d])
        z_new.append(out[0])
        all_h.append(h)
        all_pre_h.append(pre_h)

    return z_new, all_h, all_pre_h


def _sync_update(
    state: SyncState, z: Vector, pairs: SyncPairs
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Recursive synchronization: measure temporal correlation between neuron pairs.

    For each pair (i, j), tracks a decaying weighted sum of z_i * z_j over ticks.
    When two neurons fire similarly across ticks, their sync value S grows.
    When they diverge, S shrinks. The decay rate r controls the time horizon —
    large r means only recent ticks matter; small r means long memory.

    The normalization S = α / sqrt(β) prevents the values from growing without
    bound as more ticks accumulate.

    state:  running α, β accumulators and learnable decay rates for each pair
    z:      post-activation state from the NLMs — z^{t+1}, shape (d_model,).
            Individual neuron values z[i], z[j] are read for each pair.
    pairs:  which neuron indices to pair (left × right)

    Returns:
      S_vector:     (n_pairs,) — synchronization values α/√β for this tick
      new_alpha:    (n_pairs,) — updated correlation accumulators
      new_beta:     (n_pairs,) — updated normalization accumulators
      z_left_vals:  (J,) — z values at left neuron indices (retained for backprop)
      z_right_vals: (J,) — z values at right neuron indices (retained for backprop)
    """
    new_alpha: list[float] = []
    new_beta: list[float] = []
    S: list[float] = []
    z_left: list[float] = []
    z_right: list[float] = []

    pair_idx = 0
    for i in pairs.left_indices:
        for j in pairs.right_indices:
            zi = z[i]
            zj = z[j]
            r = state.decay_rates[pair_idx]
            decay = scalar.exp(scalar.negate(r))  # e^{-r}: how much old values are kept (0=forget, 1=remember)

            # α^{t+1} = e^{-r} * α^t + z_i * z_j
            a_new = scalar.add(
                scalar.multiply(decay, state.alpha[pair_idx]),
                scalar.multiply(zi, zj),
            )
            # β^{t+1} = e^{-r} * β^t + 1
            b_new = scalar.add(scalar.multiply(decay, state.beta[pair_idx]), 1.0)

            new_alpha.append(a_new)
            new_beta.append(b_new)

            # S = α / sqrt(β)
            s_val = scalar.multiply(a_new, scalar.inverse(scalar.power(b_new, 0.5)))
            S.append(s_val)

            z_left.append(zi)
            z_right.append(zj)
            pair_idx += 1

    return S, new_alpha, new_beta, z_left, z_right


def _cross_attention_forward(
    query: Vector, K: Matrix, V: Matrix, d_input: int
) -> tuple[Vector, list[float], list[float]]:
    """Single-query cross-attention: query attends to all K/V positions.

    query:    (d_input,) — derived from S_action via W_query; tells the network
              what to look for in the input this tick
    K:        (seq_len, d_input) — key matrix, projected once from the embedded
              input before the tick loop. Each row is one input position's key.
    V:        (seq_len, d_input) — value matrix, same origin as K. Each row is
              the content the network retrieves when it attends to that position.
    d_input:  key/value dimension, used for scaled dot-product normalization

    Returns:
      output:   (d_input,) — weighted sum of V rows; the observation o^t
      scores:   (seq_len,) — raw dot-product scores before softmax
      weights:  (seq_len,) — attention weights after softmax (sum to 1)
    """
    scale = math.sqrt(d_input)
    seq_len = len(K)

    # scores[j] = dot(query, K[j]) / sqrt(d_input)
    scores = [vector.dot(query, K[j]) / scale for j in range(seq_len)]
    weights = softmax(scores)

    # output = sum_j weights[j] * V[j]
    output = vector.zeros(d_input)
    for j in range(seq_len):
        output = vector.add(output, vector.scale(weights[j], V[j]))

    return output, scores, weights


# ---------------------------------------------------------------------------
# Full forward pass
# ---------------------------------------------------------------------------

def ctm_forward(model: CTM, input_seq: list[float]) -> tuple[list[Vector], CTMCache]:
    """Forward pass through the CTM over T internal ticks.

    Unlike lessons 1–4 where a forward pass is a single sweep through layers,
    the CTM forward pass is a loop: the network "thinks" for T steps, each
    building on the previous. Every tick runs the full pipeline:

      1. Synapse — concat(z, o) → MLP → pre-activations a
      2. NLMs   — each neuron's private MLP reads its recent a history → z_new
      3. Sync   — measure temporal correlation between neuron pairs → S vectors
      4. Output — project S_out to class logits (prediction at this tick)
      5. Attend — project S_action to query, cross-attend to input → new o

    The input sequence is embedded and projected to keys/values once before
    the loop. Each tick re-reads the input through attention with a new query,
    so the network can shift its focus as it thinks.

    The model produces a prediction at every tick — the loss function later
    selects the best two (most accurate and most confident) for training.

    input_seq:  list of ±1 values, length seq_len

    Returns:
      per_tick_probs:  list of T probability vectors, each (num_classes,)
      cache:           CTMCache with all intermediates needed for backward
    """
    # Embed input: ±1 -> index 0 or 1 -> embedding vector
    embedded: Matrix = []
    input_indices: list[int] = []
    for val in input_seq:
        idx = 0 if val < 0 else 1
        input_indices.append(idx)
        embedded.append(list(model.W_embed[idx]))
    # Project to KV space (cache pre-norm for backward pass)
    kv_pre_norm: Matrix = []
    kv_input: Matrix = []
    for emb in embedded:
        proj = vector.add(matrix.mat_vec(model.W_kv, emb), model.b_kv)
        kv_pre_norm.append(proj)
        kv_input.append(layer_norm(proj))

    # Compute keys and values (fixed across ticks)
    K: Matrix = [matrix.mat_vec(model.W_attn_k, kv) for kv in kv_input]
    V: Matrix = [matrix.mat_vec(model.W_attn_v, kv) for kv in kv_input]

    # Initialize state
    z = list(model.z_init)
    o = vector.zeros(model.d_input)  # initial attention output
    pre_act_history: list[Vector] = []
    post_act_history: list[Vector] = [list(z)]

    # Reset sync states
    alpha_out = list(model.sync_out.alpha)  # zeros
    beta_out = list(model.sync_out.beta)
    alpha_action = list(model.sync_action.alpha)
    beta_action = list(model.sync_action.beta)

    tick_caches: list[TickCache] = []
    per_tick_probs: list[Vector] = []

    for t in range(model.T):
        # Each tick: synapse → NLMs → sync → predict → attend → repeat

        # 1. Synapse: produce pre-activations
        a, syn_h, syn_pre_h = _synapse_forward(model.synapse, z, o)
        pre_act_history.append(a)

        # 2. NLMs: process pre-activation history
        z_new, nlm_h_list, nlm_pre_h_list = _nlm_forward(
            model.nlm, pre_act_history, model.d_model, model.M
        )
        post_act_history.append(z_new)

        # Build NLM input slices for cache
        n_hist = len(pre_act_history)
        nlm_inputs: list[Vector] = []
        for d in range(model.d_model):
            hist: Vector = []
            for t_idx in range(model.M):
                source_idx = n_hist - model.M + t_idx
                if source_idx >= 0:
                    hist.append(pre_act_history[source_idx][d])
                else:
                    hist.append(0.0)
            nlm_inputs.append(hist)

        # 3. Synchronization updates
        alpha_out_prev = list(alpha_out)
        beta_out_prev = list(beta_out)
        alpha_action_prev = list(alpha_action)
        beta_action_prev = list(beta_action)

        # Temporarily set state for sync update
        model.sync_out.alpha = alpha_out
        model.sync_out.beta = beta_out
        S_out, alpha_out, beta_out, zl_out, zr_out = _sync_update(
            model.sync_out, z_new, model.pairs_out
        )

        model.sync_action.alpha = alpha_action
        model.sync_action.beta = beta_action
        S_action, alpha_action, beta_action, zl_act, zr_act = _sync_update(
            model.sync_action, z_new, model.pairs_action
        )

        # 4. Output: project S_out to class logits
        logits = vector.add(matrix.mat_vec(model.W_out, S_out), model.b_out)
        probs = softmax(logits)
        per_tick_probs.append(probs)

        # 5. Attention: project S_action to query, attend to input
        query = matrix.mat_vec(model.W_query, S_action)
        attn_out, attn_scores, attn_weights = _cross_attention_forward(
            query, K, V, model.d_input
        )

        # Save cache
        tick_caches.append(TickCache(
            syn_input=vector.concat(z, o),
            syn_h=syn_h,
            syn_pre_h=syn_pre_h,
            a=a,
            nlm_inputs=nlm_inputs,
            nlm_h=nlm_h_list,
            nlm_pre_h=nlm_pre_h_list,
            z_new=z_new,
            alpha_out_prev=alpha_out_prev,
            beta_out_prev=beta_out_prev,
            alpha_action_prev=alpha_action_prev,
            beta_action_prev=beta_action_prev,
            z_left_out=zl_out,
            z_right_out=zr_out,
            z_left_action=zl_act,
            z_right_action=zr_act,
            S_out=S_out,
            S_action=S_action,
            logits=logits,
            probs=probs,
            query=query,
            attn_scores=attn_scores,
            attn_weights=attn_weights,
            attn_output=attn_out,
            K=K,
            V=V,
        ))

        # Update state for next tick
        z = z_new
        o = attn_out

    cache = CTMCache(
        tick_caches=tick_caches,
        pre_act_history=pre_act_history,
        post_act_history=post_act_history,
        embedded_input=embedded,
        kv_input=kv_input,
        kv_pre_norm=kv_pre_norm,
        input_indices=input_indices,
    )
    return per_tick_probs, cache


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ctm_loss(
    per_tick_probs: list[Vector], target: Vector
) -> tuple[float, int, int, list[float], list[float]]:
    """Certainty-based loss: trains the network to be both accurate and confident.

    The CTM predicts at every tick. Rather than picking one tick to supervise,
    we find two special ticks:
      t1 = the tick where the prediction was most accurate (lowest cross-entropy)
      t2 = the tick where the network was most confident (lowest entropy)

    Loss = average of L(t1) and L(t2). This encourages the network to:
      - Get the right answer (via t1)
      - Know when it has the right answer (via t2)
      - On easy inputs, t1 ≈ t2 at an early tick → adaptive compute

    Certainty is measured as 1 - normalized_entropy:
      0 = uniform distribution (completely uncertain, maximum entropy)
      1 = one-hot distribution (completely certain, zero entropy)

    Returns (loss, t1, t2, per_tick_losses, per_tick_certainties).
    """
    num_classes = len(target)
    max_entropy = math.log(num_classes)
    T = len(per_tick_probs)

    per_tick_losses: list[float] = []
    per_tick_certainties: list[float] = []

    for t in range(T):
        # Cross-entropy loss at this tick
        loss_t = cross_entropy(per_tick_probs[t], target)
        per_tick_losses.append(loss_t)

        # Certainty = 1 - normalized_entropy
        entropy = 0.0
        for p in per_tick_probs[t]:
            if p > 1e-10:
                entropy -= p * math.log(p)
        certainty = 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0
        per_tick_certainties.append(certainty)

    # t1 = tick with minimum loss
    t1 = 0
    for t in range(1, T):
        if per_tick_losses[t] < per_tick_losses[t1]:
            t1 = t

    # t2 = tick with maximum certainty
    t2 = 0
    for t in range(1, T):
        if per_tick_certainties[t] > per_tick_certainties[t2]:
            t2 = t

    loss = (per_tick_losses[t1] + per_tick_losses[t2]) / 2.0
    return loss, t1, t2, per_tick_losses, per_tick_certainties


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def ctm_backward(model: CTM, cache: CTMCache, target: Vector) -> dict:
    """Backward pass: backpropagation through time (BPTT).

    This is the most involved backward pass in the project — by a wide margin.
    Previous lessons backprop through a fixed stack of layers; here we unroll
    T thinking ticks, each containing a synapse, d_model private NLMs, two
    synchronization pathways with learnable decay rates, an output head, and
    cross-attention back into the input. Every component feeds the next tick,
    so gradients must flow backward through the full tick chain.

    What makes it harder than standard BPTT:

    1. Two loss injection points — only ticks t1 (most accurate) and t2 (most
       confident) receive direct loss gradients; other ticks propagate gradients
       solely through the z and sync state chains.
    2. Per-neuron NLMs — each of the d_model neurons has its own MLP with its
       own weight gradients, and each reads from a sliding history window that
       overlaps across ticks.
    3. Synchronization — α and β accumulators carry state across ticks with
       learnable decay rates, adding two more gradient chains (one per pathway)
       on top of the z chain.
    4. Cross-attention — the query is derived from the action sync pathway,
       so attention gradients flow back through sync, through z, and ultimately
       into the synapse and NLMs.

    model:   the CTM with all parameters
    cache:   forward-pass cache (CTMCache) containing per-tick intermediates
    target:  one-hot target vector (num_classes,)

    Returns dict of parameter gradients keyed by parameter name.
    """
    per_tick_probs = [tc.probs for tc in cache.tick_caches]
    _, t1, t2, _, _ = ctm_loss(per_tick_probs, target)
    T = model.T

    # Initialize gradient accumulators
    d_W_out = matrix.zeros(len(model.W_out), len(model.W_out[0]))
    d_b_out = vector.zeros(model.num_classes)
    d_W_query = matrix.zeros(len(model.W_query), len(model.W_query[0]))
    d_syn_W1 = matrix.zeros(len(model.synapse.W1), len(model.synapse.W1[0]))
    d_syn_b1 = vector.zeros(len(model.synapse.b1))
    d_syn_W2 = matrix.zeros(len(model.synapse.W2), len(model.synapse.W2[0]))
    d_syn_b2 = vector.zeros(len(model.synapse.b2))
    d_decay_out = vector.zeros(model.pairs_out.n_pairs)
    d_decay_action = vector.zeros(model.pairs_action.n_pairs)
    d_z_init = vector.zeros(model.d_model)

    # NLM gradient accumulators
    d_nlm_W1 = [matrix.zeros(model.d_hidden_nlm, model.M) for _ in range(model.d_model)]
    d_nlm_b1 = [vector.zeros(model.d_hidden_nlm) for _ in range(model.d_model)]
    d_nlm_W2 = [matrix.zeros(1, model.d_hidden_nlm) for _ in range(model.d_model)]
    d_nlm_b2 = [vector.zeros(1) for _ in range(model.d_model)]

    # Attention weight grads
    d_W_attn_k = matrix.zeros(model.d_input, model.d_input)
    d_W_attn_v = matrix.zeros(model.d_input, model.d_input)
    d_W_kv = matrix.zeros(model.d_input, model.d_embed)
    d_b_kv = vector.zeros(model.d_input)
    d_W_embed = matrix.zeros(2, model.d_embed)

    # Running gradients flowing backward through ticks
    d_z = vector.zeros(model.d_model)  # gradient w.r.t. z at current tick
    d_o = vector.zeros(model.d_input)  # gradient w.r.t. attention output
    # Gradients flowing back through sync state (alpha, beta)
    d_alpha_out = vector.zeros(model.pairs_out.n_pairs)
    d_beta_out = vector.zeros(model.pairs_out.n_pairs)
    d_alpha_action = vector.zeros(model.pairs_action.n_pairs)
    d_beta_action = vector.zeros(model.pairs_action.n_pairs)

    for t in range(T - 1, -1, -1):
        tc = cache.tick_caches[t]

        # --- Loss gradient at this tick ---
        # Only t1 and t2 contribute to loss. Each gets 0.5 weight,
        # unless t1 == t2 (same tick is both most accurate and most confident),
        # in which case it gets full weight 1.0.
        d_logits = vector.zeros(model.num_classes)
        if t == t1 or t == t2:
            scale_factor = 0.5
            if t1 == t2:
                scale_factor = 1.0
            d_logits = vector.scale(scale_factor,
                                    vector.subtract(tc.probs, target))

        # --- Backward through output projection ---
        # logits = W_out @ S_out + b_out
        d_b_out = vector.add(d_b_out, d_logits)
        # d_W_out += outer(d_logits, S_out)
        for row in range(model.num_classes):
            for col in range(model.pairs_out.n_pairs):
                d_W_out[row][col] += d_logits[row] * tc.S_out[col]
        d_S_out = matrix.mat_vec(matrix.transpose(model.W_out), d_logits)

        # --- Backward through output sync ---
        # Add gradient from future ticks' alpha/beta dependencies
        d_S_out_total = d_S_out
        d_z_from_sync_out, d_decay_out_t, d_alpha_out_prev, d_beta_out_prev = _sync_backward(
            d_S_out_total, d_alpha_out, d_beta_out,
            tc, model.pairs_out, model.sync_out.decay_rates,
            is_out=True,
        )
        d_alpha_out = d_alpha_out_prev
        d_beta_out = d_beta_out_prev
        d_decay_out = vector.add(d_decay_out, d_decay_out_t)

        # --- Backward through attention query and cross-attention ---
        # query = W_query @ S_action
        # Gradient from attention output flows through o
        d_query = vector.add(
            # From attention backward
            _cross_attention_query_grad(d_o, tc),
            vector.zeros(model.d_input),  # no other source of query grad
        )
        # d_W_query += outer(d_query, S_action)
        for row in range(model.d_input):
            for col in range(model.pairs_action.n_pairs):
                d_W_query[row][col] += d_query[row] * tc.S_action[col]
        d_S_action = matrix.mat_vec(matrix.transpose(model.W_query), d_query)

        # Backward through attention to KV
        d_K_t, d_V_t = _cross_attention_kv_grad(d_o, tc)
        for pos in range(model.seq_len):
            # K[pos] = W_attn_k @ kv_input[pos]
            for row in range(model.d_input):
                for col in range(model.d_input):
                    d_W_attn_k[row][col] += d_K_t[pos][row] * cache.kv_input[pos][col]
            # V[pos] = W_attn_v @ kv_input[pos]
            for row in range(model.d_input):
                for col in range(model.d_input):
                    d_W_attn_v[row][col] += d_V_t[pos][row] * cache.kv_input[pos][col]

            # Continue the chain: kv_input → layer_norm → W_kv @ emb + b_kv → W_embed
            # d_kv_input = W_attn_k^T @ d_K + W_attn_v^T @ d_V
            d_kv = vector.add(
                matrix.mat_vec(matrix.transpose(model.W_attn_k), d_K_t[pos]),
                matrix.mat_vec(matrix.transpose(model.W_attn_v), d_V_t[pos]),
            )
            # Backward through layer norm
            d_pre_norm = layer_norm_backward(
                d_kv, cache.kv_input[pos], cache.kv_pre_norm[pos],
            )
            # d_W_kv += outer(d_pre_norm, embedded[pos]), d_b_kv += d_pre_norm
            emb = cache.embedded_input[pos]
            for row in range(model.d_input):
                for col in range(model.d_embed):
                    d_W_kv[row][col] += d_pre_norm[row] * emb[col]
            d_b_kv = vector.add(d_b_kv, d_pre_norm)
            # d_emb = W_kv^T @ d_pre_norm → accumulate into d_W_embed
            d_emb = matrix.mat_vec(matrix.transpose(model.W_kv), d_pre_norm)
            idx = cache.input_indices[pos]
            for col in range(model.d_embed):
                d_W_embed[idx][col] += d_emb[col]

        # --- Backward through action sync ---
        d_z_from_sync_action, d_decay_action_t, d_alpha_action_prev, d_beta_action_prev = _sync_backward(
            d_S_action, d_alpha_action, d_beta_action,
            tc, model.pairs_action, model.sync_action.decay_rates,
            is_out=False,
        )
        d_alpha_action = d_alpha_action_prev
        d_beta_action = d_beta_action_prev
        d_decay_action = vector.add(d_decay_action, d_decay_action_t)

        # --- Combine z gradients ---
        # d_z_new comes from: sync_out, sync_action, and next tick's d_z flowing through synapse
        d_z_new = vector.add(
            vector.add(d_z_from_sync_out, d_z_from_sync_action),
            d_z,  # from next tick's synapse input
        )

        # --- Backward through NLMs ---
        d_pre_act_contributions = _nlm_backward(
            model, tc, d_z_new,
            d_nlm_W1, d_nlm_b1, d_nlm_W2, d_nlm_b2,
        )

        # --- Backward through synapse ---
        d_syn_input = _synapse_backward(
            model.synapse, tc, d_pre_act_contributions,
            d_syn_W1, d_syn_b1, d_syn_W2, d_syn_b2,
        )

        # Split d_syn_input into d_z (for previous tick) and d_o (for previous tick)
        d_z = d_syn_input[:model.d_model]
        d_o = d_syn_input[model.d_model:]

    # d_z now holds gradient w.r.t. z_init
    d_z_init = vector.add(d_z_init, d_z)

    return {
        "d_W_out": d_W_out, "d_b_out": d_b_out,
        "d_W_query": d_W_query,
        "d_syn_W1": d_syn_W1, "d_syn_b1": d_syn_b1,
        "d_syn_W2": d_syn_W2, "d_syn_b2": d_syn_b2,
        "d_nlm_W1": d_nlm_W1, "d_nlm_b1": d_nlm_b1,
        "d_nlm_W2": d_nlm_W2, "d_nlm_b2": d_nlm_b2,
        "d_decay_out": d_decay_out, "d_decay_action": d_decay_action,
        "d_z_init": d_z_init,
        "d_W_attn_k": d_W_attn_k, "d_W_attn_v": d_W_attn_v,
        "d_W_kv": d_W_kv, "d_b_kv": d_b_kv,
        "d_W_embed": d_W_embed,
    }


def _sync_backward(
    d_S: list[float],
    d_alpha_future: list[float],
    d_beta_future: list[float],
    tc: TickCache, 
    pairs: SyncPairs,
    decay_rates: list[float],
    is_out: bool,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Backward through one tick's synchronization update.

    Forward was:
      α_new = e^{-r} · α_prev + z_i · z_j
      β_new = e^{-r} · β_prev + 1
      S = α_new / sqrt(β_new)

    We receive gradients from two sources:
      d_S: from the output/attention that used this tick's sync values
      d_alpha_future, d_beta_future: from the NEXT tick (because our α_new
        feeds into the next tick's α computation via the recursive formula)

    We compute gradients for:
      d_z: how neuron activations should change (feeds back to NLMs/synapse)
      d_decay: how the decay rate r should change
      d_alpha_prev, d_beta_prev: to pass to the PREVIOUS tick

    Returns (d_z, d_decay, d_alpha_prev, d_beta_prev).
    """
    n_pairs = pairs.n_pairs
    d_z_full = vector.zeros(len(tc.z_new))
    d_decay = vector.zeros(n_pairs)
    d_alpha_prev = vector.zeros(n_pairs)
    d_beta_prev = vector.zeros(n_pairs)

    alpha_prev = tc.alpha_out_prev if is_out else tc.alpha_action_prev
    beta_prev = tc.beta_out_prev if is_out else tc.beta_action_prev
    z_left = tc.z_left_out if is_out else tc.z_left_action
    z_right = tc.z_right_out if is_out else tc.z_right_action

    pair_idx = 0
    for i_idx, i in enumerate(pairs.left_indices):
        for j_idx, j in enumerate(pairs.right_indices):
            r = decay_rates[pair_idx]
            decay = scalar.exp(scalar.negate(r))
            a_prev = alpha_prev[pair_idx]
            b_prev = beta_prev[pair_idx]

            # Forward was:
            # alpha_new = decay * a_prev + z_i * z_j
            # beta_new = decay * b_prev + 1
            # S = alpha_new / sqrt(beta_new)
            alpha_new = scalar.add(scalar.multiply(decay, a_prev),
                                   scalar.multiply(z_left[pair_idx], z_right[pair_idx]))
            beta_new = scalar.add(scalar.multiply(decay, b_prev), 1.0)

            # ∂S/∂α = 1/sqrt(β)  (more correlation → higher S)
            # ∂S/∂β = -α/(2·β^{3/2})  (more normalization → lower S)
            sqrt_beta = scalar.power(beta_new, 0.5)
            inv_sqrt_beta = scalar.inverse(sqrt_beta) if sqrt_beta > 1e-10 else 0.0
            d_alpha_new = scalar.multiply(d_S[pair_idx], inv_sqrt_beta)
            d_beta_new = scalar.multiply(
                d_S[pair_idx],
                scalar.multiply(-0.5, scalar.multiply(
                    alpha_new, scalar.inverse(scalar.power(beta_new, 1.5)) if beta_new > 1e-10 else 0.0
                ))
            )

            # Add gradients from future ticks
            d_alpha_new = scalar.add(d_alpha_new, d_alpha_future[pair_idx])
            d_beta_new = scalar.add(d_beta_new, d_beta_future[pair_idx])

            # d_alpha_new -> d_decay, d_alpha_prev, d_z_i, d_z_j
            # alpha_new = decay * a_prev + z_i * z_j
            d_alpha_prev[pair_idx] = scalar.multiply(d_alpha_new, decay)
            d_zi_from_alpha = scalar.multiply(d_alpha_new, z_right[pair_idx])
            d_zj_from_alpha = scalar.multiply(d_alpha_new, z_left[pair_idx])

            # d_beta_new -> d_decay, d_beta_prev
            # beta_new = decay * b_prev + 1
            d_beta_prev[pair_idx] = scalar.multiply(d_beta_new, decay)

            # d_decay from both alpha and beta paths
            # decay = exp(-r), d_decay/d_r = -exp(-r) = -decay
            d_decay_from_alpha = scalar.multiply(d_alpha_new, a_prev)
            d_decay_from_beta = scalar.multiply(d_beta_new, b_prev)
            d_decay_total = scalar.add(d_decay_from_alpha, d_decay_from_beta)
            # d_r = d_decay_total * d_decay/d_r = d_decay_total * (-decay)
            d_decay[pair_idx] = scalar.multiply(d_decay_total, scalar.negate(decay))

            # Accumulate z gradients
            d_z_full[i] = scalar.add(d_z_full[i], d_zi_from_alpha)
            d_z_full[j] = scalar.add(d_z_full[j], d_zj_from_alpha)

            pair_idx += 1

    return d_z_full, d_decay, d_alpha_prev, d_beta_prev


def _cross_attention_query_grad(d_o: Vector, tc: TickCache) -> Vector:
    """Gradient of attention output w.r.t. the query vector.

    This is the path that connects the sync-action pathway back into the
    attention mechanism. The query is derived from S_action (via W_query),
    so gradients here flow back through sync into the tick chain.

    The forward computation was:
      scores[j] = dot(query, K[j]) / sqrt(d)
      weights   = softmax(scores)
      output    = sum_j weights[j] * V[j]

    We reverse through: output → weights → softmax → scores → query.

    d_o:  gradient w.r.t. attention output (d_input,)
    tc:   cached attention weights, K, V, and query from this tick

    Returns gradient w.r.t. query (d_input,).
    """
    d_input = len(d_o)
    seq_len = len(tc.attn_weights)
    scale = math.sqrt(d_input)

    # d_output -> d_weights
    # output = sum_j w_j * V_j
    d_weights: list[float] = []
    for j in range(seq_len):
        dw = vector.dot(d_o, tc.V[j])
        d_weights.append(dw)

    # d_weights -> d_scores (through softmax)
    dot_prod = sum(d_weights[j] * tc.attn_weights[j] for j in range(seq_len))
    d_scores: list[float] = []
    for j in range(seq_len):
        ds = tc.attn_weights[j] * (d_weights[j] - dot_prod)
        d_scores.append(ds)

    # d_scores -> d_query
    # scores[j] = dot(query, K[j]) / scale
    d_query = vector.zeros(d_input)
    for j in range(seq_len):
        d_query = vector.add(d_query, vector.scale(d_scores[j] / scale, tc.K[j]))

    return d_query


def _cross_attention_kv_grad(d_o: Vector, tc: TickCache) -> tuple[Matrix, Matrix]:
    """Gradient of attention output w.r.t. the K and V matrices.

    Unlike the query (which changes every tick), K and V are computed once
    from the embedded input and reused across all ticks. Their gradients
    therefore accumulate across every tick in the main backward loop and
    eventually flow back into the embedding and KV projection weights.

    The forward computation is the same as _cross_attention_query_grad,
    but here we differentiate w.r.t. K (through scores) and V (directly
    from the weighted sum).

    d_o:  gradient w.r.t. attention output (d_input,)
    tc:   cached attention weights, K, V, and query from this tick

    Returns:
      d_K:  (seq_len, d_input) — gradient w.r.t. key matrix
      d_V:  (seq_len, d_input) — gradient w.r.t. value matrix
    """
    d_input = len(d_o)
    seq_len = len(tc.attn_weights)
    scale = math.sqrt(d_input)

    # d_output -> d_V and d_weights
    d_V: Matrix = []
    d_weights: list[float] = []
    for j in range(seq_len):
        # d_V[j] = weights[j] * d_output
        d_V.append(vector.scale(tc.attn_weights[j], d_o))
        # d_weights[j] = dot(d_output, V[j])
        d_weights.append(vector.dot(d_o, tc.V[j]))

    # d_weights -> d_scores (through softmax)
    dot_prod = sum(d_weights[j] * tc.attn_weights[j] for j in range(seq_len))
    d_scores = [tc.attn_weights[j] * (d_weights[j] - dot_prod) for j in range(seq_len)]

    # d_scores -> d_K
    # scores[j] = dot(query, K[j]) / scale
    d_K: Matrix = []
    for j in range(seq_len):
        d_K.append(vector.scale(d_scores[j] / scale, tc.query))

    return d_K, d_V


def _nlm_backward(
    model: CTM, 
    tc: TickCache, 
    d_z_new: Vector,
    d_W1_acc: list[Matrix], 
    d_b1_acc: list[Vector],
    d_W2_acc: list[Matrix], 
    d_b2_acc: list[Vector],
) -> Vector:
    """Backward through all d_model NLM neurons.

    Each neuron's forward pass was an independent two-layer MLP:
      hist  = last M pre-activations for this neuron
      pre_h = W1[d] @ hist + b1[d]
      h     = SiLU(pre_h)
      out   = W2[d] @ h + b2[d]
      z_new[d] = out[0]

    We reverse each neuron independently, accumulating weight/bias gradients
    into per-neuron accumulators. The returned d_pre_act vector carries
    gradient back to the most recent pre-activation (from the synapse),
    connecting the NLM backward pass to the synapse backward pass.

    Note: we only propagate gradient to the most recent history entry (the
    current tick's pre-activation). Full BPTT through the M-length history
    window would be more precise but requires substantially more bookkeeping
    for a modest gain — the synapse already receives gradient directly.

    model:      CTM model (for dimensions and NLM weights)
    tc:         this tick's cached NLM intermediates (inputs, hidden, pre-hidden)
    d_z_new:    gradient w.r.t. the new post-activation state (d_model,)
    d_W1_acc:   per-neuron running gradient accumulators for W1 (mutated)
    d_b1_acc:   per-neuron running gradient accumulators for b1 (mutated)
    d_W2_acc:   per-neuron running gradient accumulators for W2 (mutated)
    d_b2_acc:   per-neuron running gradient accumulators for b2 (mutated)

    Returns gradient w.r.t. pre-activations (d_model,) — one value per neuron.
    """
    d_pre_act = vector.zeros(model.d_model)

    for d in range(model.d_model):
        # Forward was:
        # pre_h = W1[d] @ hist + b1[d]
        # h = silu(pre_h)
        # out = W2[d] @ h + b2[d]
        # z_new[d] = out[0]

        d_out = [d_z_new[d]]  # gradient of the single output

        # Backward through output layer: out = W2 @ h + b2
        d_b2_acc[d] = vector.add(d_b2_acc[d], d_out)
        # d_W2 += outer(d_out, h)
        for row in range(1):
            for col in range(model.d_hidden_nlm):
                d_W2_acc[d][row][col] += d_out[row] * tc.nlm_h[d][col]
        # d_h = W2^T @ d_out
        d_h = matrix.mat_vec(matrix.transpose(model.nlm.W2[d]), d_out)

        # Backward through SiLU
        d_pre_h = [d_h[dim] * silu_derivative(tc.nlm_pre_h[d][dim])
                    for dim in range(model.d_hidden_nlm)]

        # Backward through first layer: pre_h = W1 @ hist + b1
        d_b1_acc[d] = vector.add(d_b1_acc[d], d_pre_h)
        hist = tc.nlm_inputs[d]
        for row in range(model.d_hidden_nlm):
            for col in range(model.M):
                d_W1_acc[d][row][col] += d_pre_h[row] * hist[col]

        # d_hist = W1^T @ d_pre_h (gradient flowing to pre-activation history)
        d_hist = matrix.mat_vec(matrix.transpose(model.nlm.W1[d]), d_pre_h)
        # The most recent entry in hist corresponds to the current pre-activation
        # We accumulate gradient to the most recent pre-activation for simplicity
        # (full BPTT through the history would require more bookkeeping)
        d_pre_act[d] = d_hist[model.M - 1] if model.M > 0 else 0.0

    return d_pre_act


def _synapse_backward(
    syn: SynapseModel, 
    tc: TickCache, 
    d_a: Vector,
    d_W1_acc: Matrix, 
    d_b1_acc: Vector,
    d_W2_acc: Matrix, 
    d_b2_acc: Vector,
) -> Vector:
    """Backward through the synapse MLP (two-layer with SiLU).

    The forward pass was:
      input = concat(z, o)
      pre_h = W1 @ input + b1
      h     = SiLU(pre_h)
      a     = W2 @ h + b2

    We reverse each step, accumulating weight/bias gradients into the
    running accumulators (shared across all T ticks — the synapse is the
    same MLP every tick). The returned gradient w.r.t. the input is split
    by the caller into d_z (flows back through the tick chain) and d_o
    (flows back through cross-attention).

    syn:        synapse weights (W1, b1, W2, b2)
    tc:         this tick's cached intermediates (syn_input, syn_h, syn_pre_h)
    d_a:        gradient w.r.t. pre-activations, arriving from the NLMs
    d_W1_acc:   running gradient accumulator for W1 (mutated in place)
    d_b1_acc:   running gradient accumulator for b1 (mutated in place)
    d_W2_acc:   running gradient accumulator for W2 (mutated in place)
    d_b2_acc:   running gradient accumulator for b2 (mutated in place)

    Returns gradient w.r.t. concat(z, o) — same length as syn_input.
    """
    d_model_out = len(d_a)

    # --- Output layer: a = W2 @ h + b2 ---
    # d_b2 = d_a (bias gradient is just the upstream gradient)
    d_b2_acc[:] = vector.add(d_b2_acc, d_a)
    # d_W2[i][j] = d_a[i] * h[j] (outer product of gradient and cached hidden)
    for row in range(d_model_out):
        for col in range(len(tc.syn_h)):
            d_W2_acc[row][col] += d_a[row] * tc.syn_h[col]
    # Propagate gradient to hidden layer: d_h = W2^T @ d_a
    d_h = matrix.mat_vec(matrix.transpose(syn.W2), d_a)

    # --- SiLU activation: h = SiLU(pre_h) ---
    # Chain rule: multiply by SiLU derivative at the cached pre-activation
    d_pre_h = [d_h[dim] * silu_derivative(tc.syn_pre_h[dim])
                for dim in range(len(tc.syn_h))]

    # --- First layer: pre_h = W1 @ input + b1 ---
    # Same pattern: bias grad = upstream, weight grad = outer product
    d_b1_acc[:] = vector.add(d_b1_acc, d_pre_h)
    for row in range(len(d_pre_h)):
        for col in range(len(tc.syn_input)):
            d_W1_acc[row][col] += d_pre_h[row] * tc.syn_input[col]
    # Propagate to input: d_input = W1^T @ d_pre_h
    d_input = matrix.mat_vec(matrix.transpose(syn.W1), d_pre_h)

    return d_input


# ---------------------------------------------------------------------------
# SGD update
# ---------------------------------------------------------------------------

def ctm_sgd_update(model: CTM, grads: dict, lr: float) -> None:
    """Update all CTM parameters with vanilla SGD: param -= lr * grad.

    Included for comparison with AdamW, but in practice SGD plateaus around
    loss ~0.6 on parity. The problem is that one learning rate must serve
    every parameter — decay rates (scalars in [0, 15]), NLM weights (d_model
    private nets), synapse weights (shared MLP), and attention projections
    all have very different gradient scales. A step size that moves the
    synapse weights enough will overshoot the decay rates, and vice versa.
    AdamW solves this by maintaining per-element adaptive step sizes.
    """
    # --- Shared parameters ---

    # Output head: S_out → class logits
    for row in range(len(model.W_out)):
        for col in range(len(model.W_out[0])):
            model.W_out[row][col] -= lr * grads["d_W_out"][row][col]
    for i in range(len(model.b_out)):
        model.b_out[i] -= lr * grads["d_b_out"][i]

    # Query projection: S_action → attention query
    for row in range(len(model.W_query)):
        for col in range(len(model.W_query[0])):
            model.W_query[row][col] -= lr * grads["d_W_query"][row][col]

    # Synapse MLP: concat(z,o) → pre-activations
    _update_matrix(model.synapse.W1, grads["d_syn_W1"], lr)
    _update_vector(model.synapse.b1, grads["d_syn_b1"], lr)
    _update_matrix(model.synapse.W2, grads["d_syn_W2"], lr)
    _update_vector(model.synapse.b2, grads["d_syn_b2"], lr)

    # --- Per-neuron parameters (d_model independent MLPs) ---
    for d in range(model.d_model):
        _update_matrix(model.nlm.W1[d], grads["d_nlm_W1"][d], lr)
        _update_vector(model.nlm.b1[d], grads["d_nlm_b1"][d], lr)
        _update_matrix(model.nlm.W2[d], grads["d_nlm_W2"][d], lr)
        _update_vector(model.nlm.b2[d], grads["d_nlm_b2"][d], lr)

    # --- Decay rates ---
    # Same [0, 15] clamp as AdamW, but SGD's fixed step size makes these
    # particularly hard to tune — small lr barely moves them, large lr
    # destabilizes everything else.
    for i in range(model.pairs_out.n_pairs):
        model.sync_out.decay_rates[i] -= lr * grads["d_decay_out"][i]
        model.sync_out.decay_rates[i] = max(0.0, min(15.0, model.sync_out.decay_rates[i]))
    for i in range(model.pairs_action.n_pairs):
        model.sync_action.decay_rates[i] -= lr * grads["d_decay_action"][i]
        model.sync_action.decay_rates[i] = max(0.0, min(15.0, model.sync_action.decay_rates[i]))

    # --- Initial state and input processing ---

    # z_init: starting point for the tick loop
    _update_vector(model.z_init, grads["d_z_init"], lr)

    # Attention key/value projections
    _update_matrix(model.W_attn_k, grads["d_W_attn_k"], lr)
    _update_matrix(model.W_attn_v, grads["d_W_attn_v"], lr)

    # KV projection: embeddings → key/value space
    _update_matrix(model.W_kv, grads["d_W_kv"], lr)
    _update_vector(model.b_kv, grads["d_b_kv"], lr)

    # Input embeddings: ±1 → learned d_embed vectors
    _update_matrix(model.W_embed, grads["d_W_embed"], lr)


def _clip_grad_norm(grads: dict, max_norm: float) -> float:
    """Clip gradients by global L2 norm to prevent exploding gradients.

    Computes the total norm across ALL gradient tensors (vectors, matrices,
    lists of matrices), then scales every gradient down if the norm exceeds
    max_norm. This keeps the overall gradient magnitude bounded without
    changing its direction.

    With AdamW, clipping is usually unnecessary (the optimizer handles scale
    naturally), so max_norm=999.0 effectively disables it.

    Returns the original (pre-clipping) norm.
    """
    sum_sq = 0.0
    for key, val in grads.items():
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], list) and len(val[0]) > 0:
                if isinstance(val[0][0], list):
                    # list[Matrix] — NLM weights (one matrix per neuron)
                    for mat in val:
                        for row in mat:
                            for x in row:
                                sum_sq += x * x
                else:
                    # Matrix
                    for row in val:
                        for x in row:
                            sum_sq += x * x
            else:
                # Vector
                for x in val:
                    sum_sq += x * x
        elif isinstance(val, (int, float)):
            sum_sq += val * val

    global_norm = scalar.power(sum_sq, 0.5)
    if global_norm > max_norm:
        scale = max_norm / global_norm
        for key, val in grads.items():
            if isinstance(val, list) and len(val) > 0:
                if isinstance(val[0], list) and len(val[0]) > 0:
                    if isinstance(val[0][0], list):
                        for mat in val:
                            for row in mat:
                                for j in range(len(row)):
                                    row[j] *= scale
                    else:
                        for row in val:
                            for j in range(len(row)):
                                row[j] *= scale
                else:
                    for j in range(len(val)):
                        val[j] *= scale
            elif isinstance(val, (int, float)):
                grads[key] = val * scale

    return global_norm


def _update_matrix(M: Matrix, dM: Matrix, lr: float) -> None:
    for row in range(len(M)):
        for col in range(len(M[0])):
            M[row][col] -= lr * dM[row][col]


def _update_vector(v: Vector, dv: Vector, lr: float) -> None:
    for i in range(len(v)):
        v[i] -= lr * dv[i]


# ---------------------------------------------------------------------------
# AdamW optimizer
# ---------------------------------------------------------------------------

def _zeros_like_vector(v: Vector) -> Vector:
    return [0.0] * len(v)


def _zeros_like_matrix(m: Matrix) -> Matrix:
    return [[0.0] * len(m[0]) for _ in range(len(m))]


def create_adamw_state(grads: dict) -> tuple[dict, dict]:
    """Create zeroed first-moment (m) and second-moment (v) dicts matching grads shape."""
    m: dict = {}
    v: dict = {}
    for key, val in grads.items():
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], list) and len(val[0]) > 0:
                if isinstance(val[0][0], list):
                    # list[Matrix] (NLMs)
                    m[key] = [_zeros_like_matrix(mat) for mat in val]
                    v[key] = [_zeros_like_matrix(mat) for mat in val]
                else:
                    # Matrix
                    m[key] = _zeros_like_matrix(val)
                    v[key] = _zeros_like_matrix(val)
            else:
                # Vector
                m[key] = _zeros_like_vector(val)
                v[key] = _zeros_like_vector(val)
    return m, v


def _adamw_update_scalar(
    param: float, 
    grad: float, 
    m: float, 
    v: float,
    lr: float, 
    beta1: float, 
    beta2: float, 
    eps: float, 
    wd: float, 
    bc1: float, 
    bc2: float,
) -> tuple[float, float, float]:
    """AdamW update for a single scalar parameter.

    AdamW maintains two running averages per parameter:
      m: the "momentum" — smoothed gradient direction (which way to go)
      v: the "velocity" — smoothed gradient magnitude (how bumpy the terrain is)

    Each step:
      1. m = β1·m + (1-β1)·grad     — blend new gradient into momentum
      2. v = β2·v + (1-β2)·grad²    — blend gradient size into velocity
      3. Bias correction: m/(1-β1^t), v/(1-β2^t)
         (early steps have small m,v because they started at 0 — correction fixes this)
      4. param -= lr · corrected_m / (sqrt(corrected_v) + ε)
         (step size adapts: large v = bumpy terrain = smaller steps)
      5. param *= (1 - lr·wd)
         (weight decay: gently shrink all weights toward zero, prevents overfitting)

    The key insight vs SGD: each parameter gets its own effective learning rate.
    Parameters with large, consistent gradients take small steps (already stable).
    Parameters with small, noisy gradients take larger steps (need more push).

    This is why AdamW unlocks CTM training — the decay rates, NLM weights, and
    synapse weights have very different gradient scales, and AdamW handles each
    appropriately.

    Returns (new_param, new_m, new_v).
    """
    m = beta1 * m + (1.0 - beta1) * grad       # update momentum (gradient direction)
    v = beta2 * v + (1.0 - beta2) * grad * grad # update velocity (gradient magnitude²)
    m_hat = m / bc1                              # bias-corrected momentum
    v_hat = v / bc2                              # bias-corrected velocity
    # Weight decay (shrink param) then adaptive gradient step
    param = param * (1.0 - lr * wd) - lr * m_hat / (scalar.power(v_hat, 0.5) + eps)
    return param, m, v


def ctm_adamw_update(
    model: CTM, 
    grads: dict, 
    m_state: dict, 
    v_state: dict,
    lr: float, 
    step: int, 
    beta1: float = 0.9, 
    beta2: float = 0.999,
    eps: float = 1e-8, 
    weight_decay: float = 0.01,
) -> None:
    """Update all CTM parameters using AdamW optimizer.

    Applies _adamw_update_scalar element-by-element to every weight and bias.
    Special handling:
      - Decay rates: no weight decay (wd=0.0) because these are architectural
        schedule parameters, not feature weights. Clamped to [0, 15] per paper.
      - All other parameters: standard AdamW with weight decay.
    """
    # Bias correction terms — early steps would otherwise underestimate m and v
    # because they're initialized to zero and haven't warmed up yet
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step

    # Helpers: apply AdamW element-by-element to a matrix or vector.
    # Each element gets its own m/v state — that's what makes Adam "adaptive".
    def update_matrix(M: Matrix, key: str) -> None:
        dM = grads[key]
        mM = m_state[key]
        vM = v_state[key]
        for row in range(len(M)):
            for col in range(len(M[0])):
                M[row][col], mM[row][col], vM[row][col] = _adamw_update_scalar(
                    M[row][col], dM[row][col], mM[row][col], vM[row][col],
                    lr, beta1, beta2, eps, weight_decay, bc1, bc2,
                )

    def update_vector(vec: Vector, key: str) -> None:
        dv = grads[key]
        mv = m_state[key]
        vv = v_state[key]
        for i in range(len(vec)):
            vec[i], mv[i], vv[i] = _adamw_update_scalar(
                vec[i], dv[i], mv[i], vv[i],
                lr, beta1, beta2, eps, weight_decay, bc1, bc2,
            )

    # --- Shared parameters (one copy, used every tick) ---

    # Output head: S_out → class logits
    update_matrix(model.W_out, "d_W_out")
    update_vector(model.b_out, "d_b_out")

    # Query projection: S_action → attention query
    update_matrix(model.W_query, "d_W_query")

    # Synapse MLP: the shared transform concat(z,o) → pre-activations
    update_matrix(model.synapse.W1, "d_syn_W1")
    update_vector(model.synapse.b1, "d_syn_b1")
    update_matrix(model.synapse.W2, "d_syn_W2")
    update_vector(model.synapse.b2, "d_syn_b2")

    # --- Per-neuron parameters (d_model independent MLPs) ---
    # This is the bulk of the parameter count. Each neuron's W1, b1, W2, b2
    # are updated independently — they don't share weights with other neurons.
    for d in range(model.d_model):
        dW1 = grads["d_nlm_W1"][d]
        mW1 = m_state["d_nlm_W1"][d]
        vW1 = v_state["d_nlm_W1"][d]
        for row in range(len(model.nlm.W1[d])):
            for col in range(len(model.nlm.W1[d][0])):
                model.nlm.W1[d][row][col], mW1[row][col], vW1[row][col] = _adamw_update_scalar(
                    model.nlm.W1[d][row][col], dW1[row][col], mW1[row][col], vW1[row][col],
                    lr, beta1, beta2, eps, weight_decay, bc1, bc2,
                )
        dB1 = grads["d_nlm_b1"][d]
        mB1 = m_state["d_nlm_b1"][d]
        vB1 = v_state["d_nlm_b1"][d]
        for i in range(len(model.nlm.b1[d])):
            model.nlm.b1[d][i], mB1[i], vB1[i] = _adamw_update_scalar(
                model.nlm.b1[d][i], dB1[i], mB1[i], vB1[i],
                lr, beta1, beta2, eps, weight_decay, bc1, bc2,
            )
        dW2 = grads["d_nlm_W2"][d]
        mW2 = m_state["d_nlm_W2"][d]
        vW2 = v_state["d_nlm_W2"][d]
        for row in range(len(model.nlm.W2[d])):
            for col in range(len(model.nlm.W2[d][0])):
                model.nlm.W2[d][row][col], mW2[row][col], vW2[row][col] = _adamw_update_scalar(
                    model.nlm.W2[d][row][col], dW2[row][col], mW2[row][col], vW2[row][col],
                    lr, beta1, beta2, eps, weight_decay, bc1, bc2,
                )
        dB2 = grads["d_nlm_b2"][d]
        mB2 = m_state["d_nlm_b2"][d]
        vB2 = v_state["d_nlm_b2"][d]
        for i in range(len(model.nlm.b2[d])):
            model.nlm.b2[d][i], mB2[i], vB2[i] = _adamw_update_scalar(
                model.nlm.b2[d][i], dB2[i], mB2[i], vB2[i],
                lr, beta1, beta2, eps, weight_decay, bc1, bc2,
            )

    # --- Decay rates (special treatment) ---
    # No weight decay (wd=0.0): these control the sync time horizon, not
    # feature detection. Shrinking them toward zero would force short memory.
    # Clamped to [0, 15]: r=0 means no decay (infinite memory), r=15 means
    # e^{-15} ≈ 3e-7 (essentially forget everything before this tick).
    for i in range(model.pairs_out.n_pairs):
        model.sync_out.decay_rates[i], m_state["d_decay_out"][i], v_state["d_decay_out"][i] = _adamw_update_scalar(
            model.sync_out.decay_rates[i], grads["d_decay_out"][i],
            m_state["d_decay_out"][i], v_state["d_decay_out"][i],
            lr, beta1, beta2, eps, 0.0, bc1, bc2,
        )
        model.sync_out.decay_rates[i] = max(0.0, min(15.0, model.sync_out.decay_rates[i]))
    for i in range(model.pairs_action.n_pairs):
        model.sync_action.decay_rates[i], m_state["d_decay_action"][i], v_state["d_decay_action"][i] = _adamw_update_scalar(
            model.sync_action.decay_rates[i], grads["d_decay_action"][i],
            m_state["d_decay_action"][i], v_state["d_decay_action"][i],
            lr, beta1, beta2, eps, 0.0, bc1, bc2,
        )
        model.sync_action.decay_rates[i] = max(0.0, min(15.0, model.sync_action.decay_rates[i]))

    # --- Initial state and input processing ---

    # z_init: the starting point for the tick loop — learned, not zero
    update_vector(model.z_init, "d_z_init")

    # Attention key/value projections (applied to embedded input)
    update_matrix(model.W_attn_k, "d_W_attn_k")
    update_matrix(model.W_attn_v, "d_W_attn_v")

    # KV projection: embeddings → key/value space
    update_matrix(model.W_kv, "d_W_kv")
    update_vector(model.b_kv, "d_b_kv")

    # Input embeddings: ±1 → learned d_embed vectors
    update_matrix(model.W_embed, "d_W_embed")


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------

def predict(model: CTM, input_seq: list[float]) -> int:
    """Predict parity at the best tick (highest certainty).

    Runs a full forward pass (T ticks), then selects the tick with
    lowest entropy (highest certainty) and returns its argmax prediction.

    Resets sync state before each call so predictions are independent
    of prior forward passes.

    Returns predicted class index (int).
    """
    # Reset sync accumulators so each prediction starts from a clean slate
    model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
    model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
    model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
    model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)
    per_tick_probs, _ = ctm_forward(model, input_seq)
    # Use the most certain tick's prediction
    num_classes = len(per_tick_probs[0])
    max_entropy = math.log(num_classes)

    best_t = 0
    best_certainty = -1.0
    for t in range(len(per_tick_probs)):
        entropy = 0.0
        for p in per_tick_probs[t]:
            if p > 1e-10:
                entropy -= p * math.log(p)
        certainty = 1.0 - entropy / max_entropy
        if certainty > best_certainty:
            best_certainty = certainty
            best_t = t

    probs = per_tick_probs[best_t]
    return probs.index(max(probs))


def train(
    model: CTM,
    data: list[list[float]],
    targets: list[list[float]],
    learning_rate: float = 0.001,
    epochs: int = 10,
    max_norm: float = 1.0,
    optimizer: str = "sgd",
    weight_decay: float = 0.01,
    train_embeddings: bool = False,
) -> list[float]:
    """Train the CTM on parity data.

    The training loop is more involved than previous lessons because of the
    CTM's internal tick structure. Each sample runs a full forward pass (T
    ticks of synapse → NLM → sync → output → attention), then BPTT unrolls
    the tick chain backward, and finally the optimizer updates ~5800 params.
    Gradients are clipped per-sample to keep the tick-chain gradients stable.

    model:          the CTM to train (modified in place)
    data:           list of ±1 input sequences, each (seq_len,)
    targets:        list of cumulative parity sequences (0.0 or 1.0 at each position)
    learning_rate:  step size — needs to be very small (e.g. 5e-5) because
                    gradients flow through T ticks
    epochs:         number of full passes over the training data
    max_norm:       gradient clipping threshold (max L2 norm per parameter group)
    optimizer:      "sgd" or "adamw"
      SGD: simple gradient descent, one learning rate for all parameters.
      AdamW: adaptive learning rates per parameter with weight decay.
      CTM works much better with AdamW because its parameter groups
      (decay rates, NLMs, synapse, attention) have very different scales.
    weight_decay:   AdamW weight decay coefficient (ignored for SGD)
    train_embeddings: whether to update W_embed, W_kv, and b_kv.
      Frozen embeddings act as implicit regularization on small datasets —
      W_attn_k/W_attn_v compensate by learning to produce useful keys/values
      from the fixed projections. Enable for larger training sets.

    Returns list of average losses per epoch.
    """
    epoch_losses: list[float] = []
    adamw_state_initialized = False
    m_state: dict = {}
    v_state: dict = {}
    step = 0

    for epoch in range(epochs):
        total_loss = 0.0
        n_samples = len(data)

        for sample_idx in range(n_samples):

            input_seq = data[sample_idx]
            target_parity = targets[sample_idx]

            # For parity, the target at each position is the cumulative parity.
            # We use the LAST position's parity as the classification target.
            target_class = int(target_parity[-1])
            target_vec = one_hot(target_class, model.num_classes)

            # Reset sync state before each sample — each input starts a fresh
            # "thinking sequence" with no memory of previous samples' dynamics
            model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
            model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
            model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
            model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)

            # Forward
            per_tick_probs, cache = ctm_forward(model, input_seq)
            loss, t1, t2, _, _ = ctm_loss(per_tick_probs, target_vec)
            total_loss += loss

            # Backward
            grads = ctm_backward(model, cache, target_vec)

            # Frozen embeddings act as implicit regularization on small
            # datasets — W_attn_k/W_attn_v compensate by learning to
            # produce useful keys/values from the fixed projections.
            # Enable train_embeddings for larger training sets.
            if not train_embeddings:
                grads["d_W_embed"] = [[0.0] * len(grads["d_W_embed"][0])
                                      for _ in range(len(grads["d_W_embed"]))]
                grads["d_W_kv"] = [[0.0] * len(grads["d_W_kv"][0])
                                   for _ in range(len(grads["d_W_kv"]))]
                grads["d_b_kv"] = [0.0] * len(grads["d_b_kv"])

            _clip_grad_norm(grads, max_norm)

            # Update
            step += 1
            if optimizer == "adamw":
                if not adamw_state_initialized:
                    m_state, v_state = create_adamw_state(grads)
                    adamw_state_initialized = True
                ctm_adamw_update(
                    model, grads, m_state, v_state,
                    learning_rate, step, weight_decay=weight_decay,
                )
            else:
                ctm_sgd_update(model, grads, learning_rate)

        avg_loss = total_loss / n_samples
        epoch_losses.append(avg_loss)
        progress_bar(epoch + 1, epochs, avg_loss)

    progress_done()
    return epoch_losses
