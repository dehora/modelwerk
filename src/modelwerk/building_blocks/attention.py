"""Level 4: Attention mechanism.

Scaled dot-product attention and multi-head attention — the core
operation of the transformer architecture. Replaces the CNN's fixed
spatial window with a learned, data-dependent receptive field.

Each position computes relevance scores against every other position,
then blends their information proportionally. Three projections
transform each input vector:

    Q (query):  what this position is looking for     (seq_len, d_k)
    K (key):    what this position offers              (seq_len, d_k)
    V (value):  the information this position carries  (seq_len, d_k)

    scores  = Q @ K^T / sqrt(d_k)                     (seq_len, seq_len)
    weights = softmax(scores)                          (seq_len, seq_len)
    output  = weights @ V                              (seq_len, d_k)

Multi-head attention runs this process with multiple sets of Q/K/V
projections in parallel, each on a d_k-dimensional slice of the
full d_model-dimensional vector. The head outputs are concatenated
and projected back to d_model dimensions.
"""

import math
from dataclasses import dataclass

from modelwerk.primitives import scalar, vector, matrix
from modelwerk.primitives.activations import softmax
from modelwerk.primitives.random import xavier_init

Vector = list[float]
Matrix = list[list[float]]


@dataclass
class MultiHeadAttentionLayer:
    """Multi-head self-attention with projection matrices."""
    W_q: Matrix  # (d_model, d_model) — all heads packed
    W_k: Matrix  # (d_model, d_model)
    W_v: Matrix  # (d_model, d_model)
    W_o: Matrix  # (d_model, d_model)
    num_heads: int
    d_k: int     # d_model // num_heads


@dataclass
class AttentionCache:
    """Intermediate values for backprop through attention."""
    inputs: Matrix          # (seq_len, d_model) — input to attention
    Q: Matrix               # (seq_len, d_model) — all heads packed
    K: Matrix               # (seq_len, d_model)
    V: Matrix               # (seq_len, d_model)
    attn_weights: list[Matrix]  # per head: (seq_len, seq_len) after softmax
    head_outputs: list[Matrix]  # per head: (seq_len, d_k)
    concat: Matrix          # (seq_len, d_model) — concatenated heads


def causal_mask(seq_len: int) -> Matrix:
    """Create a causal (lower-triangular) mask.

    Returns a (seq_len, seq_len) matrix where mask[i][j] = 0.0 if j <= i
    (allowed) and -1e9 if j > i (blocked). This prevents attending to
    future positions.
    """
    mask: Matrix = []
    for row in range(seq_len):
        vals: Vector = []
        for col in range(seq_len):
            if col <= row:
                vals.append(0.0)
            else:
                vals.append(-1e9)
        mask.append(vals)
    return mask


def scaled_dot_product_attention(
    Q: Matrix, K: Matrix, V: Matrix, mask: Matrix | None = None
) -> tuple[Matrix, Matrix]:
    """Compute scaled dot-product attention.

    Q, K, V: (seq_len, d_k) matrices for a single head.
    mask: optional (seq_len, seq_len) additive mask.

    Returns (output, attention_weights).
    output: (seq_len, d_k)
    attention_weights: (seq_len, seq_len) — the softmax probabilities.
    """
    d_k = len(Q[0])
    scale = math.sqrt(d_k)

    # Scores = Q @ K^T / sqrt(d_k)  -> (seq_len, seq_len)
    K_T = matrix.transpose(K)
    scores = matrix.mat_mat(Q, K_T)
    scores = matrix.scale(1.0 / scale, scores)

    # Apply mask
    if mask is not None:
        scores = matrix.add(scores, mask)

    # Softmax row-wise
    weights: Matrix = [softmax(row) for row in scores]

    # Output = weights @ V  -> (seq_len, d_k)
    output = matrix.mat_mat(weights, V)

    return output, weights


def create_multi_head_attention(rng, d_model: int, num_heads: int) -> MultiHeadAttentionLayer:
    """Create a multi-head attention layer."""
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    d_k = d_model // num_heads
    W_q = xavier_init(rng, d_model, d_model)
    W_k = xavier_init(rng, d_model, d_model)
    W_v = xavier_init(rng, d_model, d_model)
    W_o = xavier_init(rng, d_model, d_model)
    return MultiHeadAttentionLayer(
        W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o,
        num_heads=num_heads, d_k=d_k,
    )


def _split_heads(X: Matrix, num_heads: int, d_k: int) -> list[Matrix]:
    """Split (seq_len, d_model) into num_heads matrices of (seq_len, d_k)."""
    heads: list[Matrix] = []
    for h in range(num_heads):
        start = h * d_k
        head = [row[start:start + d_k] for row in X]
        heads.append(head)
    return heads


def _concat_heads(heads: list[Matrix]) -> Matrix:
    """Concatenate head outputs back into (seq_len, d_model)."""
    seq_len = len(heads[0])
    result: Matrix = []
    for pos in range(seq_len):
        row: Vector = []
        for head in heads:
            row.extend(head[pos])
        result.append(row)
    return result


def multi_head_forward(
    layer: MultiHeadAttentionLayer, X: Matrix, mask: Matrix | None = None
) -> tuple[Matrix, AttentionCache]:
    """Forward pass through multi-head self-attention.

    X: (seq_len, d_model) input.
    Returns (output, cache).
    """
    # Q, K, V: (seq_len, d_model) — project input through learned weight matrices
    W_q_T = matrix.transpose(layer.W_q)
    W_k_T = matrix.transpose(layer.W_k)
    W_v_T = matrix.transpose(layer.W_v)

    Q = matrix.mat_mat(X, W_q_T)
    K = matrix.mat_mat(X, W_k_T)
    V = matrix.mat_mat(X, W_v_T)

    # Split into heads
    Q_heads = _split_heads(Q, layer.num_heads, layer.d_k)
    K_heads = _split_heads(K, layer.num_heads, layer.d_k)
    V_heads = _split_heads(V, layer.num_heads, layer.d_k)

    # Attention per head
    head_outputs: list[Matrix] = []
    attn_weights: list[Matrix] = []
    for h in range(layer.num_heads):
        out_h, w_h = scaled_dot_product_attention(
            Q_heads[h], K_heads[h], V_heads[h], mask
        )
        head_outputs.append(out_h)
        attn_weights.append(w_h)

    # Concatenate and project
    concat = _concat_heads(head_outputs)
    W_o_T = matrix.transpose(layer.W_o)
    output = matrix.mat_mat(concat, W_o_T)

    cache = AttentionCache(
        inputs=X, Q=Q, K=K, V=V,
        attn_weights=attn_weights,
        head_outputs=head_outputs,
        concat=concat,
    )
    return output, cache


def multi_head_backward(
    layer: MultiHeadAttentionLayer, cache: AttentionCache,
    grad_out: Matrix, mask: Matrix | None = None
) -> tuple[Matrix, dict]:
    """Backward pass through multi-head self-attention.

    Returns (grad_input, param_grads) where param_grads is a dict with
    W_q_grad, W_k_grad, W_v_grad, W_o_grad.
    """
    seq_len = len(grad_out)
    d_model = len(grad_out[0])

    # grad_out -> through W_o projection
    # output = concat @ W_o^T, so:
    # d_concat = grad_out @ W_o   (since output = concat @ W_o^T)
    # d_W_o = grad_out^T @ concat (for W_o stored as (d_model, d_model))
    d_concat = matrix.mat_mat(grad_out, layer.W_o)
    grad_out_T = matrix.transpose(grad_out)
    W_o_grad = matrix.mat_mat(grad_out_T, cache.concat)

    # Split d_concat into heads
    d_head_outputs = _split_heads(d_concat, layer.num_heads, layer.d_k)

    # Backward through each head's attention
    d_Q_heads: list[Matrix] = []
    d_K_heads: list[Matrix] = []
    d_V_heads: list[Matrix] = []

    Q_heads = _split_heads(cache.Q, layer.num_heads, layer.d_k)
    K_heads = _split_heads(cache.K, layer.num_heads, layer.d_k)
    V_heads = _split_heads(cache.V, layer.num_heads, layer.d_k)

    for h in range(layer.num_heads):
        d_Q_h, d_K_h, d_V_h = _attention_backward(
            d_head_outputs[h],
            Q_heads[h], K_heads[h], V_heads[h],
            cache.attn_weights[h],
        )
        d_Q_heads.append(d_Q_h)
        d_K_heads.append(d_K_h)
        d_V_heads.append(d_V_h)

    # Concat head gradients back to (seq_len, d_model)
    d_Q = _concat_heads(d_Q_heads)
    d_K = _concat_heads(d_K_heads)
    d_V = _concat_heads(d_V_heads)

    # Backward through projections
    # Q = X @ W_q^T  =>  d_X_q = d_Q @ W_q,  d_W_q = d_Q^T @ X
    d_X_q = matrix.mat_mat(d_Q, layer.W_q)
    d_Q_T = matrix.transpose(d_Q)
    W_q_grad = matrix.mat_mat(d_Q_T, cache.inputs)

    d_X_k = matrix.mat_mat(d_K, layer.W_k)
    d_K_T = matrix.transpose(d_K)
    W_k_grad = matrix.mat_mat(d_K_T, cache.inputs)

    d_X_v = matrix.mat_mat(d_V, layer.W_v)
    d_V_T = matrix.transpose(d_V)
    W_v_grad = matrix.mat_mat(d_V_T, cache.inputs)

    # Sum gradients from Q, K, V paths
    d_input = matrix.add(matrix.add(d_X_q, d_X_k), d_X_v)

    param_grads = {
        "W_q_grad": W_q_grad,
        "W_k_grad": W_k_grad,
        "W_v_grad": W_v_grad,
        "W_o_grad": W_o_grad,
    }
    return d_input, param_grads


def _attention_backward(
    d_output: Matrix, Q: Matrix, K: Matrix, V: Matrix,
    weights: Matrix,
) -> tuple[Matrix, Matrix, Matrix]:
    """Backward through scaled dot-product attention for one head.

    d_output: (seq_len, d_k) gradient of attention output.
    Returns (d_Q, d_K, d_V).
    """
    d_k = len(Q[0])
    scale = math.sqrt(d_k)
    seq_len = len(Q)

    # --- Step 1: Backward through weighted sum (output = weights @ V) ---
    # d_weights = d_output @ V^T    (how much each weight needs to change)
    # d_V = weights^T @ d_output    (how much each value vector needs to change)
    V_T = matrix.transpose(V)
    d_weights = matrix.mat_mat(d_output, V_T)
    weights_T = matrix.transpose(weights)
    d_V = matrix.mat_mat(weights_T, d_output)

    # --- Step 2: Backward through softmax ---
    # Unlike sigmoid or relu, softmax couples all outputs — changing one logit
    # changes ALL probabilities (they must sum to 1). So the backward pass uses
    # a Jacobian matrix rather than an element-wise derivative:
    #   J_ij = w_i * (delta_ij - w_j)
    # For each row: ds_j = w_j * (dw_j - dot(dw, w))
    d_scores: Matrix = []
    for row in range(seq_len):
        dot_prod = 0.0
        for idx in range(seq_len):
            dot_prod = scalar.add(dot_prod,
                                  scalar.multiply(d_weights[row][idx], weights[row][idx]))
        vals: Vector = []
        for col in range(seq_len):
            ds = scalar.multiply(weights[row][col],
                                 scalar.subtract(d_weights[row][col], dot_prod))
            vals.append(ds)
        d_scores.append(vals)

    # --- Step 3: Backward through scaling (scores = raw_scores / sqrt(d_k)) ---
    d_scores = matrix.scale(1.0 / scale, d_scores)

    # --- Step 4: Backward through score computation (raw_scores = Q @ K^T) ---
    # d_Q = d_scores @ K, d_K = d_scores^T @ Q
    d_Q = matrix.mat_mat(d_scores, K)
    d_scores_T = matrix.transpose(d_scores)
    d_K = matrix.mat_mat(d_scores_T, Q)

    return d_Q, d_K, d_V
