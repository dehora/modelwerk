"""Transformer (Vaswani et al., 2017).

Decoder-only, character-level language model with causal masking.
Self-attention replaces fixed convolution windows with learned,
data-dependent receptive fields.

Architecture:
    Token embedding + sinusoidal positional encoding
    1 transformer block:
        Pre-norm → multi-head self-attention → residual
        Pre-norm → feed-forward (d_model → 4*d_model → d_model) → residual
    Layer norm → dense → softmax

Some other simplifications we made compared to the original "Attention Is All You Need" paper:

  - 1 block instead of 6: keeps training fast in pure Python.
  - Pre-norm instead of post-norm: reflects modern practice since the paper was published.
  - No learnable layer norm params: gamma/beta omitted.
  - Stochastic Gradient Descent (SGD) instead of Adam: to match up with our prior lessons.
  - Decoder-only: this closer to modern GPT-style, instead of encoder-decoder.
  - Character-level tokens: instead of the usual subword tokenization.
"""

from dataclasses import dataclass

from modelwerk.primitives import scalar, vector, matrix
from modelwerk.primitives.activations import (
    relu, relu_derivative, identity, softmax, layer_norm, layer_norm_backward,
)
from modelwerk.primitives.losses import cross_entropy
from modelwerk.primitives.progress import progress_bar, progress_done
from modelwerk.building_blocks.dense import DenseLayer, DenseCache, create_dense, dense_forward
from modelwerk.building_blocks.embedding import (
    TokenEmbedding, create_token_embedding, embed_tokens,
    sinusoidal_positional_encoding,
)
from modelwerk.building_blocks.attention import (
    MultiHeadAttentionLayer, AttentionCache,
    create_multi_head_attention, multi_head_forward, multi_head_backward,
    causal_mask,
)
from modelwerk.data.utils import one_hot

Vector = list[float]
Matrix = list[list[float]]


@dataclass
class TransformerLM:
    """Decoder-only transformer language model."""
    token_emb: TokenEmbedding
    attention: MultiHeadAttentionLayer
    ff_expand: DenseLayer      # d_model → ff_dim
    ff_contract: DenseLayer    # ff_dim → d_model
    output_proj: DenseLayer    # d_model → vocab_size
    vocab_size: int
    d_model: int
    seq_len: int


@dataclass
class TransformerCache:
    """All intermediate values needed for backprop."""
    token_ids: list[int]
    embedded: Matrix               # (seq_len, d_model) after embedding + pos enc
    # Attention sublayer
    attn_ln_input: Matrix          # (seq_len, d_model) input to attention layer norm
    attn_ln_output: Matrix         # (seq_len, d_model) normalized, feeds into attention
    attn_cache: AttentionCache
    attn_residual: Matrix          # (seq_len, d_model) after attention + residual
    # Feed-forward sublayer
    ff_ln_input: Matrix            # (seq_len, d_model) input to FF layer norm
    ff_ln_output: Matrix           # (seq_len, d_model) normalized, feeds into FF
    ff_expand_caches: list[DenseCache]   # per position
    ff_contract_caches: list[DenseCache] # per position
    ff_residual: Matrix            # (seq_len, d_model) after FF + residual
    # Output
    out_ln_input: Matrix           # (seq_len, d_model) input to final layer norm
    out_ln_output: Matrix          # (seq_len, d_model) normalized, feeds into output dense
    logits: list[Vector]           # per position: (vocab_size,)
    probs: list[Vector]            # per position: (vocab_size,)
    out_caches: list[DenseCache]   # per position


def create_transformer_lm(
    rng, vocab_size: int, d_model: int = 32, num_heads: int = 2,
    ff_dim: int = 64, seq_len: int = 32,
) -> TransformerLM:
    """Create a decoder-only transformer language model."""
    token_emb = create_token_embedding(rng, vocab_size, d_model)
    attention = create_multi_head_attention(rng, d_model, num_heads)
    ff_expand = create_dense(rng, d_model, ff_dim)
    ff_contract = create_dense(rng, ff_dim, d_model)
    output_proj = create_dense(rng, d_model, vocab_size)
    return TransformerLM(
        token_emb=token_emb, attention=attention,
        ff_expand=ff_expand, ff_contract=ff_contract, output_proj=output_proj,
        vocab_size=vocab_size, d_model=d_model, seq_len=seq_len,
    )


def transformer_forward(
    model: TransformerLM, token_ids: list[int]
) -> tuple[list[Vector], TransformerCache]:
    """Forward pass through the transformer.

    token_ids: list of integer token IDs (length = seq_len).
    Returns (probs_per_position, cache).
    """
    seq_len = len(token_ids)
    mask = causal_mask(seq_len)

    # Embedding: token IDs -> (seq_len, d_model)
    tok_emb = embed_tokens(model.token_emb, token_ids)
    pos_enc = sinusoidal_positional_encoding(seq_len, model.d_model)
    embedded: Matrix = [vector.add(tok_emb[t], pos_enc[t]) for t in range(seq_len)]

    # --- Attention sublayer (pre-norm) ---
    # Layer norm -> (seq_len, d_model)
    attn_ln_input = [list(row) for row in embedded]
    attn_ln_output: Matrix = [layer_norm(row) for row in embedded]
    # Multi-head attention -> (seq_len, d_model)
    attn_out, attn_cache = multi_head_forward(model.attention, attn_ln_output, mask)
    # + residual -> (seq_len, d_model)
    attn_residual: Matrix = [vector.add(embedded[t], attn_out[t]) for t in range(seq_len)]

    # --- Feed-forward sublayer (pre-norm) ---
    # Layer norm -> (seq_len, d_model)
    ff_ln_input = [list(row) for row in attn_residual]
    ff_ln_output: Matrix = [layer_norm(row) for row in attn_residual]
    # FF expand: dense + relu -> (seq_len, ff_dim)
    # FF contract: dense -> (seq_len, d_model)
    ff_expand_caches: list[DenseCache] = []
    ff_contract_caches: list[DenseCache] = []
    ff_outputs: Matrix = []
    for t in range(seq_len):
        h, expand_cache = dense_forward(model.ff_expand, ff_ln_output[t], relu)
        out, contract_cache = dense_forward(model.ff_contract, h, identity)
        ff_expand_caches.append(expand_cache)
        ff_contract_caches.append(contract_cache)
        ff_outputs.append(out)
    # + residual -> (seq_len, d_model)
    ff_residual: Matrix = [vector.add(attn_residual[t], ff_outputs[t]) for t in range(seq_len)]

    # --- Output projection ---
    # Layer norm -> (seq_len, d_model)
    out_ln_input = [list(row) for row in ff_residual]
    out_ln_output: Matrix = [layer_norm(row) for row in ff_residual]
    # Dense -> (seq_len, vocab_size), then softmax
    logits_list: list[Vector] = []
    probs_list: list[Vector] = []
    out_caches: list[DenseCache] = []
    for t in range(seq_len):
        logit, out_cache = dense_forward(model.output_proj, out_ln_output[t], identity)
        prob = softmax(logit)
        logits_list.append(logit)
        probs_list.append(prob)
        out_caches.append(out_cache)

    cache = TransformerCache(
        token_ids=token_ids,
        embedded=embedded,
        attn_ln_input=attn_ln_input,
        attn_ln_output=attn_ln_output,
        attn_cache=attn_cache,
        attn_residual=attn_residual,
        ff_ln_input=ff_ln_input,
        ff_ln_output=ff_ln_output,
        ff_expand_caches=ff_expand_caches,
        ff_contract_caches=ff_contract_caches,
        ff_residual=ff_residual,
        out_ln_input=out_ln_input,
        out_ln_output=out_ln_output,
        logits=logits_list,
        probs=probs_list,
        out_caches=out_caches,
    )
    return probs_list, cache


def transformer_backward(
    model: TransformerLM, cache: TransformerCache,
    targets: list[list[float]],
) -> dict:
    """Backward pass through the transformer.

    targets: list of one-hot vectors, one per position.
    Returns dict of all parameter gradients.

    Gradient flow (backward, right to left):

        d_probs → d_logits → output dense → d_out_ln → LN backward → d_ff_residual
                                                                          │
        d_ff_residual → FF contract → FF expand → d_ff_ln → LN backward → d_attn_residual
              │                                                                │
              └──────────────────────── + ─────────────────────────────────────┘
              (residual: gradient flows through BOTH the FF path and the skip)

        d_attn_residual → multi-head attention backward → d_attn_ln → LN backward → d_embedded
              │                                                                         │
              └─────────────────────────── + ───────────────────────────────────────────┘
              (same residual split for the attention sublayer)

        d_embedded → accumulate into embedding table gradients
    """
    seq_len = len(cache.token_ids)
    mask = causal_mask(seq_len)

    # --- Output projection backward ---
    # Combined softmax + cross-entropy: d_logits = probs - target (same as LeNet-5)
    d_out_ln: Matrix = [vector.zeros(model.d_model) for _ in range(seq_len)]

    out_weight_grads = matrix.zeros(model.vocab_size, model.d_model)
    out_bias_grads = vector.zeros(model.vocab_size)

    for t in range(seq_len):
        d_logits = vector.subtract(cache.probs[t], targets[t])

        # Output dense backward (identity activation, derivative = 1)
        d_out_weight = matrix.outer(d_logits, cache.out_caches[t].inputs)
        out_weight_grads = matrix.add(out_weight_grads, d_out_weight)
        out_bias_grads = vector.add(out_bias_grads, d_logits)

        # Propagate gradient to layer norm output
        d_out_ln[t] = matrix.mat_vec(
            matrix.transpose(model.output_proj.weights), d_logits
        )

    # Final layer norm backward
    # (distributes the gradient accounting for the mean and variance dependencies
    #  — see activations.py:layer_norm_backward for the full derivation)
    d_ff_residual: Matrix = []
    for t in range(seq_len):
        d_ff_residual.append(
            layer_norm_backward(d_out_ln[t], cache.out_ln_output[t], cache.out_ln_input[t])
        )

    # --- Feed-forward sublayer backward ---
    ff_expand_weight_grads = matrix.zeros(len(model.ff_expand.weights), len(model.ff_expand.weights[0]))
    ff_expand_bias_grads = vector.zeros(len(model.ff_expand.biases))
    ff_contract_weight_grads = matrix.zeros(len(model.ff_contract.weights), len(model.ff_contract.weights[0]))
    ff_contract_bias_grads = vector.zeros(len(model.ff_contract.biases))

    d_ff_ln: Matrix = [vector.zeros(model.d_model) for _ in range(seq_len)]

    for t in range(seq_len):
        # Residual passes gradient straight through: d_ff_output = d_ff_residual
        d_ff_out = d_ff_residual[t]

        # FF contract backward (identity activation)
        d_contract_weight = matrix.outer(d_ff_out, cache.ff_contract_caches[t].inputs)
        ff_contract_weight_grads = matrix.add(ff_contract_weight_grads, d_contract_weight)
        ff_contract_bias_grads = vector.add(ff_contract_bias_grads, d_ff_out)

        d_expand_out = matrix.mat_vec(matrix.transpose(model.ff_contract.weights), d_ff_out)

        # FF expand backward (relu activation)
        f_prime = vector.apply(relu_derivative, cache.ff_expand_caches[t].z)
        d_expand_delta = vector.elementwise(scalar.multiply, d_expand_out, f_prime)
        d_expand_weight = matrix.outer(d_expand_delta, cache.ff_expand_caches[t].inputs)
        ff_expand_weight_grads = matrix.add(ff_expand_weight_grads, d_expand_weight)
        ff_expand_bias_grads = vector.add(ff_expand_bias_grads, d_expand_delta)

        # Propagate gradient to layer norm output
        d_ff_ln[t] = matrix.mat_vec(
            matrix.transpose(model.ff_expand.weights), d_expand_delta
        )

    # FF layer norm backward
    # Residual splits gradient: attn_residual feeds both the FF layer norm and
    # the FF residual addition, so it receives the sum of both gradient paths
    d_attn_residual: Matrix = []
    for t in range(seq_len):
        d_ln = layer_norm_backward(
            d_ff_ln[t], cache.ff_ln_output[t], cache.ff_ln_input[t]
        )
        d_attn_residual.append(vector.add(d_ff_residual[t], d_ln))

    # --- Attention sublayer backward ---
    # Multi-head attention backward
    d_attn_ln, attn_param_grads = multi_head_backward(
        model.attention, cache.attn_cache, d_attn_residual, mask
    )

    # Attention layer norm backward
    # Same residual split: embedded feeds both the attention layer norm and
    # the attention residual addition
    d_embedded: Matrix = []
    for t in range(seq_len):
        d_ln = layer_norm_backward(
            d_attn_ln[t], cache.attn_ln_output[t], cache.attn_ln_input[t]
        )
        d_embedded.append(vector.add(d_attn_residual[t], d_ln))

    # --- Embedding backward ---
    # Accumulate gradients into the token embedding table
    emb_table_grads = [vector.zeros(model.d_model) for _ in range(model.vocab_size)]
    for t in range(seq_len):
        tid = cache.token_ids[t]
        emb_table_grads[tid] = vector.add(emb_table_grads[tid], d_embedded[t])

    return {
        "emb_table_grads": emb_table_grads,
        "attn_W_q_grad": attn_param_grads["W_q_grad"],
        "attn_W_k_grad": attn_param_grads["W_k_grad"],
        "attn_W_v_grad": attn_param_grads["W_v_grad"],
        "attn_W_o_grad": attn_param_grads["W_o_grad"],
        "ff_expand_weight_grads": ff_expand_weight_grads,
        "ff_expand_bias_grads": ff_expand_bias_grads,
        "ff_contract_weight_grads": ff_contract_weight_grads,
        "ff_contract_bias_grads": ff_contract_bias_grads,
        "out_weight_grads": out_weight_grads,
        "out_bias_grads": out_bias_grads,
    }


def transformer_sgd_update(model: TransformerLM, grads: dict, lr: float):
    """Update all parameters using SGD. Modifies model in place."""
    # Embedding table
    for i in range(model.vocab_size):
        model.token_emb.table[i] = vector.add(
            model.token_emb.table[i],
            vector.scale(-lr, grads["emb_table_grads"][i])
        )

    # Attention projections
    model.attention.W_q = matrix.add(model.attention.W_q, matrix.scale(-lr, grads["attn_W_q_grad"]))
    model.attention.W_k = matrix.add(model.attention.W_k, matrix.scale(-lr, grads["attn_W_k_grad"]))
    model.attention.W_v = matrix.add(model.attention.W_v, matrix.scale(-lr, grads["attn_W_v_grad"]))
    model.attention.W_o = matrix.add(model.attention.W_o, matrix.scale(-lr, grads["attn_W_o_grad"]))

    # Feed-forward
    model.ff_expand.weights = matrix.add(model.ff_expand.weights, matrix.scale(-lr, grads["ff_expand_weight_grads"]))
    model.ff_expand.biases = vector.add(model.ff_expand.biases, vector.scale(-lr, grads["ff_expand_bias_grads"]))
    model.ff_contract.weights = matrix.add(model.ff_contract.weights, matrix.scale(-lr, grads["ff_contract_weight_grads"]))
    model.ff_contract.biases = vector.add(model.ff_contract.biases, vector.scale(-lr, grads["ff_contract_bias_grads"]))

    # Output projection
    model.output_proj.weights = matrix.add(
        model.output_proj.weights, matrix.scale(-lr, grads["out_weight_grads"])
    )
    model.output_proj.biases = vector.add(
        model.output_proj.biases, vector.scale(-lr, grads["out_bias_grads"])
    )


def predict(model: TransformerLM, token_ids: list[int]) -> list[int]:
    """Return predicted next token for each position."""
    probs, _ = transformer_forward(model, token_ids)
    return [max(range(len(p)), key=lambda i: p[i]) for p in probs]


def generate(
    model: TransformerLM, prompt_ids: list[int],
    length: int, id_to_char: dict[int, str],
    temperature: float = 0.8,
) -> tuple[str, list[Matrix]]:
    """Generate text autoregressively from a prompt.

    Returns (generated_text, attention_weights_from_last_step).
    """
    generated = list(prompt_ids)
    all_attn_weights: list[list[Matrix]] = []

    for _ in range(length):
        # Use last seq_len tokens as context
        context = generated[-model.seq_len:]
        probs, cache = transformer_forward(model, context)
        all_attn_weights.append(cache.attn_cache.attn_weights)

        # Get the distribution at the last position
        last_probs = probs[-1]

        # Temperature < 1 sharpens the distribution, > 1 flattens it
        if temperature != 1.0:
            logits = [scalar.log(max(p, 1e-10)) / temperature for p in last_probs]
            last_probs = softmax(logits)

        # Argmax: pick the most probable token
        next_token = max(range(len(last_probs)), key=lambda i: last_probs[i])
        generated.append(next_token)

    text = "".join(id_to_char[tid] for tid in generated)
    # Return attention weights from the last step
    last_attn = all_attn_weights[-1] if all_attn_weights else []
    return text, last_attn


def train(
    model: TransformerLM,
    inputs: list[list[int]],
    targets: list[list[int]],
    vocab_size: int,
    learning_rate: float = 0.01,
    epochs: int = 30,
) -> list[float]:
    """Train the transformer on sequences.

    Returns loss_history (average loss per epoch).
    """
    loss_history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for seq_in, seq_tgt in zip(inputs, targets):
            # Forward
            probs, cache = transformer_forward(model, seq_in)

            # Loss: average cross-entropy over positions
            target_onehots = [one_hot(t, vocab_size) for t in seq_tgt]
            seq_loss = 0.0
            for t in range(len(seq_tgt)):
                seq_loss += cross_entropy(probs[t], target_onehots[t])
            seq_loss /= len(seq_tgt)
            epoch_loss += seq_loss

            # Backward
            grads = transformer_backward(model, cache, target_onehots)

            # Update
            transformer_sgd_update(model, grads, learning_rate)

        avg_loss = epoch_loss / len(inputs)
        loss_history.append(avg_loss)
        progress_bar(epoch + 1, epochs, avg_loss)

    progress_done()
    return loss_history
