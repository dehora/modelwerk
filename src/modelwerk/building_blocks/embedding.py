"""Level 4: Embeddings.

Token embeddings and sinusoidal positional encodings.
Converts discrete tokens into continuous vector representations.
"""

import math
from dataclasses import dataclass

from modelwerk.primitives.random import random_vector

Vector = list[float]
Matrix = list[list[float]]


@dataclass
class TokenEmbedding:
    """Lookup table mapping token indices to dense vectors."""
    table: Matrix  # (vocab_size, d_model)


def create_token_embedding(rng, vocab_size: int, d_model: int) -> TokenEmbedding:
    """Create a token embedding table with small random values."""
    limit = 0.1
    table = [random_vector(rng, d_model, -limit, limit) for _ in range(vocab_size)]
    return TokenEmbedding(table=table)


def embed_tokens(embedding: TokenEmbedding, token_ids: list[int]) -> Matrix:
    """Look up embeddings for a sequence of token IDs.

    Returns (seq_len, d_model) matrix.
    """
    return [list(embedding.table[tid]) for tid in token_ids]


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> Matrix:
    """Compute fixed sinusoidal positional encodings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Each position gets a unique pattern of sines and cosines that
    encodes its absolute position in the sequence.
    """
    pe: Matrix = []
    for pos in range(seq_len):
        row: Vector = []
        for dim in range(d_model):
            # The formula uses pairs: even indices get sin, odd get cos.
            # dim // 2 gives the pair index (0,0,1,1,2,2,...) used in the exponent.
            angle = pos / math.pow(10000.0, (2.0 * (dim // 2)) / d_model)
            if dim % 2 == 0:
                row.append(math.sin(angle))   # PE(pos, 2i) = sin(...)
            else:
                row.append(math.cos(angle))   # PE(pos, 2i+1) = cos(...)
        pe.append(row)
    return pe


