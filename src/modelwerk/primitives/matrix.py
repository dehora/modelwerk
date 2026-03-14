"""Level 1: Matrix operations.

Operations on lists of lists of floats — matrix-vector multiply,
transpose, outer product. Built from vector operations.
"""

from modelwerk.primitives import scalar, vector

Matrix = list[list[float]]
Vector = list[float]


def mat_vec(M: Matrix, v: Vector) -> Vector:
    return [vector.dot(row, v) for row in M]


def mat_mat(A: Matrix, B: Matrix) -> Matrix:
    B_T = transpose(B)
    return [[vector.dot(a_row, b_col) for b_col in B_T] for a_row in A]


def transpose(M: Matrix) -> Matrix:
    if not M:
        return []
    rows, cols = len(M), len(M[0])
    return [[M[r][c] for r in range(rows)] for c in range(cols)]


def outer(a: Vector, b: Vector) -> Matrix:
    return [[scalar.multiply(ai, bj) for bj in b] for ai in a]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def add(A: Matrix, B: Matrix) -> Matrix:
    return [vector.add(A[r], B[r]) for r in range(len(A))]


def scale(c: float, M: Matrix) -> Matrix:
    return [vector.scale(c, row) for row in M]


def flatten(M: Matrix) -> Vector:
    result: Vector = []
    for row in M:
        result.extend(row)
    return result


def reshape(v: Vector, rows: int, cols: int) -> Matrix:
    return [v[r * cols:(r + 1) * cols] for r in range(rows)]
