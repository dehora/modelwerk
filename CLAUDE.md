# modelwerk

Neural networks from scratch, built piece by piece from scalar operations up to complete architectures.

## Rules

- **Python standard library only** — no numpy, torch, tensorflow, or any ML/data framework
- **matplotlib is the sole exception** — allowed for visualization only (in `src/modelwerk/viz/`)
- **Compositional layering** — each level imports only from levels below:
  - L0: scalar ops → L1: vector/matrix ops → L2: activations/losses → L3: neuron → L4: layers → L5: network → L6: gradients/optimizers → L7: models
- Types: `list[float]` for vectors, `list[list[float]]` for matrices, dataclasses for structured objects
- All randomness goes through `src/modelwerk/primitives/random.py` with explicit seeds

## Running

- `uv run python lessons/01_perceptron.py` — run a lesson
- `uv run pytest tests/` — run tests

## Structure

- `src/modelwerk/primitives/` — scalar, vector, matrix ops, activations, losses
- `src/modelwerk/building_blocks/` — neuron, layers, network, backprop, optimizers
- `src/modelwerk/models/` — complete architectures (perceptron, mlp, lenet5, transformer)
- `src/modelwerk/data/` — dataset generation and loading
- `src/modelwerk/viz/` — matplotlib visualizations
- `lessons/` — runnable scripts, one per model, with narrative explanations
