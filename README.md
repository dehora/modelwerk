# modelwerk

Neural networks from scratch, built piece by piece from scalar operations up to complete architectures. Pure Python, no frameworks — just `math` and lists.

This is for programmers who want to understand how neural networks work, not use them. If you're looking for a PyTorch tutorial, this isn't it. If you want to know what PyTorch is doing under the hood, read on.

The project follows five landmark papers chronologically, each one building on the previous lesson's code and concepts:

| Lesson | Paper | Year | Network | Time | Status |
|--------|-------|------|---------|------|--------|
| 01 | Rosenblatt, "The Perceptron" | 1958 | Single-layer perceptron | ~5s | Done |
| 02 | Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors" | 1986 | Multi-layer perceptron | ~10s | Done |
| 03 | LeCun et al., "Gradient-based learning applied to document recognition" | 1998 | LeNet-5 (CNN) | ~3 min | Done |
| 04 | Vaswani et al., "Attention Is All You Need" | 2017 | Transformer | ~5 min | Done |
| 05 | Darlow et al., "Continuous Thought Machines" | 2025 | CTM | — | Planned |

## Why

Most ML tutorials start with `import torch`. This project starts with `1.0 + 1.0`.

Every operation is composed from scalar arithmetic up through vectors, matrices, activations, neurons, layers, networks, and finally complete models. Nothing is hidden behind a library call. The goal is to understand what the frameworks do, not how to call them.

## Lessons

### Lesson 1: The Perceptron (1958)

Rosenblatt's perceptron — a single neuron that learns a linear decision boundary. The lesson trains it on AND, OR, and NAND (which it solves), then XOR (which it can't). This failure motivates everything that follows.

**Concepts introduced:** weighted sum, step activation, perceptron learning rule, linear separability.

```
uv run python lessons/01_perceptron.py
```

### Lesson 2: The Multi-Layer Perceptron (1986)

Rumelhart, Hinton, and Williams showed that networks with hidden layers can learn non-linear patterns using backpropagation. The lesson walks through the chain rule concretely — tracing gradients through a 2-layer network weight by weight — then trains an MLP on XOR and concentric circles.

**Concepts introduced:** hidden layers, sigmoid activation, MSE (mean squared error) loss, forward/backward pass, chain rule, gradient descent, vanishing gradients, learned representations.

```
uv run python lessons/02_mlp.py
```

### Lesson 3: LeNet-5 (1998)

LeCun et al. introduced convolutional neural networks — learning spatial features directly from pixel data. The lesson trains LeNet-5 on MNIST handwritten digits, reaching ~90% accuracy with weight sharing and pooling.

**Concepts introduced:** convolution, feature maps, receptive fields, pooling, weight sharing, spatial hierarchy, softmax + cross-entropy.

```
uv run python lessons/03_lenet5.py
```

### Lesson 4: The Transformer (2017)

Vaswani et al. replaced recurrence with self-attention — each position directly looks at every other position and learns what's relevant. The lesson trains a decoder-only transformer on Shakespeare sonnets, generating character-level text.

**Concepts introduced:** self-attention (Q/K/V), causal masking, multi-head attention, positional encoding, residual connections, layer normalization, autoregressive generation.

```
uv run python lessons/04_transformer.py
```

### Lesson 5: The Continuous Thought Machine (2025)

*Coming soon.* Darlow et al. reintroduce neural timing as a core computational principle. The CTM gives each neuron its own temporal dynamics via neuron-level models (NLMs) with private weights, and uses neural synchronization — temporal correlations between neurons — as the latent representation. Unlike static feedforward or fixed-step recurrent networks, the CTM iterates over an internal time dimension, refining representations across "thought steps" and naturally exhibiting adaptive compute (stopping early on simple inputs, thinking longer on hard ones).

## Running

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Run a lesson
uv run python lessons/01_perceptron.py    # ~5 seconds
uv run python lessons/02_mlp.py           # ~10 seconds
uv run python lessons/03_lenet5.py        # ~3 minutes
uv run python lessons/04_transformer.py   # ~5 minutes

# Run tests
uv run pytest tests/
```

Lessons 1 and 2 run in seconds. Lessons 3 and 4 take a few minutes — this is pure Python doing real training, not a demo. If it seems hung, it's not — the progress bar shows where you are.

Lessons print narrative text, training output, and decision boundaries to the terminal. Matplotlib plots are saved to `output/`. Reference outputs from completed runs are in `examples/` so you can see results without running the code.

## Project structure

```
src/modelwerk/
  primitives/         Scalar, vector, matrix ops, activations, losses
    scalar.py           Addition, multiplication, exp, log — the atoms
    vector.py           Dot product, element-wise ops, norms
    matrix.py           Matrix multiply, transpose, outer product
    activations.py      Step, sigmoid, tanh, ReLU, softmax, layer norm + derivatives
    losses.py           MSE, cross-entropy + derivatives
    random.py           Seeded RNG for reproducibility

  building_blocks/    Neurons, layers, networks, backprop, optimizers
    neuron.py           Single neuron (weighted sum + activation)
    dense.py            Dense (fully-connected) layer
    network.py          Sequential network of layers
    grad.py             Gradient computation (backpropagation)
    optimizers.py       SGD with momentum
    conv.py             Convolutional layer
    pool.py             Pooling layer
    attention.py        Self-attention mechanism
    embedding.py        Token and positional embeddings

  models/             Complete architectures
    perceptron.py       Single-layer classifier
    mlp.py              Multi-layer perceptron with backprop
    lenet5.py           Convolutional network for image recognition
    transformer.py      Self-attention network

  data/               Dataset generation and loading
    generators.py       XOR, circles, spirals, moons
    mnist.py            MNIST digit loading
    text.py             Text tokenization

  viz/                Matplotlib visualizations
    boundaries.py       Decision boundary plots (2D + ASCII)
    plots.py            Loss curves, accuracy plots
    weights.py          Weight matrix heatmaps
    attention_maps.py   Attention pattern visualization

lessons/              Runnable scripts — one per paper
examples/             Reference outputs (plots, screenshots) from completed runs
tests/                Unit tests for primitives, building blocks, and models
```

## Rules

- **Python standard library only** — no numpy, torch, tensorflow, or any ML/data framework
- **matplotlib is the sole exception** — allowed for visualization only
- **Compositional layering** — each level imports only from levels below (scalar → vector → matrix → activations → neuron → layer → network → optimizer → model)
- All randomness goes through `random.py` with explicit seeds
