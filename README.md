# modelwerk

Neural networks from scratch, built piece by piece from scalar operations up to complete architectures. Pure Python, no frameworks — just `math` and lists.

The project follows five landmark papers chronologically, each one building on the previous lesson's code and concepts:

| Lesson | Paper | Year | Network | Status |
|--------|-------|------|---------|--------|
| 01 | Rosenblatt, "The Perceptron" | 1958 | Single-layer perceptron | Done |
| 02 | Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors" | 1986 | Multi-layer perceptron | Done |
| 03 | LeCun et al., "Gradient-based learning applied to document recognition" | 1998 | LeNet-5 (CNN) | Planned |
| 04 | Vaswani et al., "Attention Is All You Need" | 2017 | Transformer | Planned |
| 05 | Darlow et al., "Continuous Thought Machines" | 2025 | CTM | Planned |

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

*Coming soon.* Convolutional neural networks — learning spatial features directly from pixel data, with weight sharing and pooling.

### Lesson 4: The Transformer (2017)

*Coming soon.* Self-attention as a replacement for recurrence — the architecture behind modern large language models.

### Lesson 5: The Continuous Thought Machine (2025)

*Coming soon.* Darlow et al. reintroduce neural timing as a core computational principle. The CTM gives each neuron its own temporal dynamics via neuron-level models (NLMs) with private weights, and uses neural synchronization — temporal correlations between neurons — as the latent representation. Unlike static feedforward or fixed-step recurrent networks, the CTM iterates over an internal time dimension, refining representations across "thought steps" and naturally exhibiting adaptive compute (stopping early on simple inputs, thinking longer on hard ones).

## Running

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Run a lesson
uv run python lessons/01_perceptron.py
uv run python lessons/02_mlp.py

# Run tests
uv run pytest tests/
```

Lessons print narrative text, training output, and ASCII decision boundaries to the terminal. Matplotlib plots are saved to `output/`.

## Project structure

```
src/modelwerk/
  primitives/         Scalar, vector, matrix ops, activations, losses
    scalar.py           Addition, multiplication, exp, log — the atoms
    vector.py           Dot product, element-wise ops, norms
    matrix.py           Matrix multiply, transpose, outer product
    activations.py      Step, sigmoid, tanh, ReLU, softmax + derivatives
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
tests/                Unit tests for primitives, building blocks, and models
```

## Rules

- **Python standard library only** — no numpy, torch, tensorflow, or any ML/data framework
- **matplotlib is the sole exception** — allowed for visualization only
- **Compositional layering** — each level imports only from levels below (scalar → vector → matrix → activations → neuron → layer → network → optimizer → model)
- All randomness goes through `random.py` with explicit seeds
