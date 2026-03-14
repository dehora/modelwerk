"""Lesson 2: The Multi-Layer Perceptron (Rumelhart, Hinton & Williams, 1986).

In 1986, the backpropagation algorithm showed that networks with hidden
layers could learn complex, non-linear patterns — overcoming the
perceptron's fundamental limitation.

Run: uv run python lessons/02_mlp.py
"""

import os

from modelwerk.primitives.random import create_rng
from modelwerk.primitives.activations import sigmoid
from modelwerk.primitives.losses import mse
from modelwerk.models.mlp import create_mlp, predict, train
from modelwerk.data.generators import xor_gate, circles
from modelwerk.viz.boundaries import plot_decision_boundary_2d, ascii_decision_boundary_2d
from modelwerk.viz.plots import plot_loss_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_mlp_on_xor():
    """
    Train an MLP on XOR, the problem the perceptron couldn't solve.
    """

    print(f"\n{'='*60}")
    print("  PART 1: SOLVING XOR")
    print(f"{'='*60}")

    print("""
The perceptron failed on XOR because no single line can separate
(0,0)/(1,1) from (0,1)/(1,0). An MLP adds a hidden layer between
input and output. Here's a network with two hidden neurons:

     Input            Hidden         Output

     x₁  o ─ w₁₁ ─┐
                  ├─► o h₁ ─ v₁  ─┐
     x₂  o ─ w₂₁ ─┘    + b₁       │
                                  ├─► o y
     x₁  o ─ w₁₂ ─┐               │    + b₃
                  ├─► o h₂ ─ v₂  ─┘
     x₂  o ─ w₂₂ ─┘    + b₂

Each hidden neuron receives both inputs through its own weights,
adds a bias, and passes the result through sigmoid (written σ). The output
neuron combines the hidden outputs the same way.

For the input → hidden connections:
          
  w₁₁: weight from input x₁ → hidden neuron h₁
  w₂₁: weight from input x₂ → hidden neuron h₁
  w₁₂: weight from input x₁ → hidden neuron h₂
  w₂₂: weight from input x₂ → hidden neuron h₂
          
You can read the indices as:
          
  w₁₁ = weight from input 1 → hidden neuron 1
          
So the hidden neurons compute weighted sums like:
          
  h₁ = σ(w₁₁ * x₁ + w₂₁ * x₂ + b₁)
  h₂ = σ(w₁₂ * x₁ + w₂₂ * x₂ + b₂)
          
Then the hidden layer connects to the output using another set of weights:
          
  v₁: weight from h₁ → y
  v₂: weight from h₂ → y

So the output computes:

  y = σ(v₁ * h₁ + v₂ * h₂ + b₃)

The key idea is that every arrow in the diagram gets its own weight.                    

In theory two hidden neurons suffice for XOR, but in practice a
few more gives gradient descent an easier landscape to navigate.
We'll use four. The hidden neurons learn features like "at least
one input is on" and "both inputs are on." The output neuron
combines them: "at least one, but not both."

Our diagram shows a 2-2-1 network (9 weights) and we'll train a
2-4-1 (17 weights). Real networks in the late 1980s were bigger
but still modest by modern standards. Typical MLPs had one or two
hidden layers with tens to a few hundred neurons — maybe 10,000
to 100,000 weights total. NETtalk (1987), which learned to
pronounce English text, used a 203-80-26 architecture with about
18,000 weights. These networks ran on workstations and could
take hours or days to train.
""")

    data, labels = xor_gate()
    # The MLP outputs a vector (one value per output neuron), so targets must
    # be lists too — unlike lesson 1 where the perceptron used scalar labels.
    targets = [[y] for y in labels]

    print("  Data points (same as lesson 1):")
    for x, y in zip(data, labels):
        print(f"    {x} → {int(y)}")

    # Create network: 2 inputs → 4 hidden → 1 output
    rng = create_rng(42)
    net = create_mlp(rng, [2, 4, 1])

    # Show initial (random) predictions
    print("\n  Before training (random weights):")
    for x, y in zip(data, labels):
        out = predict(net, x)
        # out is a list (one element per output neuron); out[0] is our single output
        print(f"    {x} → {out[0]:.3f}  (expected {int(y)})")

    # Train — note the higher learning rate vs lesson 1 (lr=0.1). MLPs with
    # sigmoid activations need larger steps because sigmoid squashes gradients
    # into a narrow range, making each individual update small.
    learning_rate = 2.0
    epochs = 3000
    print(f"\n  Training: lr={learning_rate}, epochs={epochs}")
    losses = train(net, data, targets, learning_rate=learning_rate, epochs=epochs)

    # Show results
    print(f"\n  After training:")
    print(f"    Final loss: {losses[-1]:.6f}")
    all_correct = True
    for x, y in zip(data, labels):
        out = predict(net, x)
        pred = 1 if out[0] > 0.5 else 0
        mark = "✓" if pred == int(y) else "✗"
        if pred != int(y):
            all_correct = False
        print(f"    {x} → {out[0]:.3f} → {pred}  (expected {int(y)}) {mark}")

    print(f"\n  {'Solved!' if all_correct else 'Did not converge.'}")

    # Show what the hidden layer learned
    _print_hidden_representations(net, data, labels)

    # ASCII boundary
    predict_fn = lambda x, n=net: 1 if predict(n, x)[0] > 0.5 else 0
    print()
    print(ascii_decision_boundary_2d(predict_fn, data, labels, title="XOR (MLP)"))

    # Show learned weights
    print("\n  Learned weights:")
    for i, layer in enumerate(net.layers):
        name = "hidden" if i == 0 else "output"
        weights = [[round(w, 3) for w in row] for row in layer.weights]
        biases = [round(b, 3) for b in layer.biases]
        print(f"    {name}: W={weights}, b={biases}")

    return net, data, labels, losses


def _print_hidden_representations(net, data, labels):
    """Show what the hidden layer does to the inputs.

    This is the key insight: the hidden layer transforms the inputs
    into a space where they ARE linearly separable.
    """
    print("\n  What the hidden layer sees:")
    print("  The hidden layer transforms each input into a new representation.")
    print("  In this new space, XOR becomes linearly separable:\n")

    from modelwerk.building_blocks.dense import dense_forward
    hidden_layer = net.layers[0]
    hidden_activation = net.activation_fns[0]

    for x, y in zip(data, labels):
        h, _ = dense_forward(hidden_layer, x, hidden_activation)
        h_str = [f"{v:.3f}" for v in h]
        print(f"    {x} → [{', '.join(h_str)}]  (class {int(y)})")

    print("\n  Notice that same-class points cluster to similar activation patterns,")
    print("  while different classes separate — the hidden layer has untangled them.")


def train_mlp_on_circles():
    """Train an MLP on concentric circles — a harder non-linear problem."""

    print(f"\n\n{'='*60}")
    print("  PART 2: CONCENTRIC CIRCLES")
    print(f"{'='*60}")

    print("""
XOR has only 4 points. Let's try something harder: two concentric
circles. Points near the center are class 1, points on the outer
ring are class 0. No straight line can separate inside from outside,
you need a curved boundary, which is what the hidden layer learns.
""")

    rng = create_rng(42)
    data, labels = circles(rng, n_samples=80, noise=0.05)
    targets = [[y] for y in labels]

    inner = sum(1 for y in labels if y == 1.0)
    outer = sum(1 for y in labels if y == 0.0)
    print(f"  Data: {len(data)} points ({inner} inner, {outer} outer)")

    # Larger hidden layer for a harder problem
    rng2 = create_rng(123)
    net = create_mlp(rng2, [2, 8, 1])

    learning_rate = 1.0
    epochs = 2000
    print(f"  Training: lr={learning_rate}, epochs={epochs}")
    losses = train(net, data, targets, learning_rate=learning_rate, epochs=epochs)

    # Evaluate
    correct = 0
    for x, y in zip(data, labels):
        out = predict(net, x)
        pred = 1 if out[0] > 0.5 else 0
        if pred == int(y):
            correct += 1
    accuracy = correct / len(labels)
    print(f"\n  Final loss: {losses[-1]:.6f}")
    print(f"  Accuracy:   {accuracy:.0%}")

    # ASCII boundary
    predict_fn = lambda x, n=net: 1 if predict(n, x)[0] > 0.5 else 0
    print()
    print(ascii_decision_boundary_2d(predict_fn, data, labels, title="Circles (MLP)"))

    return net, data, labels, losses


def main():
    print("=" * 60)
    print("  LESSON 2: BACKPROPAGATION AND THE MLP")
    print("  Rumelhart, Hinton & Williams, 1986")
    print("=" * 60)

    print("""
In Lesson 1 we saw the perceptron fail on XOR showing a fundamental 
limit. Minsky and Papert proved in 1969 that a single-layer perceptron 
cannot learn any function that isn't linearly separable. Their book 
"Perceptrons" was so influential that research funding for neural 
networks largely dried up.

The field went quiet for over a decade. Other approaches to AI—
expert systems, symbolic reasoning—took center stage. Neural
networks were considered a dead end.

Then in 1986, Rumelhart, Hinton, and Williams published "Learning
representations by back-propagating errors." The key ideas:

  1. Add hidden layers between input and output. These layers
     learn intermediate representations, transforming the input
     into a space where the problem becomes linearly separable.

  2. Use smooth activation functions (sigmoid instead of step).
     The step function is flat everywhere except at the threshold,
     so there's no gradient to follow. Sigmoid is smooth — you can
     ask "if I nudge this weight, how much does the output change?"

  3. Apply the chain rule layer by layer (backpropagation).
     Start from the loss, work backwards through each layer,
     and compute how much each weight contributed to the error.
     Then adjust every weight simultaneously.

The chain rule is simple: if A affects B and B affects the loss,
multiply the two effects to find how A affects the loss. That's it.
Backpropagation just applies this rule repeatedly, layer by layer:

  Forward:   a₀ → [W₁] → z₁ → [σ] → a₁ → [W₂] → z₂ → [σ] → a₂ → L
  Backward:  dL/da₂ → dL/dz₂ → dL/dW₂, dL/da₁ → dL/dz₁ → dL/dW₁

For each layer:

  δ = dL/da ⊙ σ'(z)        (⊙ = element-wise multiply: scale each neuron's blame by its slope)
  dL/dW = outer(δ, input)   (outer product: multiply every pair to get a matrix of weight gradients)
  dL/da_prev = Wᵀ @ δ       (pass the blame backwards)

Let's trace this concretely through our 2-hidden-neuron XOR network.
Same variables as the forward pass: w₁₁, w₂₁, w₁₂, w₂₂, v₁, v₂,
b₁, b₂, b₃, h₁, h₂, y.

Start at the loss. We used MSE — Mean Squared Error — which measures
how far off a prediction is by squaring the difference:
MSE = (y - target)². Squaring means bigger mistakes are penalized
more than small ones. Its derivative with respect to y is
2 * (y - target):

  dL/dy = 2 * (y - target)

This is "how far off was y?" If y = 0.8 and target = 1, the
gradient is -0.4 — the output was too low.

Step 1: Output layer δ. Multiply the loss gradient by the slope
of sigmoid at the output neuron's pre-activation:

  δ₃ = dL/dy * σ'(z₃)

σ'(z) is largest at z = 0 (where sigmoid is steepest) and shrinks
toward 0 at extremes. So neurons that are very confident (near 0
or 1) get small updates — they're already saturated. This is the
"vanishing gradient" problem: in deeper networks, these tiny slopes
multiply together and gradients can shrink to near zero. We'll
return to this in later lessons.

Step 2: Output weight gradients. How much did v₁ and v₂ contribute?
Each hidden neuron's output was multiplied by its weight, so:

  dL/dv₁ = δ₃ * h₁
  dL/dv₂ = δ₃ * h₂
  dL/db₃ = δ₃

If h₁ was large, v₁ gets more blame. If h₁ was near zero,
v₁ barely mattered.

Step 3: Pass blame to the hidden layer. Each hidden neuron
contributed to the error through its output weight:

  dL/dh₁ = v₁ * δ₃
  dL/dh₂ = v₂ * δ₃

This is the chain rule at work: v₁ tells us how much h₁
influenced the output, and δ₃ tells us how much the output
influenced the loss.

Step 4: Hidden layer δ. Same pattern as Step 1 — we scale the blame
by how responsive the neuron was (its sigmoid slope):

  δ₁ = dL/dh₁ * σ'(z₁)
  δ₂ = dL/dh₂ * σ'(z₂)

Step 5: Input weight gradients. Now we can blame the original
weights:

  dL/dw₁₁ = δ₁ * x₁       dL/dw₁₂ = δ₂ * x₁
  dL/dw₂₁ = δ₁ * x₂       dL/dw₂₂ = δ₂ * x₂
  dL/db₁  = δ₁             dL/db₂  = δ₂

Every weight gets told exactly how much it contributed to the
error, no matter how deep in the network.

This is what makes deep learning possible: every layer gets a
gradient, every weight gets an update, all from one forward pass
and one backward pass.

Let's see it solve XOR.
""")

    # Part 1: XOR
    xor_net, xor_data, xor_labels, xor_losses = train_mlp_on_xor()

    # Part 2: Circles
    circles_net, circles_data, circles_labels, circles_losses = train_mlp_on_circles()

    # Wrap-up
    print(f"\n\n{'='*60}")
    print("  WHAT CHANGED")
    print(f"{'='*60}")

    print("""
The perceptron had one neuron and one straight line. The MLP has
layers and curves. But the real breakthrough isn't the architecture —
it's the learning algorithm.

The perceptron learning rule was direct: if wrong, nudge the weights.
But it only works for one layer. There's no way to tell a hidden
neuron "you contributed to the mistake". The error signal stops
at the output.

Backpropagation solves this. By applying the chain rule layer by
layer, it assigns credit (or blame) to every weight in the network.
A weight deep in the first hidden layer gets a gradient that says
"here's how much you contributed to the final error, through every
layer between you and the output."

The 1986 paper didn't invent backpropagation (the idea appeared
independently several times in the 1960s-70s), but it demonstrated
convincingly that it worked. Networks with hidden layers can
learn useful internal representations. This revived the field and
set the stage for everything that followed.

But notice the cost. The perceptron updated each weight with a
single multiply. Backpropagation requires a full forward pass,
then a full backward pass, computing and storing intermediate
values at every layer. Our 17-weight XOR network trains in
milliseconds. NETtalk's 18,000 weights took hours on a 1987
workstation. This pattern — more capacity means more compute —
will intensify with every lesson that follows.

Next up: convolutional networks (Lesson 3), where we'll see how
to exploit the structure of images instead of treating every pixel
as an independent input.
""")

    # Generate plots
    os.makedirs("output", exist_ok=True)

    # XOR decision boundary and loss curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MLP on XOR", fontsize=14)

    predict_fn = lambda x, n=xor_net: 1 if predict(n, x)[0] > 0.5 else 0
    plot_decision_boundary_2d(predict_fn, xor_data, xor_labels,
                              title="Decision Boundary", ax=axes[0])

    axes[1].plot(xor_losses, color="#5CB8B2", linewidth=1.5)
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig("output/02_mlp_xor.png", dpi=150)
    plt.close(fig)

    # Circles decision boundary and loss curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MLP on Concentric Circles", fontsize=14)

    predict_fn = lambda x, n=circles_net: 1 if predict(n, x)[0] > 0.5 else 0
    plot_decision_boundary_2d(predict_fn, circles_data, circles_labels,
                              title="Decision Boundary", ax=axes[0])

    axes[1].plot(circles_losses, color="#5CB8B2", linewidth=1.5)
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale("log")

    plt.tight_layout()
    fig.savefig("output/02_mlp_circles.png", dpi=150)
    plt.close(fig)

    print("Plots saved to:")
    print("  - ./output/02_mlp_xor.png")
    print("  - ./output/02_mlp_circles.png")

    print(f"\n{'='*60}")
    print("  END OF LESSON 2")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
