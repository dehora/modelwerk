"""Lesson 1: The Perceptron (Rosenblatt, 1958).

In 1958, Frank Rosenblatt proposed a simple learning machine that could
classify inputs into two categories. This lesson builds one from scratch
and explores what it can — and can't — learn.

Run: uv run python lessons/01_perceptron.py
"""

import os

from modelwerk.primitives.random import create_rng
from modelwerk.building_blocks.neuron import Neuron
from modelwerk.models.perceptron import create_perceptron, predict, train
from modelwerk.data.generators import and_gate, or_gate, nand_gate, xor_gate
from modelwerk.viz.boundaries import plot_decision_boundary_2d, ascii_decision_boundary_2d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_perceptron_on_logic_gates():
    """
    Train a single perceptron on each logic gate and display results.

    Logic gates (AND, OR, NAND, XOR) are functions that take two binary
    inputs and produce a binary output — the simplest possible
    classification problems. They're ideal for testing whether a
    perceptron can learn a pattern.

    The sequence:
      1. create one perceptron with random weights
      2. for each gate:
         a. load the data (truth table for the gate)
         b. train: pass data through the perceptron, adjust weights on errors
         c. evaluate: check predictions against expected labels
         d. visualize: show the decision boundary
    """
    results = {}

    # Logic gates take two inputs and produce one output:
    #   AND  — true only when both inputs are true
    #   OR   — true when at least one input is true
    #   NAND — true unless both inputs are true (NOT AND)
    #   XOR  — true when inputs differ (exclusive or)
    logic_gates = [
        ("AND",  and_gate),
        ("OR",   or_gate),
        ("NAND", nand_gate),
        ("XOR",  xor_gate),
    ]

    # 1. Create the network
    # Make one perceptron and train it on each gate in sequence.
    # Weights carry over — what it learned on AND affects how it starts OR.
    # We start with random weights and let the network 'learn' them during training
    weight_init = create_rng(42)
    num_inputs = 2
    perceptron = create_perceptron(weight_init, num_inputs)

    for name, logic_gate in logic_gates:
        
        # 2. Load data: 
        #    The inputs and the correct answers.
        #    "labels" are the expected outputs — the right answer for
        #    each input. The perceptron's job is to learn to produce
        #    these labels from the data.
        data, labels = logic_gate()
        _print_data(name, data, labels)
        
        # 3. Train: 
        #    Adjust weights repeatedly until the perceptron
        #    learns the pattern (or fails to).
        #    - learning_rate: how much to adjust weights after each
        #      mistake. Too high and it overshoots, too low and it
        #      barely moves.
        #    - epochs: how many times to loop through all the data.
        #      Each pass gives the perceptron another chance to correct
        #      its mistakes.
        learning_rate = 0.1
        epochs = 20
        errors = train(perceptron, data, labels, learning_rate, epochs)
         
        # 4. Evaluate:
        #    Feed every input back through the trained perceptron and compare
        #    its predictions to the expected labels. This tells us whether
        #    training worked — did adjusting the weights teach it the pattern?
        predictions = []
        for input in data:
            prediction = predict(perceptron, input)
            predictions.append(prediction)
        
        correct = 0
        for prediction, label in zip(predictions, labels):
            if prediction == int(label):
                correct += 1

        accuracy = correct / len(labels)

        weights = []
        for weight in perceptron.weights:
            weights.append(round(weight, 3))

        bias = round(perceptron.bias, 3)

        # 5. Print and Visualize
        _print_results(predictions, accuracy, weights, bias, errors, data, labels, epochs)
        _print_boundary(name, perceptron, data, labels)

        # Snapshot the perceptron's current state — weights change
        # with each gate, so we need a copy for plotting later.
        snapshot = Neuron(weights=list(perceptron.weights), bias=perceptron.bias)
        results[name] = (snapshot, data, labels)

    return results


# --- Display helpers (no training logic) ---

_NOTICES = {
    "AND": [
        "The line cuts off just the top-right corner",
        "Only when both inputs are 1 does the output land in the ⣿ region",
    ],
    "OR": [
        "Weights carried over from AND — the perceptron didn't start from scratch",
        "The line cuts off just the bottom-left corner",
        "Only when both inputs are 0 does the output land in the · region",
    ],
    "NAND": [
        "Weights carried over from OR — the perceptron had to unlearn OR to learn NAND",
        "This is AND's mirror image — the ⣿ and · regions are flipped",
        "NAND is just AND with the decision reversed",
    ],
    "XOR": [
        "T and F points are on the same side of the line",
        "No single straight line can separate (0,0)/(1,1) from (0,1)/(1,0)",
        "This is what 'not linearly separable' looks like",
    ],
}


def _print_data(name, data, labels):
    print(f"\n{'='*60}")
    print(f"  Training perceptron on {name}")
    print(f"{'='*60}")
    print(f"\n  Data points:")
    for x, y in zip(data, labels):
        print(f"    {x} → {int(y)}")


def _print_results(predictions, accuracy, weights, bias, errors, data, labels, epochs):
    final_error = errors[-1]
    converged = final_error == 0.0

    print(f"\n  After {epochs} epochs (passes through all data points):")
    print(f"    Accuracy:    {accuracy:.0%}")
    print(f"    Final error: {final_error}  {'(converged)' if converged else '(did not converge)'}")
    print(f"    Weights:     {weights}")
    print(f"    Bias:        {bias}")

    print(f"    Predictions:")
    for x, pred, y in zip(data, predictions, labels):
        mark = "✓" if pred == int(y) else "✗"
        print(f"      {x} → {pred} (expected {int(y)}) {mark}")


def _print_boundary(name, neuron, data, labels):
    predict_fn = lambda x, n=neuron: predict(n, x)
    print()
    print(ascii_decision_boundary_2d(predict_fn, data, labels, title=name))
    if name in _NOTICES:
        print()
        for point in _NOTICES[name]:
            print(f"  - {point}")


# --- Lesson ---

def main():
    print("=" * 60)
    print("  LESSON 1: THE PERCEPTRON")
    print("  Rosenblatt, 1958")
    print("=" * 60)

    print("""
In 1958, Frank Rosenblatt at the Cornell Aeronautical Laboratory
built the Mark I Perceptron — a machine that could learn to classify
simple patterns. A perceptron is a single neuron that looks like this:

      x₁ * w₁ ──▶ o₁ ──┐
                        ├──▶ sum(o₁, o₂) + bias ──▶ step() ──▶ prediction
      x₂ * w₂ ──▶ o₂ ──┘

Each input (x₁, x₂) is multiplied by its weight (w₁, w₂) to
produce a weighted input (o₁, o₂). The weighted inputs are summed,
the bias is added, and the total is passed to a step function.

Weights are importance scores — they control how much each input
matters for the decision. A large positive weight means that input
pushes strongly toward "true"; a negative weight pushes toward "false."

If the weighted sum exceeds the threshold (zero, by default), the
step function outputs 1 ("true"); otherwise 0 ("false"). The bias
is a nudge — it shifts the threshold up or down, like setting the
bar higher or lower for what counts as "true."

The learning rule follows the same spirit:
  error  = target - prediction
  wᵢ     = wᵢ + learning_rate × error × xᵢ
  bias   = bias + learning_rate × error

When it gets the wrong answer, adjust the weights: make the ones
that contributed to the mistake smaller, and the ones that would
have helped bigger. The learning rate controls how big each
adjustment is — too big and it overshoots, too small and it
takes forever.

Here's one training step to make this concrete. Say we're learning
AND with weights [0.3, -0.1] and bias 0. Input [1, 1], expected 1:
  weighted sum = 0.3*1 + (-0.1)*1 + 0 = 0.2
  step(0.2) = 1 → correct, no update needed.
Now input [0, 1], expected 0:
  weighted sum = 0.3*0 + (-0.1)*1 + 0 = -0.1
  step(-0.1) = 0 → correct again.
But input [1, 0], expected 0:
  weighted sum = 0.3*1 + (-0.1)*0 + 0 = 0.3
  step(0.3) = 1 → wrong! error = 0 - 1 = -1
  new weights: [0.3 + 0.1*(-1)*1, -0.1 + 0.1*(-1)*0] = [0.2, -0.1]
  new bias:    0 + 0.1*(-1) = -0.1
The perceptron nudged w₁ down — it learned that x₁ alone isn't
enough to say "true."

Let's see what this can learn.
""")

    results = train_perceptron_on_logic_gates()

    # XOR discussion
    print(f"\n{'='*60}")
    print("  WHY XOR FAILS")
    print(f"{'='*60}")
    print("""
The perceptron draws a single straight line to separate classes.
AND, OR, and NAND are all linearly separable — you can draw a line
with class 0 on one side and class 1 on the other.

XOR is different. The points (0,0) and (1,1) are class 0, while
(0,1) and (1,0) are class 1. No single line can separate them.
Training longer doesn't help — even after 1000 epochs, XOR stays
at 25% accuracy. The problem isn't patience, it's architecture.

This limitation was famously highlighted by Minsky & Papert in 1969,
contributing to the first "AI winter." The solution — multiple layers
of neurons (the MLP) — would come later, which we'll see in Lesson 2.
""")

    # Plot decision boundaries in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Perceptron Decision Boundaries", fontsize=16)

    for ax, (name, (p, data, labels)) in zip(axes.flat, results.items()):
        predict_fn = lambda x, neuron=p: predict(neuron, x)
        plot_decision_boundary_2d(predict_fn, data, labels, title=name, ax=ax)

    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    fig.savefig("output/01_perceptron_boundaries.png", dpi=150)
    plt.close(fig)

    print("You can see the results graphically as well in:")
    print("  - ./output/01_perceptron_boundaries.png")

    print(f"\n{'='*60}")
    print("  END OF LESSON 1")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
