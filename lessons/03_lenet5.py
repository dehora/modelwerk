"""Lesson 3: LeNet-5 (LeCun et al., 1998).

In 1998, Yann LeCun demonstrated that convolutional neural networks
could recognize handwritten digits by learning spatial features
directly from pixel data.

Run: uv run python lessons/03_lenet5.py
"""

import json
import os

from modelwerk.primitives.random import create_rng
from modelwerk.data.mnist import load_mnist
from modelwerk.models.lenet5 import create_lenet5, predict, train

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ascii_digit(image):
    """Render a 1x28x28 image as Unicode block art."""
    channel = image[0]  # single channel
    h = len(channel)
    w = len(channel[0])
    lines = []
    # skip every other row — terminal chars are ~2x taller than wide,
    # so 14 rows of full-width chars renders roughly square
    for r in range(0, h, 2):
        line = ""
        for c in range(w):
            val = channel[r][c]
            if val < 0.15:
                line += "·"
            elif val < 0.4:
                line += "░"
            elif val < 0.7:
                line += "▒"
            else:
                line += "█"
        lines.append("    " + line)
    return "\n".join(lines)


def _load_sample_digits():
    """Load pre-selected sample digits (2, 3, 7) from data/samples/."""
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "src", "modelwerk", "data", "samples")
    digits = []
    for d in [2, 3, 7]:
        path = os.path.join(samples_dir, f"digit_{d}.json")
        with open(path) as f:
            data = json.load(f)
        digits.append((data["image"], data["label"]))
    return digits


def _count_conv_params(layer):
    """Count parameters in a conv layer: filters * channels * kH * kW + biases."""
    f = layer.filters
    return len(f) * len(f[0]) * len(f[0][0]) * len(f[0][0][0]) + len(layer.biases)


def _count_dense_params(layer):
    """Count parameters in a dense layer: weights + biases."""
    return len(layer.weights) * len(layer.weights[0]) + len(layer.biases)


def _show_samples():
    """Display pre-selected MNIST digits as ASCII art."""
    print(f"\n  Sample digits from the dataset:\n")
    for image, label in _load_sample_digits():
        print(f"  Label: {label}")
        print(_ascii_digit(image))
        print()


def main():
    print("=" * 60)
    print("  LESSON 3: CONVOLUTIONAL NEURAL NETWORKS")
    print("  LeCun, Bottou, Bengio & Haffner, 1998")
    print("=" * 60)

    print("""
In Lesson 2 we trained an MLP on XOR and concentric circles —
problems with a handful of inputs. But what about images?

A 28x28 grayscale image has 784 pixels. An MLP treating each pixel
as an independent input needs a weight for every connection between
every pixel and every hidden neuron. For a 784-input, 120-hidden MLP,
that's 94,080 weights in just the first layer, and it learns nothing
about spatial structure. That means a '3' in the top-left corner looks 
like a completely different input to a '3' in the center, even though 
they're the same digit.

In the late 1980s and 1990s, Yann LeCun at Bell Labs was working on
a practical problem: the US Postal Service needed to automatically
read handwritten zip codes on envelopes. Processing millions of
letters a day, human sorting was expensive and slow.

LeCun's insight was that images have structure that a clever
architecture can exploit:

  1. Locality: nearby pixels matter more than distant ones.
     A small 5x5 window captures edges and curves.

  2. Translation invariance: a '3' is a '3' wherever it appears.
     The same small filter (called a 'kernel') can slide across the 
     entire image, looking for the same pattern everywhere.

  3. Hierarchy: simple features combine into complex ones.
     Edges become curves, curves become loops, loops become digits.

These ideas led to the convolutional neural network (CNN):

     Input         C1            S2          C3          S4        Dense
    1x28x28    3x24x24       3x12x12     6x8x8       6x4x4      96->32->10

    ┌─────┐    ┌─────┐       ┌───┐       ┌───┐       ┌──┐
    │     │    │ ┌─┐ │       │   │       │   │       │  │       ┌──┐  ┌──┐
    │image│─5x5│ │f│ │─pool──│   │─5x5───│   │─pool──│  │─flat──│32│──│10│
    │     │    │ └─┘ │       │   │       │   │       │  │       └──┘  └──┘
    └─────┘    └─────┘       └───┘       └───┘       └──┘
               3 filters     avg 2x2    6 filters    avg 2x2   tanh  softmax

Reading left to right:

  Input (1x28x28): a single grayscale image. The "1" is the
  channel count: grayscale has one channel, color (RGB) would
  have three. Each pixel is a float from 0 (black) to 1 (white).

  C1: Convolution (3x24x24) — three 5x5 filters slide across
  the image. Each filter is a tiny 5x5 grid of learnable weights.
  At each position, it computes a dot product with the 25 pixels
  underneath, producing one output value. A 28x28 image with a
  5x5 filter yields a 24x24 output (28-5+1=24). Three filters
  produce three such maps, each detecting a different pattern —
  one might respond to horizontal edges, another to vertical ones,
  a third to corners. This is weight sharing: the same 25 weights
  check every position.

  Here's a concrete example with a tiny 3x3 filter on a 4x4 patch:

    Image patch:         Filter (horizontal edge detector):
    0.0  0.0  0.0  0.0       -1  -1  -1
    0.0  0.9  0.8  0.0        0   0   0
    0.0  0.1  0.0  0.0        1   1   1
    0.0  0.0  0.0  0.0

    Position (0,0): dot product of top-left 3x3 with filter:
      (-1*0.0)+(-1*0.0)+(-1*0.0) + (0*0.0)+(0*0.9)+(0*0.8)
      + (1*0.0)+(1*0.1)+(1*0.0) = 0.1  (weak: no horizontal edge here)

    Position (1,0): slide down one row:
      (-1*0.0)+(-1*0.9)+(-1*0.8) + (0*0.0)+(0*0.1)+(0*0.0)
      + (1*0.0)+(1*0.0)+(1*0.0) = -1.7  (strong negative: bright-to-dark edge)

  The filter responds strongly where the image matches its pattern.
  This same 3x3 filter (9 weights) slides over every position —
  our 5x5 filters have 25 weights each, but the idea is identical.

  S2: Pooling (3x12x12): each 2x2 block is replaced by its
  average, halving the spatial dimensions. This makes the
  representation robust to small shifts — a feature detected at
  pixel (10,10) and one at (11,11) both end up in the same pooled
  cell. (Modern networks often use max pooling — taking the
  strongest activation instead of the average — but the effect
  is similar.)

  C3: Convolution (6x8x8): six 5x5 filters, but now each filter
  reads all three channels from S2. So each filter has 3x5x5=75
  weights, combining features across the previous layer's maps.
  The first layer found edges; this layer finds combinations of
  edges — curves, corners, junctions.

  S4: Pooling (6x4x4): another 2x2 average pool. The spatial
  dimensions shrink further, but the feature count grows. We
  started with 1 channel of raw pixels and now have 6 channels
  of abstract features.

  Flatten (96): reshape the 6x4x4 volume into a flat vector.
  This is the bridge from spatial processing to classification:
  the conv layers extracted features, now the dense layers
  decide what digit they represent.

  Dense (32 then 10): standard fully-connected layers, just
  like the MLP from Lesson 2. The first uses tanh activation,
  the second uses softmax to produce probabilities for each of
  the 10 digits.

Two new primitives since Lesson 2:

  Softmax: converts raw scores (logits) into probabilities.
  If the dense layer outputs [2.1, 0.3, -0.5, ...] for 10
  digits, softmax exponentiates each value and normalizes so
  they sum to 1: [0.72, 0.12, 0.05, ...]. The highest score
  becomes the highest probability. Unlike sigmoid (which treats
  each output independently), softmax makes outputs compete —
  raising one probability lowers the others.

  Cross-entropy: measures how wrong the predictions are.
  If the true label is "3", the ideal output is [0,0,0,1,0,...].
  Cross-entropy is -log(predicted probability of the true class).
  If the model says P("3") = 0.9, loss = -log(0.9) = 0.105.
  If it says P("3") = 0.01, loss = -log(0.01) = 4.6. The log
  means confident wrong answers are penalized severely. MSE
  (from Lesson 2) works for binary classification but doesn't
  have this sharp penalty for wrong confidence, which is why
  multi-class networks use cross-entropy instead.

Our simplified LeNet has 3+6 filters (the original used 6+16)
and 32 dense neurons (the original used 120+84). Fewer parameters
means faster training in pure Python while preserving the
architecture's structure. We also modernized two things: the
original's subsampling layers had trainable coefficients and
sigmoid activations (ours use plain average pooling), and its
output used Euclidean RBF units that measured distance to fixed
digit templates (ours uses softmax, which is standard practice
today).

The convolution and pooling operations live in
modelwerk/building_blocks/conv.py and pool.py — read them
to see the loops that implement sliding windows and gradient
flow. The lesson script calls model-level functions so we
can focus on the architecture and results.

Let's see it learn to read handwriting.
""")

    # --- Load MNIST ---
    print(f"{'='*60}")
    print("  PART 1: THE DATA")
    print(f"{'='*60}")

    print("""
  MNIST is the "hello world" of machine learning — 70,000
  handwritten digits (0-9) collected from census workers and
  high school students. Each image is 28x28 grayscale pixels,
  normalized to [0, 1]. We'll use a small subset here.

  Loading MNIST dataset (this takes a few seconds)...""")

    train_images, train_labels, test_images, test_labels = load_mnist(
        train_subset=1000, test_subset=200
    )

    print(f"\n  Training set: {len(train_images)} images")
    print(f"  Test set:     {len(test_images)} images")
    print(f"  Image shape:  {len(train_images[0])}x{len(train_images[0][0])}x{len(train_images[0][0][0])}")

    # Count digits
    digit_counts = [0] * 10
    for label in train_labels:
        digit_counts[label] += 1
    print(f"  Digit distribution: {digit_counts}")

    _show_samples()

    # --- Create and train model ---
    print(f"{'='*60}")
    print("  PART 2: TRAINING")
    print(f"{'='*60}")

    rng = create_rng(42)
    model = create_lenet5(rng)

    # Count parameters in each layer to show weight sharing advantage
    conv1_params = _count_conv_params(model.conv1)
    conv2_params = _count_conv_params(model.conv2)
    dense1_params = _count_dense_params(model.dense1)
    dense2_params = _count_dense_params(model.dense2)
    total_params = conv1_params + conv2_params + dense1_params + dense2_params

    print(f"""
  Model architecture:
    C1: conv 5x5, 1->3 filters    {conv1_params:>5} params
    S2: avg pool 2x2                   0 params
    C3: conv 5x5, 3->6 filters    {conv2_params:>5} params
    S4: avg pool 2x2                   0 params
    F5: dense 96->32               {dense1_params:>5} params
    F6: dense 32->10               {dense2_params:>5} params
    Total:                         {total_params:>5} params

  Compare: a fully-connected MLP with 784->120->84->10 would
  need 784*120 + 120*84 + 84*10 = 105,000+ params. The CNN
  achieves more with {total_params}: weight sharing works.
""")

    learning_rate = 0.01
    epochs = 5
    print(f"  Training: lr={learning_rate}, epochs={epochs}, samples={len(train_images)}")
    print(f"  (This takes a few minutes in pure Python — progress bar appears shortly)\n")

    loss_history, accuracy_history = train(
        model, train_images, train_labels,
        learning_rate=learning_rate, epochs=epochs,
    )

    print(f"\n  Training results:")
    for i, (loss, acc) in enumerate(zip(loss_history, accuracy_history)):
        print(f"    Epoch {i+1}: loss={loss:.4f}  accuracy={acc:.0%}")

    # --- Evaluate on test set ---
    print(f"\n{'='*60}")
    print("  PART 3: EVALUATION")
    print(f"{'='*60}")

    correct = 0
    predictions = []
    for image, label in zip(test_images, test_labels):
        pred = predict(model, image)
        predictions.append(pred)
        if pred == label:
            correct += 1

    test_accuracy = correct / len(test_labels)
    print(f"\n  Test accuracy: {correct}/{len(test_labels)} = {test_accuracy:.0%}")

    # Show some predictions — pick a mix of correct and wrong to illustrate
    # what the model gets right and where it struggles
    correct_samples = []
    wrong_samples = []
    for i in range(len(test_images)):
        if predictions[i] == test_labels[i]:
            correct_samples.append(i)
        else:
            wrong_samples.append(i)

    # Show 4 correct and up to 2 wrong (if any)
    sample_indices = correct_samples[:4] + wrong_samples[:2]

    print("\n  Sample predictions:\n")
    for i in sample_indices:
        pred = predictions[i]
        label = test_labels[i]
        mark = "correct" if pred == label else f"WRONG (expected {label})"
        print(f"  Predicted: {pred}  ({mark})")
        print(_ascii_digit(test_images[i]))
        print()

    # --- Wrap-up ---
    print(f"{'='*60}")
    print("  WHAT CHANGED")
    print(f"{'='*60}")

    print(f"""
The MLP treated every pixel as an independent input. The CNN
treats the image as a 2D grid and exploits its spatial structure.

The key innovations:

  Weight sharing: one 5x5 filter with 25 weights checks every
  position in the image. An MLP would need separate weights for
  every pixel-to-neuron connection.

  Feature hierarchy: the first conv layer learns edges and
  simple textures. The second conv layer combines those into
  curves, corners, and strokes. The dense layers combine those
  into digit classifications. (Zeiler & Fergus, 2013, produced
  striking visualizations of what each layer learns — search
  for "Visualizing and Understanding Convolutional Networks"
  to see actual filter responses.)

  Pooling: averaging 2x2 blocks makes the representation
  robust to small translations. A "7" shifted one pixel right
  still activates the same pooled features.

Backpropagation still works — it's the same chain rule from
Lesson 2 — but with a twist. In the dense layers, the backward
pass is a matrix multiply. In the conv layers, the backward pass
is another convolution: you slide the error signal across the
filter (rotated 180°) to compute how much each input pixel
contributed. The gradient for each filter weight sums up its
contribution at every position it was applied — one gradient
update teaches the filter from every location simultaneously.

Using the 4x4 / 3x3 example from earlier: the forward pass
produced a 2x2 output. Suppose the loss gradient at that
output is:

    dL/dout:
    0.0   0.5
   -0.3   0.0

How much should we adjust filter weight w[0][0] (top-left)?
It was used at every position, multiplied by the input pixel
at that position's top-left corner:

  dL/dw[0][0] = dL/dout[0,0] * input[0,0]    position (0,0)
              + dL/dout[0,1] * input[0,1]    position (0,1)
              + dL/dout[1,0] * input[1,0]    position (1,0)
              + dL/dout[1,1] * input[1,1]    position (1,1)
              = 0.0*0.0 + 0.5*0.0 + (-0.3)*0.0 + 0.0*0.0 = 0.0

This weight touched only black (0.0) pixels, so it gets no
update. But w[1][1] (center of the filter) touched the bright
pixel at input[1,1] = 0.9:

  dL/dw[1][1] = 0.0*0.9 + 0.5*0.8 + (-0.3)*0.1 + 0.0*0.0 = 0.37

The gradient is nonzero — this weight contributed to the error
and gets adjusted. Same chain rule, same idea as Lesson 2,
just summed across all spatial positions.

LeCun's LeNet-5 was deployed in real systems: ATMs that read
checks, postal machines that sorted mail by zip code. By the
late 1990s, it was reading 10-20% of all checks deposited in
US banks.

But again notice the cost. Our simplified LeNet with {total_params} parameters
takes minutes to train on 1000 images in pure Python. The original
LeNet-5 had ~60,000 parameters and trained on 60,000 images.
Modern CNNs like ResNet have millions of parameters and train
on millions of images using GPUs. It's the same algorithm, but 
with vastly more compute.

Next up: the Transformer (Lesson 4), where we replace spatial
structure with attention, learning which parts of the input
matter for each part of the output.
""")

    # --- Plots ---
    os.makedirs("output", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LeNet-5 on MNIST", fontsize=14)

    axes[0].plot(range(1, epochs + 1), loss_history, color="#5CB8B2", linewidth=2, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")

    axes[1].plot(range(1, epochs + 1), [a * 100 for a in accuracy_history],
                 color="#E8915C", linewidth=2, marker="o")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig("output/03_lenet5_training.png", dpi=150)
    plt.close(fig)

    print("  Plot saved to ./output/03_lenet5_training.png")

    print(f"\n{'='*60}")
    print("  END OF LESSON 3")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
