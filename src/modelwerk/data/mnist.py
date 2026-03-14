"""MNIST dataset loader.

Downloads and parses the MNIST handwritten digit dataset
from the IDX binary format using only stdlib (urllib, struct, gzip).
"""

import gzip
import os
import struct
import urllib.request

from modelwerk.primitives.random import create_rng
from modelwerk.data.utils import subsample

Tensor3D = list[list[list[float]]]

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename: str, data_dir: str) -> str:
    """Download a file if not already cached. Returns local path."""
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        return path
    os.makedirs(data_dir, exist_ok=True)
    url = MIRROR + filename
    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, path)
    return path


def _parse_images(filepath: str) -> list[Tensor3D]:
    """Parse IDX image file. Returns list of 1x28x28 tensors (values 0-1)."""
    with gzip.open(filepath, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        images = []
        for _ in range(count):
            pixels = f.read(rows * cols)
            channel = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    row.append(pixels[r * cols + c] / 255.0)
                channel.append(row)
            images.append([channel])  # wrap in channel dimension
        return images


def _parse_labels(filepath: str) -> list[int]:
    """Parse IDX label file. Returns list of ints 0-9."""
    with gzip.open(filepath, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 2049
        return list(f.read(count))


def load_mnist(
    data_dir: str = "data/mnist",
    train_subset: int | None = 1000,
    test_subset: int | None = 200,
    seed: int = 42,
) -> tuple[list[Tensor3D], list[int], list[Tensor3D], list[int]]:
    """Download (if needed) and load MNIST.

    Returns (train_images, train_labels, test_images, test_labels).
    Images are 1x28x28 Tensor3D (values 0-1), labels are ints 0-9.
    """
    train_images = _parse_images(_download(_FILES["train_images"], data_dir))
    train_labels = _parse_labels(_download(_FILES["train_labels"], data_dir))
    test_images = _parse_images(_download(_FILES["test_images"], data_dir))
    test_labels = _parse_labels(_download(_FILES["test_labels"], data_dir))

    rng = create_rng(seed)
    if train_subset and train_subset < len(train_images):
        train_images, train_labels = subsample(rng, train_images, train_labels, train_subset)
    if test_subset and test_subset < len(test_images):
        test_images, test_labels = subsample(rng, test_images, test_labels, test_subset)

    return train_images, train_labels, test_images, test_labels
