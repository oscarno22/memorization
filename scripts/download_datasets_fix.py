#!/usr/bin/env python3
"""
Quick fix to download MNIST, CIFAR-10, and CelebA datasets with SSL verification disabled.
Run this once before running the main memorization script.
"""

import ssl
import os
from torchvision import datasets

# Temporarily disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context


def download_datasets():
    """Download all required datasets."""
    base_root = os.getenv("DL_DATASETS_ROOT", "/tmp")

    # Download MNIST
    mnist_root = os.path.join(base_root, "MNIST")
    os.makedirs(mnist_root, exist_ok=True)
    print(f"Downloading MNIST to {mnist_root}...")
    try:
        datasets.MNIST(mnist_root, train=True, download=True)
        datasets.MNIST(mnist_root, train=False, download=True)
        print("MNIST download completed successfully!")
    except Exception as e:
        print(f"MNIST download failed: {e}")

    # Download CIFAR-10
    cifar_root = os.path.join(base_root, "CIFAR10")
    os.makedirs(cifar_root, exist_ok=True)
    print(f"Downloading CIFAR-10 to {cifar_root}...")
    try:
        datasets.CIFAR10(cifar_root, train=True, download=True)
        datasets.CIFAR10(cifar_root, train=False, download=True)
        print("CIFAR-10 download completed successfully!")
    except Exception as e:
        print(f"CIFAR-10 download failed: {e}")

    # Download CelebA
    celeba_root = os.path.join(base_root, "CelebA")
    os.makedirs(celeba_root, exist_ok=True)
    print(f"Downloading CelebA to {celeba_root}...")
    try:
        # CelebA requires split parameter
        datasets.CelebA(celeba_root, split="train", download=True)
        datasets.CelebA(celeba_root, split="valid", download=True)
        datasets.CelebA(celeba_root, split="test", download=True)
        print("CelebA download completed successfully!")
    except Exception as e:
        print(f"CelebA download failed: {e}")

    print("All dataset downloads completed!")


if __name__ == "__main__":
    download_datasets()
