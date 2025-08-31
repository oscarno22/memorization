#!/usr/bin/env python3
"""
Quick fix to download MNIST data with SSL verification disabled.
Run this once before running the main memorization script.
"""

import ssl
import os
from torchvision import datasets

# Temporarily disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Download MNIST to the expected location
root = os.path.join(os.getenv("DL_DATASETS_ROOT", "/tmp"), "MNIST")
os.makedirs(root, exist_ok=True)

print(f"Downloading MNIST to {root}...")
try:
    datasets.MNIST(root, train=True, download=True)
    datasets.MNIST(root, train=False, download=True)
    print("MNIST download completed successfully!")
except Exception as e:
    print(f"Download failed: {e}")
