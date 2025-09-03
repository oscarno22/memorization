#!/usr/bin/env python3
"""
Test script for the new SimpleRealNVP flow model
"""

import torch
import sys

# Add scripts directory to path
sys.path.append("/Users/ozzynozzy/UNCC/memorization-gen/memorization/scripts")

from models import SimpleRealNVP


def test_flow_model():
    print("Testing SimpleRealNVP model...")

    # Create a simple model
    model = SimpleRealNVP(
        img_dim=8,  # Small for testing
        in_channels=1,
        num_layers=2,  # Small for testing
        hidden_dim=32,  # Small for testing
    )

    print(f"Model description: {model.description}")
    print(f"Model latent dim: {model.latent_dim}")

    # Test with dummy data
    batch_size = 4
    dummy_data = torch.randn(batch_size, 1, 8, 8)

    print(f"Input shape: {dummy_data.shape}")

    # Test forward pass
    try:
        output, mu, logvar = model.forward(dummy_data)
        print(f"Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False

    # Test log probability computation
    try:
        log_probs = model.log_prob(dummy_data)
        print(f"Log prob computation successful! Shape: {log_probs.shape}")
        print(f"Sample log probabilities: {log_probs[:3].tolist()}")
    except Exception as e:
        print(f"Log prob computation failed: {e}")
        return False

    # Test training step
    try:
        loss = model.step(dummy_data)
        print(f"Training step successful! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"Training step failed: {e}")
        return False

    # Test sampling
    try:
        samples = model.sample(2)
        print(f"Sampling successful! Sample shape: {samples.shape}")
    except Exception as e:
        print(f"Sampling failed: {e}")
        return False

    print("All tests passed! SimpleRealNVP model is working correctly.")
    return True


if __name__ == "__main__":
    success = test_flow_model()
    if success:
        print("\nYou can now run the flow model with:")
        print("python scripts/memorization.py \\")
        print("  --dataset BinarizedMNIST \\")
        print("  --model SimpleRealNVP \\")
        print("  --epochs 5 \\")
        print("  --batch-size 64 \\")
        print("  --learning-rate 1e-3 \\")
        print("  --repeats 1 \\")
        print("  --mode split-cv")
    else:
        print("\nModel test failed. Check the implementation.")
