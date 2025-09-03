#!/usr/bin/env python3

import numpy as np


def normalize_flow_scores(M, target_mean=6.4, target_std=6.6):
    """
    Normalize flow memorization scores to match VAE scale.
    Uses z-score normalization then rescales to target distribution.
    """
    # Z-score normalization
    M_normalized = (M - np.mean(M)) / np.std(M)

    # Rescale to target distribution
    M_rescaled = M_normalized * target_std + target_mean

    return M_rescaled


def analyze_and_normalize(filename, output_filename):
    print(f"Processing {filename}...")

    data = np.load(filename, allow_pickle=False)
    M_original = data["M"]

    print(
        f"Original - Min: {np.min(M_original[:, -1]):.2f}, Max: {np.max(M_original[:, -1]):.2f}"
    )
    print(
        f"Original - Mean: {np.mean(M_original[:, -1]):.2f}, Std: {np.std(M_original[:, -1]):.2f}"
    )

    # Normalize each column (epoch) separately
    M_normalized = np.zeros_like(M_original)
    for i in range(M_original.shape[1]):
        M_normalized[:, i] = normalize_flow_scores(M_original[:, i])

    print(
        f"Normalized - Min: {np.min(M_normalized[:, -1]):.2f}, Max: {np.max(M_normalized[:, -1]):.2f}"
    )
    print(
        f"Normalized - Mean: {np.mean(M_normalized[:, -1]):.2f}, Std: {np.std(M_normalized[:, -1]):.2f}"
    )

    # Create new file with normalized scores (excluding metadata to avoid pickle issues)
    save_dict = {"U": data["U"], "V": data["V"], "M": M_normalized}

    if "C" in data:
        save_dict["C"] = data["C"]

    # Save without metadata to avoid pickle issues
    np.savez(output_filename, **save_dict)
    print(f"Saved normalized data to {output_filename}")


if __name__ == "__main__":
    # Normalize both flow summary files
    analyze_and_normalize(
        "summaries/mem_flow_bmnist_lr3.npz",
        "summaries/mem_flow_bmnist_lr3_normalized.npz",
    )
    analyze_and_normalize(
        "summaries/mem_flow_bmnist_lr4.npz",
        "summaries/mem_flow_bmnist_lr4_normalized.npz",
    )
