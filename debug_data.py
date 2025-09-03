#!/usr/bin/env python3

import numpy as np
import sys


def diagnose_data(filename):
    print(f"Diagnosing {filename}...")

    try:
        data = np.load(filename)
        print(f"Keys in file: {list(data.keys())}")

        if "M" in data:
            M = data["M"]
            print(f"M shape: {M.shape}")

            # Check last column (memorization scores)
            m = M[:, -1]
            print(f"Memorization scores shape: {m.shape}")
            print(f"Min: {np.min(m)}")
            print(f"Max: {np.max(m)}")
            print(f"Mean: {np.mean(m)}")
            print(f"Std: {np.std(m)}")
            print(f"Number of NaN: {np.sum(np.isnan(m))}")
            print(f"Number of Inf: {np.sum(np.isinf(m))}")
            print(f"Number of unique values: {len(np.unique(m))}")

            # Check for constant values
            if len(np.unique(m)) == 1:
                print("WARNING: All memorization scores are identical!")

            # Try histogram to see if it fails
            try:
                hist, bins = np.histogram(m, bins=100, density=True, range=(-15, 35))
                print("Histogram computation: SUCCESS")
                print(f"Histogram sum: {np.sum(hist)}")
                print(
                    f"Bin widths: min={np.min(np.diff(bins))}, max={np.max(np.diff(bins))}"
                )
            except Exception as e:
                print(f"Histogram computation FAILED: {e}")

            # Show first few values
            print(f"First 10 values: {m[:10]}")
            print(f"Last 10 values: {m[-10:]}")

        else:
            print("No 'M' key found in data file")

    except Exception as e:
        print(f"Error loading file: {e}")

    print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            diagnose_data(filename)
    else:
        # Check the summary files
        diagnose_data("summaries/mem_flow_bmnist_lr3.npz")
        diagnose_data("summaries/mem_flow_bmnist_lr4.npz")
