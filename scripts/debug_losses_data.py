#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to examine the structure and contents of result files
for the losses analysis script.
"""

import argparse
import gzip
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr3-file",
        help="Result file with learning rate 1e-3",
        required=True,
    )
    parser.add_argument(
        "--lr4-file",
        help="Result file with learning rate 1e-4",
        required=True,
    )
    return parser.parse_args()


def examine_file(filename):
    print(f"\n=== Examining {filename} ===")
    
    try:
        with gzip.open(filename, "rb") as fp:
            contents = fp.read()
            data = json.loads(contents.decode("utf-8"))
        
        print("✓ File successfully loaded as gzipped JSON")
        
        # Check top-level structure
        print(f"Top-level keys: {list(data.keys())}")
        
        if "meta" in data:
            print(f"Metadata keys: {list(data['meta'].keys())}")
            for key, value in data["meta"].items():
                print(f"  {key}: {value}")
        else:
            print("❌ No 'meta' key found")
        
        if "results" in data:
            print(f"Results keys: {list(data['results'].keys())}")
            
            if "losses" in data["results"]:
                losses = data["results"]["losses"]
                print(f"Losses keys: {list(losses.keys())}")
                
                for loss_type in ["train", "test"]:
                    if loss_type in losses:
                        loss_data = losses[loss_type]
                        print(f"  {loss_type} loss:")
                        print(f"    Type: {type(loss_data)}")
                        if isinstance(loss_data, list):
                            print(f"    Length: {len(loss_data)}")
                            if len(loss_data) > 0:
                                print(f"    First 5 values: {loss_data[:5]}")
                                print(f"    Last 5 values: {loss_data[-5:]}")
                                print(f"    Min: {min(loss_data):.4f}")
                                print(f"    Max: {max(loss_data):.4f}")
                                print(f"    Mean: {np.mean(loss_data):.4f}")
                        elif isinstance(loss_data, dict):
                            print(f"    Dict keys: {list(loss_data.keys())}")
                            # If it's a dict, show a few entries
                            for i, (k, v) in enumerate(loss_data.items()):
                                if i < 3:
                                    print(f"      {k}: {v}")
                                elif i == 3:
                                    print(f"      ... (and {len(loss_data)-3} more)")
                                    break
                        else:
                            print(f"    Data: {loss_data}")
                    else:
                        print(f"  ❌ No '{loss_type}' loss found")
            else:
                print("❌ No 'losses' key found in results")
                print(f"Available keys in results: {list(data['results'].keys())}")
        else:
            print("❌ No 'results' key found")
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False
    
    return True


def compare_files(lr3_file, lr4_file):
    print(f"\n=== Comparing Files ===")
    
    try:
        # Load both files
        with gzip.open(lr3_file, "rb") as fp:
            lr3_data = json.loads(fp.read().decode("utf-8"))
        
        with gzip.open(lr4_file, "rb") as fp:
            lr4_data = json.loads(fp.read().decode("utf-8"))
        
        # Compare metadata
        print("Metadata comparison:")
        for key in set(lr3_data.get("meta", {}).keys()) | set(lr4_data.get("meta", {}).keys()):
            lr3_val = lr3_data.get("meta", {}).get(key, "MISSING")
            lr4_val = lr4_data.get("meta", {}).get(key, "MISSING")
            if lr3_val == lr4_val:
                print(f"  ✓ {key}: {lr3_val}")
            else:
                print(f"  ⚠ {key}: lr3={lr3_val}, lr4={lr4_val}")
        
        # Compare loss data lengths
        if ("results" in lr3_data and "results" in lr4_data and 
            "losses" in lr3_data["results"] and "losses" in lr4_data["results"]):
            
            print("\nLoss data length comparison:")
            for loss_type in ["train", "test"]:
                lr3_losses = lr3_data["results"]["losses"].get(loss_type, [])
                lr4_losses = lr4_data["results"]["losses"].get(loss_type, [])
                
                if isinstance(lr3_losses, list) and isinstance(lr4_losses, list):
                    print(f"  {loss_type}: lr3={len(lr3_losses)}, lr4={len(lr4_losses)}")
                    if len(lr3_losses) == len(lr4_losses):
                        print(f"    ✓ Same length")
                    else:
                        print(f"    ⚠ Different lengths!")
                else:
                    print(f"  {loss_type}: lr3={type(lr3_losses)}, lr4={type(lr4_losses)}")
                    print(f"    ⚠ Not both lists!")
        
    except Exception as e:
        print(f"❌ Error comparing files: {e}")


def main():
    args = parse_args()
    
    print("Debugging loss data files for analysis_figure_mem_losses.py")
    print("=" * 60)
    
    # Examine each file
    lr3_ok = examine_file(args.lr3_file)
    lr4_ok = examine_file(args.lr4_file)
    
    if lr3_ok and lr4_ok:
        compare_files(args.lr3_file, args.lr4_file)
    
    print(f"\n=== Summary ===")
    if lr3_ok and lr4_ok:
        print("✓ Both files loaded successfully")
        print("Check the output above to see if the data structure matches what")
        print("analysis_figure_mem_losses.py expects:")
        print("  - data['results']['losses']['train'] should be a list of numbers")
        print("  - data['results']['losses']['test'] should be a list of numbers")
        print("  - Both lists should have the same length")
    else:
        print("❌ One or both files failed to load")


if __name__ == "__main__":
    main()
