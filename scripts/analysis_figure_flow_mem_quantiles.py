#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate figure showing the evolution of quantiles of the memorization score
for models trained with two different learning rates.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import Line
from analysis_utils import dict2tex


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr3-file",
        help="NumPy compressed archive with results for learning rate 1e-3",
        required=True,
    )
    parser.add_argument(
        "--lr4-file",
        help="NumPy compressed archive with results for learning rate 1e-4",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file to write to (.tex)",
    )
    parser.add_argument(
        "--use-percentiles",
        action="store_true",
        help="Use 5th-95th percentiles for y-axis limits instead of min-max (useful for outliers)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for y-axis (adds small constant to handle negative values)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize data by subtracting mean and dividing by std",
    )
    parser.add_argument(
        "--height",
        type=str,
        default="8cm",
        help="Height of the plot (default: 8cm, try 10cm or 12cm for taller plots)",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Remove legend from the plot",
    )
    parser.add_argument(
        "--legend-pos",
        type=str,
        default="north west",
        help="Legend position (default: 'north west', try 'south east', 'outer north east', etc.)",
    )
    return parser.parse_args()


def make_tex(
    epochs,
    M3,
    M4,
    use_percentiles=False,
    log_scale=False,
    normalize=False,
    height="8cm",
    no_legend=False,
    legend_pos="north west",
):
    # Apply data transformations
    if normalize:
        # Combine all data for global normalization
        all_data = np.concatenate([M3.flatten(), M4.flatten()])
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)
        M3 = (M3 - mean_val) / std_val
        M4 = (M4 - mean_val) / std_val
        print(f"Normalized data with mean={mean_val:.3f}, std={std_val:.3f}")

    if log_scale:
        # Shift data to make all values positive, then take log
        all_data = np.concatenate([M3.flatten(), M4.flatten()])
        min_val = np.min(all_data)
        shift = abs(min_val) + 1 if min_val <= 0 else 0
        M3 = np.log(M3 + shift + 1e-8)
        M4 = np.log(M4 + shift + 1e-8)
        print(f"Applied log scale with shift={shift:.3f}")

    # Calculate quantiles for all data to determine y-axis limits
    data = [
        np.quantile(M3, 0.95, axis=0),
        np.quantile(M3, 0.999, axis=0),
        np.quantile(M4, 0.95, axis=0),
        np.quantile(M4, 0.999, axis=0),
    ]

    all_quantile_data = np.concatenate(data)
    data_min = np.min(all_quantile_data)
    data_max = np.max(all_quantile_data)
    data_range = data_max - data_min

    if use_percentiles:
        # Use percentiles of the quantile data to handle outliers
        p5, p95 = np.percentile(all_quantile_data, [5, 95])
        range_90 = p95 - p5
        padding_90 = 0.1 * range_90
        ymin = p5 - padding_90
        ymax = p95 + padding_90
        print(f"Using percentile-based y-range: [{p5:.2f}, {p95:.2f}]")
    else:
        # Add some padding (10% on each side)
        padding = 0.1 * data_range
        ymin = data_min - padding
        ymax = data_max + padding

        # For very large ranges, automatically switch to percentiles
        if data_range > 50:
            p5, p95 = np.percentile(all_quantile_data, [5, 95])
            range_90 = p95 - p5
            padding_90 = 0.1 * range_90
            ymin = p5 - padding_90
            ymax = p95 + padding_90
            print(f"Large range detected, using percentiles: [{p5:.2f}, {p95:.2f}]")

    print(f"Quantile data range: [{data_min:.2f}, {data_max:.2f}]")
    print(f"Plot y-range: [{ymin:.2f}, {ymax:.2f}]")

    # Adjust ylabel based on transformations
    ylabel = "Quantile value"
    if normalize and log_scale:
        ylabel = "Log(normalized quantile value)"
    elif normalize:
        ylabel = "Normalized quantile value"
    elif log_scale:
        ylabel = "Log(quantile value)"

    tex = []

    tex.append("\\documentclass[10pt,preview=true]{standalone}")
    tex.append("\\pdfinfoomitdate=1")
    tex.append("\\pdftrailerid{}")
    tex.append("\\pdfsuppressptexinfo=1")
    tex.append("\\pdfinfo{ /Creator () /Producer () }")
    tex.append("\\usepackage[utf8]{inputenc}")
    tex.append("\\usepackage[T1]{fontenc}")
    tex.append("\\usepackage{pgfplots}")
    tex.append("\\pgfplotsset{compat=newest}")

    tex.append("\\definecolor{MyBlue}{HTML}{004488}")
    tex.append("\\definecolor{MyYellow}{HTML}{DDAA33}")
    tex.append("\\definecolor{MyRed}{HTML}{BB5566}")

    tex.append("\\begin{document}")
    tex.append("\\begin{tikzpicture}")

    fontsize = "\\normalsize"
    thickness = "ultra thick"

    axis_opts = {
        "xmin": 0,
        "xmax": 107,
        "ymin": ymin,
        "ymax": ymax,
        "scale only axis": None,
        "xlabel": "Epochs",
        "ylabel": ylabel,
        "width": "6cm",
        "height": height,
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": fontsize},
        "yticklabel style": {"font": fontsize},
    }

    # Add legend options only if legend is not disabled
    if not no_legend:
        axis_opts.update(
            {
                "legend pos": legend_pos,
                "legend style": {"font": fontsize},
                "legend cell align": "left",
            }
        )

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    data = [
        np.quantile(M3, 0.95, axis=0),
        np.quantile(M3, 0.999, axis=0),
        np.quantile(M4, 0.95, axis=0),
        np.quantile(M4, 0.999, axis=0),
    ]
    styles = [
        ", ".join(["solid", thickness, "MyBlue"]),
        ", ".join(["densely dashed", thickness, "MyBlue"]),
        ", ".join(["solid", thickness, "MyYellow"]),
        ", ".join(["densely dashed", thickness, "MyYellow"]),
    ]
    labels = [
        "$\\eta = 10^{-3}$, $q = 0.95$",
        "$\\eta = 10^{-3}$, $q = 0.999$",
        "$\\eta = 10^{-4}$, $q = 0.95$",
        "$\\eta = 10^{-4}$, $q = 0.999$",
    ]

    lines = [
        Line(xs=epochs, ys=ys, style=style, label=label)
        for ys, style, label in zip(data, styles, labels)
    ]
    for line in lines:
        tex.append(f"\\addplot [{line.style}] table {{%")
        tex.extend([f"{x} {y}" for x, y in zip(line.xs, line.ys)])
        tex.append("};")
        if not no_legend:
            tex.append(f"\\addlegendentry{{{line.label}}}")

    tex.append("\\end{axis}")
    tex.append("\\end{tikzpicture}")
    tex.append("\\end{document}")
    return tex


def show_plot(epochs, M3, M4, use_percentiles=False, log_scale=False, normalize=False):
    # Apply the same transformations as in make_tex
    if normalize:
        all_data = np.concatenate([M3.flatten(), M4.flatten()])
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)
        M3 = (M3 - mean_val) / std_val
        M4 = (M4 - mean_val) / std_val

    if log_scale:
        all_data = np.concatenate([M3.flatten(), M4.flatten()])
        min_val = np.min(all_data)
        shift = abs(min_val) + 1 if min_val <= 0 else 0
        M3 = np.log(M3 + shift + 1e-8)
        M4 = np.log(M4 + shift + 1e-8)

    # Calculate dynamic y-limits
    data = [
        np.quantile(M3, 0.95, axis=0),
        np.quantile(M3, 0.999, axis=0),
        np.quantile(M4, 0.95, axis=0),
        np.quantile(M4, 0.999, axis=0),
    ]
    all_quantile_data = np.concatenate(data)

    if use_percentiles:
        p5, p95 = np.percentile(all_quantile_data, [5, 95])
        ymax_plot = p95 + 0.1 * (p95 - p5)
    else:
        data_max = np.max(all_quantile_data)
        data_min = np.min(all_quantile_data)
        data_range = data_max - data_min
        if data_range > 50:
            p5, p95 = np.percentile(all_quantile_data, [5, 95])
            ymax_plot = p95 + 0.1 * (p95 - p5)
        else:
            ymax_plot = data_max + 0.1 * data_range

    plt.figure()
    plt.plot(
        epochs,
        np.quantile(M3, 0.95, axis=0),
        ls="-",
        color="tab:blue",
        label="$\\eta = 10^{-3}$, q = 0.95",
    )
    plt.plot(
        epochs,
        np.quantile(M3, 0.999, axis=0),
        ls="--",
        color="tab:blue",
        label="$\\eta = 10^{-3}$, q = 0.999",
    )

    plt.plot(
        epochs,
        np.quantile(M4, 0.95, axis=0),
        ls="-",
        color="tab:orange",
        label="$\\eta = 10^{-4}$, q = 0.95",
    )
    plt.plot(
        epochs,
        np.quantile(M4, 0.999, axis=0),
        ls="--",
        color="tab:orange",
        label="$\\eta = 10^{-4}$, q = 0.999",
    )
    plt.ylim(ymax=ymax_plot)
    plt.legend()
    plt.xlabel("Epochs")

    # Set appropriate ylabel
    ylabel = "Memorization score"
    if normalize and log_scale:
        ylabel = "Log(normalized memorization score)"
    elif normalize:
        ylabel = "Normalized memorization score"
    elif log_scale:
        ylabel = "Log(memorization score)"
    plt.ylabel(ylabel)

    plt.show()


def main():
    args = parse_args()

    bmnist3 = np.load(args.lr3_file)
    bmnist4 = np.load(args.lr4_file)

    M3 = bmnist3["M"]
    M4 = bmnist4["M"]

    epochs = np.array(list(range(5, 105, 5)))

    if args.output is None:
        show_plot(epochs, M3, M4, args.use_percentiles, args.log_scale, args.normalize)
    else:
        tex = make_tex(
            epochs,
            M3,
            M4,
            args.use_percentiles,
            args.log_scale,
            args.normalize,
            args.height,
            args.no_legend,
            args.legend_pos,
        )
        with open(args.output, "w") as fp:
            fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
