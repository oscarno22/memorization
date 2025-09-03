#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate figure showing the distribution of memorization values for models
trained with two different learning rates - simplified version without distribution fitting.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import numpy as np

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
        "-o", "--output", help="Output file to write to (.tex)", required=True
    )
    return parser.parse_args()


def make_tex(M3, M4):
    m3 = M3[:, -1]
    m4 = M4[:, -1]
    assert len(m3) == len(m4) == 60_000
    del M3, M4

    tex = []

    # Simplified LaTeX without distribution fitting
    tex.append("\\documentclass[10pt,preview=true]{standalone}")
    tex.append(
        "\\pdfvariable suppressoptionalinfo \\numexpr1+2+8+16+32+64+128+512\\relax"
    )
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
    bins = 50

    # Auto-determine reasonable range based on data
    all_data = np.concatenate([m3, m4])
    xmin = np.quantile(all_data, 0.01)  # 1st percentile
    xmax = np.quantile(all_data, 0.99)  # 99th percentile
    ymin = 0
    ymax = 0.4  # Adjust as needed

    print(f"Using plot range: [{xmin:.2f}, {xmax:.2f}]")

    axis_opts = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "scale only axis": None,
        "xlabel": "Normalized memorization score",
        "ylabel": "Density",
        "width": "8cm",
        "height": "6cm",
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": fontsize},
        "yticklabel style": {"font": fontsize},
        "legend pos": "north east",
        "legend style": {"font": fontsize},
        "legend cell align": "left",
    }

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    hist_plot_opts = {
        "ybar interval": None,
        "fill opacity": 0.6,
        "draw": "black",
        "line width": "0.5pt",
    }

    ms = [m3, m4]
    labels = ["$\\eta = 10^{-3}$", "$\\eta = 10^{-4}$"]
    colors = ["MyBlue", "MyYellow"]

    for i, (m, label, color) in enumerate(zip(ms, labels, colors)):
        hist_plot_opts["fill"] = color

        # Filter data to plotting range
        m_filtered = m[(m >= xmin) & (m <= xmax)]
        print(f"{label}: Using {len(m_filtered)}/{len(m)} points in plot range")

        if len(m_filtered) == 0:
            print(f"WARNING: No data points for {label} in plot range")
            continue

        # Compute histogram
        counts, bin_edges = np.histogram(
            m_filtered, bins=bins, range=(xmin, xmax), density=True
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        tex.append(f"\\addplot+[{dict2tex(hist_plot_opts)}] coordinates {{")
        for center, count in zip(bin_centers, counts):
            tex.append(f"({center:.6f},{count:.6f})")
        tex.append("};")
        tex.append(f"\\addlegendentry{{{label}}}")

        # Add vertical line at 95th percentile
        q95 = np.quantile(m, 0.95)
        if xmin <= q95 <= xmax:
            vert_plot_opts = {
                "forget plot": None,
                "densely dashed": None,
                "ultra thick": None,
                "draw": color,
            }
            tex.append(f"\\addplot [{dict2tex(vert_plot_opts)}] coordinates {{%")
            tex.append(f"({q95:.6f}, {ymin}) ({q95:.6f}, {ymax})")
            tex.append("};")

    tex.append("\\end{axis}")
    tex.append("\\end{tikzpicture}")
    tex.append("\\end{document}")
    return tex


def main():
    args = parse_args()

    bmnist3 = np.load(args.lr3_file)
    bmnist4 = np.load(args.lr4_file)

    M3 = bmnist3["M"]
    M4 = bmnist4["M"]

    tex = make_tex(M3, M4)
    with open(args.output, "w") as fp:
        fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
