#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate figure showing the distribution of memorization values for models
trained with two different learning rates.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import numpy as np

from fitter import Fitter

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
    parser.add_argument(
        "--use-percentiles",
        action="store_true",
        help="Use 1st-99th percentiles for axis limits instead of min-max (useful for outliers)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for x-axis (adds small constant to handle negative values)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize data by subtracting mean and dividing by std",
    )
    return parser.parse_args()


def make_tex(M3, M4, use_percentiles=False, log_scale=False, normalize=False):
    m3 = M3[:, -1]
    m4 = M4[:, -1]
    assert len(m3) == len(m4) == 60_000
    del M3, M4

    # Apply data transformations
    if normalize:
        # Combine data for global normalization
        all_data = np.concatenate([m3, m4])
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)
        m3 = (m3 - mean_val) / std_val
        m4 = (m4 - mean_val) / std_val
        print(f"Normalized data with mean={mean_val:.3f}, std={std_val:.3f}")

    if log_scale:
        # Shift data to make all values positive, then take log
        all_data = np.concatenate([m3, m4])
        min_val = np.min(all_data)
        shift = abs(min_val) + 1 if min_val <= 0 else 0
        m3 = np.log(m3 + shift + 1e-8)
        m4 = np.log(m4 + shift + 1e-8)
        print(f"Applied log scale with shift={shift:.3f}")

    # Calculate data-driven axis limits
    all_data = np.concatenate([m3, m4])
    data_min = np.min(all_data)
    data_max = np.max(all_data)
    data_range = data_max - data_min

    if use_percentiles:
        # Use percentiles to handle outliers
        p1, p99 = np.percentile(all_data, [1, 99])
        range_99 = p99 - p1
        padding_99 = 0.1 * range_99
        xmin = p1 - padding_99
        xmax = p99 + padding_99
        print(f"Using percentile-based range: [{p1:.2f}, {p99:.2f}]")
    else:
        # Add some padding (10% on each side)
        padding = 0.1 * data_range
        xmin = data_min - padding
        xmax = data_max + padding

        # For very large ranges, automatically switch to percentiles
        if data_range > 100:
            p1, p99 = np.percentile(all_data, [1, 99])
            range_99 = p99 - p1
            padding_99 = 0.1 * range_99
            xmin = p1 - padding_99
            xmax = p99 + padding_99
            print(f"Large range detected, using percentiles: [{p1:.2f}, {p99:.2f}]")

    print(f"Data range: [{data_min:.2f}, {data_max:.2f}]")
    print(f"Plot range: [{xmin:.2f}, {xmax:.2f}]")

    # Adjust xlabel based on transformations
    xlabel = "Memorization score"
    if normalize and log_scale:
        xlabel = "Log(normalized memorization score)"
    elif normalize:
        xlabel = "Normalized memorization score"
    elif log_scale:
        xlabel = "Log(memorization score)"

    tex = []

    pgfplotsset = """
    \\pgfplotsset{compat=newest,%
        /pgf/declare function={
    		gauss(\\x) = 1/sqrt(2*pi) * exp(-0.5 * \\x * \\x);%
    		johnson(\\x,\\a,\\b) = \\b/sqrt(\\x * \\x + 1) * gauss(\\a+\\b*ln(\\x+sqrt(\\x*\\x+1)));%
    		johnsonsu(\\x,\\a,\\b,\\loc,\\scale) = johnson((\\x - \\loc)/\\scale,\\a,\\b)/\\scale;%
    	},
    }
    """

    tex.append("\\documentclass[10pt,preview=true]{standalone}")
    # LuaLaTeX
    tex.append(
        "\\pdfvariable suppressoptionalinfo \\numexpr1+2+8+16+32+64+128+512\\relax"
    )
    tex.append("\\usepackage[utf8]{inputenc}")
    tex.append("\\usepackage[T1]{fontenc}")
    tex.append("\\usepackage{pgfplots}")
    tex.append(pgfplotsset)

    tex.append("\\definecolor{MyBlue}{HTML}{004488}")
    tex.append("\\definecolor{MyYellow}{HTML}{DDAA33}")
    tex.append("\\definecolor{MyRed}{HTML}{BB5566}")

    tex.append("\\begin{document}")
    tex.append("\\begin{tikzpicture}")

    fontsize = "\\normalsize"
    bins = 100
    ymin = 0
    ymax = 0.12

    axis_opts = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "scale only axis": None,
        "xlabel": xlabel,
        "ylabel": "Density",
        "width": "6cm",
        "height": "8cm",
        "ytick": "{0, 0.05, 0.10}",
        "yticklabels": "{0, 0.05, 0.10}",
        "xlabel style": {"font": fontsize},
        "ylabel style": {"font": fontsize},
        "xticklabel style": {"font": fontsize},
        "yticklabel style": {"font": fontsize},
        "legend pos": "north east",
        "legend style": {"font": fontsize},
        "legend cell align": "left",
    }

    tex.append(f"\\begin{{axis}}[{dict2tex(axis_opts)}]")

    thickness = "ultra thick"

    hist_plot_opts = {
        "forget plot": None,
        "draw": "none",
        "fill opacity": 0.3,
        "hist": {
            "bins": bins,
            "density": "true",
            "data min": xmin,
            "data max": xmax,
        },
    }
    line_plot_opts = {
        "domain": f"{xmin}:{xmax}",
        "samples": 201,
        "mark": "none",
        "solid": None,
        thickness: None,
    }
    vert_plot_opts = {
        "forget plot": None,
        "densely dashed": None,
        thickness: None,
    }

    ms = [m3, m4]
    labels = ["$\\eta = 10^{-3}$", "$\\eta = 10^{-4}$"]
    colors = ["MyBlue", "MyYellow"]

    for m, label, color in zip(ms, labels, colors):
        hist_plot_opts["fill"] = color
        line_plot_opts["draw"] = color
        vert_plot_opts["draw"] = color

        tex.append(f"\\addplot [{dict2tex(hist_plot_opts)}] table[y index=0] {{%")
        tex.append("data")
        for v in m:
            tex.append(str(v.item()))
        tex.append("};")

        f = Fitter(m, distributions=["johnsonsu"], xmin=xmin, xmax=xmax, bins=bins)
        f.fit()
        params = f.get_best()
        a, b, loc, scale = (
            params["johnsonsu"]["a"],
            params["johnsonsu"]["b"],
            params["johnsonsu"]["loc"],
            params["johnsonsu"]["scale"],
        )

        tex.append(f"\\addplot [{dict2tex(line_plot_opts)}] {{%")
        tex.append(f"johnsonsu(x, {a}, {b}, {loc}, {scale})")
        tex.append("};")
        tex.append(f"\\addlegendentry{{{label}}}")

        tex.append(f"\\addplot [{dict2tex(vert_plot_opts)}] coordinates {{%")
        tex.append(f"({np.quantile(m, 0.95)}, {ymin}) ({np.quantile(m, 0.95)}, {ymax})")
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

    tex = make_tex(M3, M4, args.use_percentiles, args.log_scale, args.normalize)
    with open(args.output, "w") as fp:
        fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
