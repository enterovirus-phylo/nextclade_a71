"""
Figures for the Nextclade vs RIVM fragment benchmark.

Reads the per-fragment report and the region x length summary written by
compare_clade_assignments.py and writes PNG + SVG to the output directory.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS = {"nextclade": "#2b8cbe", "RIVM": "#e6550d"}
OUTCOME_COLORS = {
    "exact": "#238b45",
    "ancestral": "#a1d99b",
    "descendant": "#74c476",
    "wrong_clade": "#fdae6b",
    "wrong_type": "#d94801",
    "unassigned": "#bdbdbd",
}
OUTCOME_ORDER = ["exact", "ancestral", "descendant", "wrong_clade", "wrong_type", "unassigned"]
REGION_ORDER = ["VP1", "3D", "genome"]
REGION_LABEL = {"VP1": "VP1 fragments", "3D": "3D fragments",
                "genome": "Random genome windows"}


def save(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outdir / name}.png")


def regions_present(df):
    present = [r for r in REGION_ORDER if r in set(df["gene"])]
    return present + sorted(set(df["gene"]) - set(REGION_ORDER))


def rate_vs_length(summary, outdir, value, lo, hi, title, ylabel, name):
    """One facet per region, one line per tool, Wilson CI ribbons."""
    regions = regions_present(summary)
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 4.2),
                             sharey=True, squeeze=False)
    for ax, region in zip(axes[0], regions):
        facet = summary[summary["gene"] == region]
        for method, group in facet.groupby("method"):
            group = group.sort_values("length")
            colour = METHOD_COLORS.get(method, "#666666")
            ax.plot(group["length"], group[value], marker="o", ms=4,
                    color=colour, label=method)
            ax.fill_between(group["length"], group[lo], group[hi],
                            color=colour, alpha=0.18, linewidth=0)
        ax.set_title(REGION_LABEL.get(region, region))
        ax.set_xlabel("Fragment length (nt)")
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(frameon=False)
    fig.suptitle(title, y=1.02, fontsize=13)
    save(fig, outdir, name)


def outcome_bars(report, outdir):
    """Stacked outcome composition per tool x region."""
    regions = regions_present(report)
    methods = [("nextclade", "nextclade_outcome"), ("RIVM", "rivm_outcome")]
    labels, bottoms, data = [], None, {o: [] for o in OUTCOME_ORDER}
    for region in regions:
        for method, col in methods:
            subset = report[(report["gene"] == region) & (report[col] != "no_truth")]
            total = max(len(subset), 1)
            labels.append(f"{region}\n{method}")
            counts = subset[col].value_counts()
            for outcome in OUTCOME_ORDER:
                data[outcome].append(100 * counts.get(outcome, 0) / total)

    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 3, 4.5))
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    for outcome in OUTCOME_ORDER:
        values = np.array(data[outcome])
        ax.bar(x, values, bottom=bottoms, color=OUTCOME_COLORS[outcome],
               label=outcome.replace("_", " "), width=0.72)
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of fragments with ground truth")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("Assignment outcome by region and tool")
    save(fig, outdir, "outcome_composition")


def confusion(report, outdir):
    """Truth x call heatmap per tool."""
    panels = [("nextclade", "nextclade_clade"), ("RIVM", "rivm_clade")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (method, col) in zip(axes, panels):
        subset = report[report["nextstrain_clade"] != "NA"]
        table = pd.crosstab(subset["nextstrain_clade"], subset[col])
        im = ax.imshow(table.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index, fontsize=8)
        ax.set_xlabel(f"{method} call")
        ax.set_ylabel("Nextstrain clade (truth)")
        ax.set_title(method)
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.values[i, j]
                if value:
                    ax.text(j, i, value, ha="center", va="center", fontsize=7,
                            color="white" if value > table.values.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Truth vs call", y=1.01, fontsize=13)
    save(fig, outdir, "confusion")


def outcome_vs_position(report, outdir, vp1_range=None):
    """Where in the genome each tool succeeds - the VP1 limitation, drawn directly."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (method, col) in zip(axes, [("nextclade", "nextclade_outcome"),
                                        ("RIVM", "rivm_outcome")]):
        subset = report[report[col] != "no_truth"]
        for outcome in OUTCOME_ORDER:
            rows = subset[subset[col] == outcome]
            if rows.empty:
                continue
            ax.scatter(rows["start"], rows["length"], s=18, alpha=0.75,
                       color=OUTCOME_COLORS[outcome], label=outcome.replace("_", " "))
        if vp1_range:
            ax.axvspan(vp1_range[0], vp1_range[1], color="#2b8cbe", alpha=0.08, zorder=0)
            ax.text(sum(vp1_range) / 2, ax.get_ylim()[1], "VP1", ha="center", va="top",
                    fontsize=9, color="#2b8cbe")
        ax.set_ylabel("Fragment length (nt)")
        ax.set_title(method, loc="left", fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[-1].set_xlabel("Fragment start position on parent genome (nt)")
    axes[0].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.suptitle("Assignment outcome by genomic position", y=1.0, fontsize=13)
    save(fig, outdir, "outcome_by_position")


def parse_vp1_range(gff_file):
    """VP1 coordinates from the dataset annotation, for shading the position plot."""
    path = Path(gff_file)
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 9 and parts[2] in ("gene", "CDS") and "VP1" in parts[8]:
            return (int(parts[3]) - 1, int(parts[4]))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gff", default="dataset/genome_annotation.gff3")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    report = pd.read_csv(args.report, sep="\t", keep_default_na=False)
    summary = pd.read_csv(args.summary, sep="\t")

    # Figures are drawn per arm: the matched arm is the one where cross-region
    # comparison is valid, the main arm is the higher-powered within-region measurement.
    if "arm" in summary.columns:
        arms = list(dict.fromkeys(summary["arm"]))
    else:
        arms = [None]

    for arm in arms:
        draw_all(report, summary, outdir, arm, args.gff)


def draw_all(report, summary, outdir, arm, gff):
    if arm is not None:
        summary = summary[summary["arm"] == arm]
        report = report[report["arm"] == arm]
        outdir = outdir / arm

    # Figure 1 - the headline: can the tool answer at all?
    rate_vs_length(summary, outdir, "pct_assigned", "assigned_lo", "assigned_hi",
                   "Fragments assigned a clade, by region and length",
                   "% assigned (95% CI)", "assignment_rate")

    # Figure 2 - is it right when it does answer? Frames the difference as scope,
    # not quality, which is the harder claim to rebut.
    rate_vs_length(summary, outdir, "pct_correct_given_assigned",
                   "correct_given_assigned_lo", "correct_given_assigned_hi",
                   "Accuracy given an assignment was made",
                   "% correct | assigned (95% CI)", "accuracy_given_assigned")

    rate_vs_length(summary, outdir, "pct_correct", "correct_lo", "correct_hi",
                   "Correct clade assignments, by region and length",
                   "% correct of all fragments (95% CI)", "overall_accuracy")

    outcome_bars(report, outdir)
    confusion(report, outdir)
    outcome_vs_position(report, outdir, parse_vp1_range(gff))


if __name__ == "__main__":
    main()
