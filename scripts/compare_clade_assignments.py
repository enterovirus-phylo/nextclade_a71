"""
Benchmark clade assignment on sequence fragments: Nextclade vs the RIVM typing tool.

Each fragment is cut from a parent sequence whose whole-genome clade is known from two
independent sources, so every fragment call can be scored two ways:

  * agreement  - against the shared Nextstrain ground truth, after nomenclature mapping
  * self-consistency - against the SAME tool's own whole-genome call on the parent

Results are stratified by region (VP1 / 3D / random genome window) and fragment length,
because the aggregate hides the finding that matters: RIVM's subgenogroup assignment is
VP1-based, so it is structurally unable to call fragments from elsewhere in the genome,
while remaining accurate on the fragments it can call.
"""
import argparse
import math
import sys

import pandas as pd

NA = "NA"
# compared case-insensitively: nextclade emits lowercase "unassigned", RIVM "Could not assign"
UNASSIGNED_VALUES = {"", "na", "nan", "none", "could not assign", "unassigned"}

# Per-row outcomes. Ordered worst-to-best for stacked plots.
OUTCOMES = ["unassigned", "wrong_type", "wrong_clade", "descendant", "ancestral", "exact"]
# a coarser or finer answer than the reference is not an error: both identify the lineage
CORRECTISH = {"exact", "ancestral", "descendant"}


def norm(value):
    """Normalise a clade/type cell to a plain string, with all flavours of missing as NA."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NA
    text = str(value).strip()
    return NA if text.lower() in UNASSIGNED_VALUES else text


def load_name_map(path):
    """Load the tool-specific clade renaming table into {(tool, from): to}."""
    if not path:
        return {}
    rows = pd.read_csv(path, sep="\t", comment="#")
    if rows.empty:
        return {}
    return {(r["tool"], str(r["from"])): str(r["to"]) for _, r in rows.iterrows()}


def apply_name_map(series, tool, name_map, unmapped):
    """Rename a tool's clade calls onto the Nextstrain nomenclature, recording pass-throughs."""
    def rename(value):
        if value == NA:
            return NA
        mapped = name_map.get((tool, value))
        if mapped is None:
            unmapped.setdefault(tool, set()).add(value)
            return value
        return mapped
    return series.map(rename)


def is_within(child, parent):
    """True if clade `child` sits inside clade `parent` in the EV-A71 nomenclature.

    The hierarchy mixes two conventions: a letter-to-digit boundary opens the first
    sublevel (B -> B2, C -> C1) and a dot opens each further one (C1 -> C1.2). So a
    new level starts with a dot, or with a digit where the parent ended in a letter.
    Requiring that boundary is what keeps C1 from being read as a parent of a
    hypothetical C10, which a plain prefix match would get wrong.
    """
    if child == parent or not child.startswith(parent) or len(child) <= len(parent):
        return False
    nxt = child[len(parent)]
    if nxt == ".":
        return True
    return nxt.isdigit() and not parent[-1].isdigit()


def relation(call, truth):
    """How a call relates to the truth: exact, ancestral, descendant or wrong_clade."""
    if call == truth:
        return "exact"
    if is_within(truth, call):
        # the tool gave a correct but less-resolved answer (C for a C1 fragment)
        return "ancestral"
    if is_within(call, truth):
        # the tool was MORE specific than the reference, which happens wherever the
        # ground-truth column is coarser than the dataset's clade set
        return "descendant"
    return "wrong_clade"


def classify(call, truth, call_type, virus):
    """Assign one row to an outcome bucket for a single tool."""
    if truth == NA:
        return "no_truth"
    if call == NA:
        # a tool that typed the fragment as a different serotype gave a wrong answer,
        # not a missing one - worth separating from an honest "could not assign"
        if call_type not in (NA, virus):
            return "wrong_type"
        return "unassigned"
    return relation(call, truth)


def wilson(successes, total, z=1.96):
    """Wilson score interval - well-behaved at the 0% and 100% rates this benchmark hits."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, 100 * (centre - margin)), min(100.0, 100 * (centre + margin)))


def read_nextclade(path):
    """Read a nextclade TSV down to the columns this benchmark uses."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    keep = [c for c in ["seqName", "clade", "coverage", "qc.overallStatus",
                        "alignmentStart", "alignmentEnd"] if c in df.columns]
    df = df[keep].copy()
    df["clade"] = df["clade"].map(norm)
    return df


def summarise(report, virus, group_cols):
    """Build per-tool outcome counts and rates over the given grouping."""
    records = []
    for method, prefix in (("nextclade", "nextclade"), ("RIVM", "rivm")):
        outcome_col = f"{prefix}_outcome"
        self_col = f"{prefix}_self_outcome"
        for keys, group in report.groupby(group_cols, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            scored = group[group[outcome_col] != "no_truth"]
            counts = scored[outcome_col].value_counts()
            n_truth = len(scored)
            n_assigned = int(sum(counts.get(o, 0) for o in OUTCOMES if o != "unassigned"
                                 and o != "wrong_type"))
            n_correct = int(sum(counts.get(o, 0) for o in CORRECTISH))

            self_scored = group[group[self_col] != "no_truth"]
            n_self = len(self_scored)
            n_self_ok = int(self_scored[self_col].isin(CORRECTISH).sum())

            row = dict(zip(group_cols, keys))
            row.update({
                "virus": virus,
                "method": method,
                "n_fragments": len(group),
                "n_with_truth": n_truth,
                "n_assigned": n_assigned,
                "n_correct": n_correct,
                "n_self_comparable": n_self,
                "n_self_consistent": n_self_ok,
            })
            for outcome in OUTCOMES:
                row[f"n_{outcome}"] = int(counts.get(outcome, 0))

            row["pct_assigned"] = 100 * n_assigned / n_truth if n_truth else float("nan")
            row["pct_correct"] = 100 * n_correct / n_truth if n_truth else float("nan")
            row["pct_correct_given_assigned"] = (
                100 * n_correct / n_assigned if n_assigned else float("nan"))
            row["pct_self_consistent"] = 100 * n_self_ok / n_self if n_self else float("nan")

            row["assigned_lo"], row["assigned_hi"] = wilson(n_assigned, n_truth)
            row["correct_lo"], row["correct_hi"] = wilson(n_correct, n_truth)
            row["correct_given_assigned_lo"], row["correct_given_assigned_hi"] = wilson(
                n_correct, n_assigned)
            records.append(row)

    ordered = group_cols + ["virus", "method", "n_fragments", "n_with_truth", "n_assigned",
                            "n_correct"] + [f"n_{o}" for o in OUTCOMES] + [
        "n_self_comparable", "n_self_consistent", "pct_assigned", "pct_correct",
        "pct_correct_given_assigned", "pct_self_consistent", "assigned_lo", "assigned_hi",
        "correct_lo", "correct_hi", "correct_given_assigned_lo", "correct_given_assigned_hi"]
    return pd.DataFrame(records)[ordered]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fragment-metadata", required=True,
                        help="TSV from generate_test_sequences.py (one row per fragment)")
    parser.add_argument("--nextclade", required=True, help="nextclade TSV for the fragments")
    parser.add_argument("--nextclade-wg", required=False,
                        help="nextclade TSV for the parent whole-genome sequences")
    parser.add_argument("--rivm", required=True, help="RIVM typing tool CSV export")
    parser.add_argument("--clade-name-map", required=False, help="Clade nomenclature map TSV")
    parser.add_argument("--virus", required=True, help="Expected virus/type name, e.g. EV-A71")
    parser.add_argument("--recombinant-clades", default="",
                        help="Comma-separated clades that denote recombinant forms")
    parser.add_argument("--max-missing-rivm", type=float, default=0.1,
                        help="Fail if more than this fraction of fragments are absent from the "
                             "RIVM export (catches a stale rivm_results.csv)")
    parser.add_argument("--report", required=True, help="Output per-fragment TSV")
    parser.add_argument("--summary", required=True, help="Output summary TSV, by region and length")
    parser.add_argument("--summary-global", required=False, help="Output roll-up summary TSV")
    args = parser.parse_args()

    virus = args.virus
    recombinant_clades = [c for c in args.recombinant_clades.split(",") if c]
    name_map = load_name_map(args.clade_name_map)
    unmapped = {}

    # keep_default_na=False: the metadata writes a literal "NA" for missing truths,
    # which pandas would otherwise reparse as NaN
    meta = pd.read_csv(args.fragment_metadata, sep="\t", keep_default_na=False)
    meta["nextstrain_clade"] = meta["nextstrain_clade"].map(norm)
    meta["rivm_wg_clade"] = meta["rivm_wg_clade"].map(norm)

    # --- Nextclade on the fragments -------------------------------------------------
    nc = read_nextclade(args.nextclade).rename(columns={"seqName": "fragment_id",
                                                        "clade": "nextclade_clade"})
    report = meta.merge(nc, on="fragment_id", how="left")
    missing_nc = report["nextclade_clade"].isna().sum()
    if missing_nc:
        sys.exit(f"ERROR: {missing_nc} fragments have no row in {args.nextclade}. "
                 "The nextclade run and the fragment set are out of sync.")

    # --- Nextclade on the parent whole genomes (self-consistency baseline) -----------
    if args.nextclade_wg:
        wg = read_nextclade(args.nextclade_wg)[["seqName", "clade"]].rename(
            columns={"seqName": "accession", "clade": "nextclade_wg_clade"})
        report = report.merge(wg, on="accession", how="left")
    else:
        report["nextclade_wg_clade"] = NA
    report["nextclade_wg_clade"] = report["nextclade_wg_clade"].map(norm)

    # --- RIVM on the fragments ------------------------------------------------------
    rivm = pd.read_csv(args.rivm)
    rivm = rivm.rename(columns={"name": "fragment_id",
                                "VP1 subgenogroup": "rivm_clade",
                                "type": "rivm_type"})
    rivm["fragment_id"] = rivm["fragment_id"].astype(str).str.strip()
    rivm = rivm[["fragment_id", "rivm_clade", "rivm_type"]].drop_duplicates("fragment_id")
    report = report.merge(rivm, on="fragment_id", how="left")

    # A stale testing/rivm_results.csv is the most likely failure mode here: RIVM is a
    # manual web upload with no producing rule, so a regenerated fragment set silently
    # matches nothing and RIVM looks like it assigned zero fragments - a spuriously
    # strong result for Nextclade. Fail instead of reporting it.
    n_absent = int((~report["fragment_id"].isin(rivm["fragment_id"])).sum())
    if n_absent > args.max_missing_rivm * len(report):
        sys.exit(
            f"ERROR: {n_absent}/{len(report)} fragments are absent from {args.rivm}.\n"
            "This usually means the RIVM export is stale - regenerate the fragments, "
            "re-upload test_out/fragments.fasta to the RIVM typing tool and replace the CSV."
        )
    if n_absent:
        print(f"WARNING: {n_absent} fragments absent from the RIVM export, scored as unassigned",
              file=sys.stderr)

    report["rivm_clade"] = report["rivm_clade"].map(norm)
    report["rivm_type"] = report["rivm_type"].map(norm)

    # --- Nomenclature mapping -------------------------------------------------------
    for col, tool in (("nextclade_clade", "nextclade"), ("nextclade_wg_clade", "nextclade"),
                      ("rivm_clade", "rivm"), ("rivm_wg_clade", "rivm")):
        report[col] = apply_name_map(report[col], tool, name_map, unmapped)
    for tool, names in unmapped.items():
        print(f"NOTE: {tool} names passed through unmapped: {sorted(names)}", file=sys.stderr)

    # --- Outcomes -------------------------------------------------------------------
    nc_type = pd.Series(virus, index=report.index)  # single-pathogen dataset: no typing step
    report["nextclade_outcome"] = [
        classify(c, t, ct, virus) for c, t, ct
        in zip(report["nextclade_clade"], report["nextstrain_clade"], nc_type)]
    report["rivm_outcome"] = [
        classify(c, t, ct, virus) for c, t, ct
        in zip(report["rivm_clade"], report["nextstrain_clade"], report["rivm_type"])]

    # self-consistency: each tool against its own whole-genome call
    report["nextclade_self_outcome"] = [
        classify(c, t, ct, virus) for c, t, ct
        in zip(report["nextclade_clade"], report["nextclade_wg_clade"], nc_type)]
    report["rivm_self_outcome"] = [
        classify(c, t, ct, virus) for c, t, ct
        in zip(report["rivm_clade"], report["rivm_wg_clade"], report["rivm_type"])]

    report["recombinant_truth"] = report["nextstrain_clade"].isin(recombinant_clades).astype(int)
    report["nextclade_correct"] = report["nextclade_outcome"].isin(CORRECTISH).astype(int)
    report["rivm_correct"] = report["rivm_outcome"].isin(CORRECTISH).astype(int)

    report.sort_values(["gene", "length", "replicate"], inplace=True)
    report.to_csv(args.report, sep="\t", index=False)
    print(f"Wrote {len(report)} rows to {args.report}")

    # The matched arm exists so cross-region comparisons are not confounded by the
    # regions drawing on different parent populations; pooling the arms would undo that.
    group_cols = ["arm", "gene", "length"] if "arm" in report.columns else ["gene", "length"]
    summary = summarise(report, virus, group_cols)
    summary.to_csv(args.summary, sep="\t", index=False)
    print(f"Wrote {len(summary)} rows to {args.summary}")

    if args.summary_global:
        report["_all"] = "all"
        arm_cols = ["arm"] if "arm" in report.columns else []
        rollup = pd.concat([
            summarise(report, virus, arm_cols + ["gene"]),
            summarise(report, virus, arm_cols + ["_all"]).rename(columns={"_all": "gene"}),
        ], ignore_index=True)
        rollup.to_csv(args.summary_global, sep="\t", index=False)
        print(f"Wrote {len(rollup)} rows to {args.summary_global}")


if __name__ == "__main__":
    main()
