"""Usage:
  nextclade3 run -D out-dataset --output-tsv nextclade.tsv seqs.fasta
  python scripts/hmm_eval.py nextclade.tsv [-o calls.tsv] [--pathogen dataset/pathogen.json] [--all] [--muR 0.2 ...]
  (or as a module: import hmm_eval as H; res = H.run(H.load("nextclade.tsv")) -> {seqName: [(start, end), ...]})

Re-implementation of Nextclade's recombination HMM (PR nextstrain/nextclade#1774) for parameter tuning.

Observations per reference site: 0 = ref (covered, same as parent), 1 = private substitution, 2 = missing
(outside alignment, N/missing range, deletion, non-ACGTN). Two-state Viterbi in log space, then the call
checks: per-region missing fraction / min length, coverage (maxRecombinantFraction), and the rest-of-sequence
private-mutation check (weighted total >= (typical+cutoff) * Lm / L drops the sequence's calls).
Placement-masked sites are not modelled (not exposed in the 3.25 TSV).

Settings come from a Nextclade pathogen.json (default: dataset/pathogen.json of this repo), read at import:
  - reference length L from the reference FASTA named in files.reference (same directory as pathogen.json)
  - qc.privateMutations: typical, cutoff, weight{Reversion,Labeled,Unlabeled}Substitutions
  - recombination: gamma, muW, muR, minRegionLength, maxRegionMissingFraction, maxRecombinantFraction
Call configure(path) to switch to another pathogen.json. Explicit arguments to run()/call() override it.
"""
import csv, json, re
from pathlib import Path
import numpy as np

MUT = re.compile(r"[A-Z](\d+)[A-Z]")
DEFAULT_PATHOGEN_JSON = Path(__file__).resolve().parent.parent / "dataset" / "pathogen.json"

# Filled by configure()
L = None
W_REV = W_LAB = W_UNL = None
TYPICAL = CUTOFF = None
REC = {}  # recombination block of pathogen.json


def configure(pathogen_json=DEFAULT_PATHOGEN_JSON):
    """Load reference length, private-mutation QC settings and recombination settings from pathogen.json."""
    global L, W_REV, W_LAB, W_UNL, TYPICAL, CUTOFF, REC
    pathogen_json = Path(pathogen_json)
    cfg = json.loads(pathogen_json.read_text())

    ref = pathogen_json.parent / cfg["files"]["reference"]
    L = sum(len(line.strip()) for line in open(ref) if not line.startswith(">"))

    pm = cfg["qc"]["privateMutations"]
    TYPICAL, CUTOFF = pm["typical"], pm["cutoff"]
    W_REV = pm["weightReversionSubstitutions"]
    W_LAB = pm["weightLabeledSubstitutions"]
    W_UNL = pm["weightUnlabeledSubstitutions"]

    REC = cfg.get("recombination", {})
    missing = [k for k in ("gamma", "muW", "muR") if k not in REC]
    if missing:
        raise KeyError(f"{pathogen_json}: recombination block lacks {missing}; set them explicitly "
                       "(this script does not estimate them from the tree)")
    # Nextclade defaults for the call checks
    REC.setdefault("minRegionLength", 50)
    REC.setdefault("maxRegionMissingFraction", 0.5)
    REC.setdefault("maxRecombinantFraction", 2 / 3)
    return REC


configure()


def ranges(field, strip_char=False):
    out = []
    for tok in filter(None, field.split(",")):
        if strip_char and ":" in tok:
            tok = tok.split(":", 1)[1]
        a, _, b = tok.partition("-")
        out.append((int(a), int(b or a)))
    return out


def load(nc_tsv, names=None):
    """Return list of dict(name, obs(np.int8[L]), w(np.float32[L]) per-site QC weights, alnStart, alnEnd)."""
    recs = []
    for r in csv.DictReader(open(nc_tsv), delimiter="\t"):
        if r["errors"] or not r["alignmentStart"]:
            continue
        if names is not None and r["seqName"] not in names:
            continue
        obs = np.full(L, 2, np.int8)
        s, e = int(r["alignmentStart"]), int(r["alignmentEnd"])
        obs[s - 1:e] = 0
        w = np.zeros(L, np.float32)
        for col, wt in (("privateNucMutations.reversionSubstitutions", W_REV),
                        ("privateNucMutations.labeledSubstitutions", W_LAB),
                        ("privateNucMutations.unlabeledSubstitutions", W_UNL)):
            for m in MUT.finditer(r[col]):
                p = int(m.group(1)) - 1
                obs[p] = 1
                w[p] = wt
        for a, b in ranges(r["missing"]) + ranges(r["deletions"]) + ranges(r["nonACGTNs"], True):
            obs[a - 1:b] = 2
        recs.append(dict(name=r["seqName"], obs=obs, w=w, s=s, e=e, clade=r["clade"],
                         cov=float(r["coverage"] or 0), npriv=int(r["privateNucMutations.totalPrivateSubstitutions"] or 0),
                         qctotal=float(r["qc.privateMutations.total"] or 0)))
    return recs


def viterbi(O, gamma, muW, muR):
    """O: (N, L) int8. Returns (N, L) bool, True = recombinant state."""
    N = O.shape[0]
    eW = np.array([np.log1p(-muW), np.log(muW), 0.0])
    eR = np.array([np.log1p(-muR), np.log(muR), 0.0])
    stay, sw = np.log1p(-gamma), np.log(gamma)
    vW = np.full(N, np.log(0.5)) + eW[O[:, 0]]
    vR = np.full(N, np.log(0.5)) + eR[O[:, 0]]
    bpW = np.zeros((N, L), bool)  # True: came from R
    bpR = np.zeros((N, L), bool)  # True: came from W
    for t in range(1, L):
        ww, rw = vW + stay, vR + sw
        rr, wr = vR + stay, vW + sw
        bpW[:, t] = rw > ww
        bpR[:, t] = wr > rr
        vW = np.maximum(ww, rw) + eW[O[:, t]]
        vR = np.maximum(rr, wr) + eR[O[:, t]]
    path = np.zeros((N, L), bool)
    st = vR > vW
    path[:, L - 1] = st
    for t in range(L - 1, 0, -1):
        st = np.where(st, ~bpR[:, t], bpW[:, t])
        path[:, t - 1] = st
    return path


def regions_of(path_row, obs):
    out = []
    idx = np.flatnonzero(np.diff(np.concatenate([[0], path_row.astype(np.int8), [0]])))
    for a, b in zip(idx[::2], idx[1::2]):  # [a, b)
        cov = np.flatnonzero(obs[a:b] != 2)
        if cov.size == 0:
            continue
        out.append((a + cov[0], a + cov[-1] + 1))
    return out


def call(rec, path_row, minLen=None, maxMiss=None, maxFrac=None):
    minLen = REC["minRegionLength"] if minLen is None else minLen
    maxMiss = REC["maxRegionMissingFraction"] if maxMiss is None else maxMiss
    maxFrac = REC["maxRecombinantFraction"] if maxFrac is None else maxFrac
    obs = rec["obs"]
    regs = []
    for a, b in regions_of(path_row, obs):
        miss = np.count_nonzero(obs[a:b] == 2) / (b - a)
        if miss > maxMiss:
            continue
        if b - a < minLen:
            continue
        regs.append((a, b))
    if not regs:
        return []
    span = sum(b - a for a, b in regs)
    if span / (rec["e"] - rec["s"] + 1) > maxFrac:
        return []
    mask = np.ones(L, bool)
    for a, b in regs:
        mask[a:b] = False
    Lm = np.count_nonzero(mask & (obs != 2))
    if rec["w"][mask].sum() >= (TYPICAL + CUTOFF) * Lm / L:
        return []
    return [(a + 1, b) for a, b in regs]  # 1-based inclusive


def run(recs, gamma=None, muW=None, muR=None, chunk=4000, **kw):
    """Decode and call. HMM parameters and call-check kwargs default to the pathogen.json values."""
    gamma = REC["gamma"] if gamma is None else gamma
    muW = REC["muW"] if muW is None else muW
    muR = REC["muR"] if muR is None else muR
    res = {}
    todo = [r for r in recs if r["npriv"] > 0]
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        P = viterbi(np.stack([r["obs"] for r in part]), gamma, muW, muR)
        for r, row in zip(part, P):
            res[r["name"]] = call(r, row, **kw) if row.any() else []
    for r in recs:
        res.setdefault(r["name"], [])
    return res


def main():
    import argparse, sys

    p = argparse.ArgumentParser(
        description="Call putative recombinant regions from a Nextclade TSV with the recombination HMM. "
                    "Settings come from pathogen.json; the options below override them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("nextclade_tsv", help="Nextclade output TSV (nextclade3 run --output-tsv)")
    p.add_argument("-o", "--output", help="output TSV (default: stdout)")
    p.add_argument("--pathogen", default=str(DEFAULT_PATHOGEN_JSON), help="pathogen.json with the settings")
    p.add_argument("--all", action="store_true", help="also list sequences without a call")
    for opt in ("gamma", "muW", "muR", "minRegionLength", "maxRegionMissingFraction", "maxRecombinantFraction"):
        p.add_argument(f"--{opt}", type=float, help="override the pathogen.json value")
    a = p.parse_args()

    configure(a.pathogen)
    recs = load(a.nextclade_tsv)
    res = run(recs, a.gamma, a.muW, a.muR,
              minLen=a.minRegionLength, maxMiss=a.maxRegionMissingFraction, maxFrac=a.maxRecombinantFraction)

    out = open(a.output, "w", newline="") if a.output else sys.stdout
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    w.writerow(["seqName", "clade", "coverage", "privateSubstitutions", "recombination.totalRegions",
                "recombination.regions", "recombination.regionDensities", "recombination.totalLength",
                "recombination.longestRegion.range", "recombination.longestRegion.length"])
    n_called = 0
    for r in recs:
        regs = res[r["name"]]
        n_called += bool(regs)
        if not regs and not a.all:
            continue
        lens = [b - a_ + 1 for a_, b in regs]
        dens = [np.count_nonzero(r["obs"][a_ - 1:b] == 1) / max(1, np.count_nonzero(r["obs"][a_ - 1:b] != 2))
                for a_, b in regs]
        longest = regs[int(np.argmax(lens))] if regs else None
        w.writerow([r["name"], r["clade"], f'{r["cov"]:.3f}', r["npriv"], len(regs),
                    ",".join(f"{a_}-{b}" for a_, b in regs), ",".join(f"{d:.3f}" for d in dens),
                    sum(lens), f"{longest[0]}-{longest[1]}" if longest else "", max(lens, default="")])
    if a.output:
        out.close()
    print(f"{n_called} of {len(recs)} sequences with recombination calls "
          f"(gamma={a.gamma or REC['gamma']}, muW={a.muW or REC['muW']}, muR={a.muR or REC['muR']})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
