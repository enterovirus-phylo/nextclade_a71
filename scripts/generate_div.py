## Python script to generate mutated copies of input sequences at given p-distances.
# run with
# python scripts/generate_div.py -i input.fasta -o out/mutated.fasta -d 0.01 0.05 -n 3 [-ka 2.0] [-s 1]
# Non-ACGT characters (N, gaps, IUPAC ambiguity codes) are kept unchanged at their positions and are never
# mutated; target p-distances are computed over the called (A/C/G/T) sites only.
# -d sets the divergence levels.
# -n sets how many copies to make per sequence and level.
# -ka sets the transition/transversion ratio.
# -s sets the random seed.


import numpy as np
import os
from collections import Counter
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import argparse
import logging

# Encoding: A=0, C=1, G=2, T=3; any other character (N, gap, ambiguity code) is encoded as UNCALLED
MAPPING = {"A": 0, "C": 1, "G": 2, "T": 3}
BASES = np.array(["A", "C", "G", "T"])
UNCALLED = -1

# ==============================================================
# 1. Command-line argument parser
# ==============================================================

def get_args(argv=None):
    """Parse command-line arguments. argv=None reads sys.argv; pass a list to call from Python."""
    parser = argparse.ArgumentParser(description="Create mutated copies of input sequences at given p-distances.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file with one or more sequences. Non-ACGT characters are kept and never mutated.")
    parser.add_argument("-o", "--output", required=True, help="Output FASTA file with the mutated sequences.")
    parser.add_argument("-d", "--div", type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.05, 0.10, 0.20], help="Target p-distance(s) of each mutated copy from its input sequence. Default: 0.005 0.01 0.02 0.05 0.10 0.20.")
    parser.add_argument("-n", "--n-copies", type=int, default=1, help="Number of mutated copies per input sequence and divergence level. Default: 1.")
    parser.add_argument("-ka", "--kappa", type=float, default=1.0, help="Transition/transversion rate ratio (1.0 = all three alternative bases equally likely). Default: 1.0.")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed. Default: 1.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args(argv)

    if len(set(args.div)) != len(args.div):
        parser.error("--div must not contain duplicate values (they would produce duplicate sequence IDs).")
    if any(not (0 <= d < 0.75) for d in args.div):
        parser.error("--div values must be >= 0 and < 0.75 (0.75 is the p-distance between unrelated random sequences).")
    if args.n_copies < 1:
        parser.error("--n-copies must be at least 1.")
    if args.kappa <= 0:
        parser.error("--kappa must be > 0.")

    return args


# ==============================================================
# 2. Sequence loading, encoding and writing
# ==============================================================

def load_sequences(fasta_path, logger=None):
    """
    Load sequences from a FASTA file and encode them as integer arrays (A=0, C=1, G=2, T=3, other=UNCALLED).

    Sequences are upper-cased. Non-ACGT characters (N, gaps, ambiguity codes) are allowed: they are encoded
    as UNCALLED and the original characters are kept so they can be restored unchanged on output.
    Returns a dictionary mapping sequence ID -> (encoded np.ndarray, original upper-case string,
    original description without the leading ID).
    """
    logger = logger or logging.getLogger(__name__)
    seqs = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        if record.id in seqs:
            raise ValueError(f"Sequence ID '{record.id}' is not unique.")

        encoded = np.array([MAPPING.get(base, UNCALLED) for base in seq], dtype=np.int8)
        n_called = int(np.count_nonzero(encoded != UNCALLED))
        if n_called == 0:
            raise ValueError(f"Sequence '{record.id}' has no A/C/G/T sites to mutate.")

        n_uncalled = len(seq) - n_called
        if n_uncalled:
            chars = ", ".join(f"{c}={n}" for c, n in sorted(Counter(c for c in seq if c not in MAPPING).items()))
            logger.info(f"[INFO] '{record.id}': {n_uncalled}/{len(seq)} non-ACGT site(s) kept unchanged ({chars})")

        # Biopython's description repeats the ID at the start; keep only the part after it
        description = record.description
        if description.startswith(record.id):
            description = description[len(record.id):].strip()

        seqs[record.id] = (encoded, seq, description)

    if not seqs:
        raise ValueError(f"No sequences found in {fasta_path}.")
    return seqs


def decode_sequence(int_seq, template):
    """
    Convert an encoded sequence back to a string. Called sites (0-3) become A/C/G/T;
    UNCALLED sites take the original character from template (same length as int_seq).
    """
    out = np.array(list(template))
    called = int_seq != UNCALLED
    out[called] = BASES[int_seq[called]]
    return "".join(out)


def format_level(value):
    """Format a divergence level for IDs, e.g. 0.05 -> '0.05', 0.1 -> '0.1', 0 -> '0'."""
    return f"{value:g}"


# ==============================================================
# 3. Substitutions
# ==============================================================

def n_sites(div, length):
    """Number of sites to mutate for a target p-distance: div * length, rounded half up (not Python's round-half-to-even)."""
    return int(np.floor(div * length + 0.5 + 1e-9))   # Small tolerance so e.g. 0.005 * 7500 = 37.4999... still rounds to 38


def mutate_sites(seq, sites, rng, kappa):
    """
    Return a copy of seq with a substitution at each position in sites (every substitution changes the base).

    With the encoding A=0, C=1, G=2, T=3, adding 2 (mod 4) gives the transition partner (A<->G, C<->T), and
    adding 1 or 3 gives the two transversions. Offsets are drawn with probabilities 1 : kappa : 1.
    """
    new_seq = seq.copy()
    offset_probs = np.array([1.0, kappa, 1.0]) / (kappa + 2)
    offsets = rng.choice([1, 2, 3], size=len(sites), p=offset_probs)
    new_seq[sites] = (seq[sites] + offsets) % 4
    return new_seq


def mutate_exact(seq, div, rng, kappa):
    """
    Mutate exactly n_sites(div, L_called) distinct, randomly chosen called (A/C/G/T) positions of seq,
    where L_called is the number of called sites. UNCALLED sites (N, gaps, ...) are never touched.

    Because sites are drawn without replacement and every substitution changes the base, the p-distance
    between input and output over called sites is exactly n / L_called (no multiple hits or back-mutations).
    Returns the mutated sequence and the number of substituted sites.
    """
    called_sites = np.flatnonzero(seq != UNCALLED)
    n_sub = n_sites(div, len(called_sites))
    sites = rng.choice(called_sites, size=n_sub, replace=False)
    return mutate_sites(seq, sites, rng, kappa), n_sub


# ==============================================================
# 4. Main pipeline
# ==============================================================

def main(argv=None):
    """Run the pipeline. argv=None reads sys.argv; pass a list, e.g. main(["-i", "in.fasta", "-o", "out.fasta"])."""
    args = get_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")
    logger = logging.getLogger(__name__)

    seqs = load_sequences(args.input, logger)
    logger.info(f"[INFO] Loaded {len(seqs)} sequence(s) from {args.input}")

    rng = np.random.default_rng(args.seed)
    records = []
    # Keep the original ID when each input gives exactly one output; only add a
    # _d<div>_r<copy> suffix when it is needed to keep IDs unique.
    one_copy_per_input = len(args.div) == 1 and args.n_copies == 1
    for seq_id, (seq, template, orig_description) in seqs.items():
        n_called = int(np.count_nonzero(seq != UNCALLED))
        for div in args.div:
            for r in range(args.n_copies):
                mutated, n_sub = mutate_exact(seq, div, rng, args.kappa)
                new_id = seq_id if one_copy_per_input else f"{seq_id}_d{format_level(div)}_r{r + 1}"
                # original description first, mutation info appended
                mutation_info = f"source={seq_id} div={format_level(div)} n_substitutions={n_sub}"
                description = f"{orig_description} {mutation_info}".strip()
                records.append(SeqRecord(Seq(decode_sequence(mutated, template)), id=new_id, description=description))
                logger.debug(f"    {new_id}: {n_sub} substitutions ({n_sub / n_called:.5f} of {n_called} called sites)")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    SeqIO.write(records, args.output, "fasta")
    logger.info(f"[INFO] Wrote {len(records)} mutated sequence(s) to {args.output}")


if __name__ == "__main__":
    main()
