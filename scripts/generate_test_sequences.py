# scripts/generate_test_sequences.py
"""
Generate all test sequences: fragments, recombinants, non-EV species controls.
"""
import random
import argparse
from pathlib import Path
from Bio import SeqIO, Entrez
import pandas as pd
import time
import sys
import os

def fetch_sequences_from_entrez(taxid, virus_name, num_seqs=5, email=None, api_key=None, min_length=7000):
    """
    Fetch random sequences from NCBI Entrez for a given taxon ID.
    Excludes sequences matching the dataset virus name.
    Only fetches complete, full-length genomes.
    
    Args:
        taxid: NCBI taxonomy ID for the species group
        virus_name: Name of the dataset virus to exclude (e.g., "EV-D68", "EV-D71")
        num_seqs: Number of sequences to fetch (default 5)
        email: Email for NCBI Entrez access
        api_key: Optional NCBI API key for faster access
        min_length: Minimum sequence length in bp (default 7000)
    
    Returns:
        List of (accession, virus_short, sequence) tuples
    """
    if email:
        Entrez.email = email
    else:
        from dotenv import load_dotenv, find_dotenv
        import os
        
        load_dotenv(find_dotenv())
        Entrez.email = os.environ.get("EMAIL")

    if api_key:
        Entrez.api_key = api_key
    
    print(f"Fetching {num_seqs} complete sequences from taxid {taxid}, excluding {virus_name}...", file=sys.stderr)
    
    # Search for complete genomes in the taxon
    search_term = (
        f'txid{taxid}[Organism] AND '
        f'(complete genome[Title] OR complete sequence[Title]) AND '
        f'biomol_genomic[PROP]'
    )
    
    try:
        # First, find out how many records match so we don't silently truncate
        # the pool: esearch's default ordering is newest-submission-first, so a
        # small retmax can bury rarer/older serotypes behind hundreds of recent
        # accessions and they'd never be sampled at all.
        count_handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=0)
        total_count = int(Entrez.read(count_handle)["Count"])
        count_handle.close()

        handle = Entrez.esearch(
            db="nucleotide",
            term=search_term,
            retmax=max(total_count, 1),  # fetch the full matching set, not just the newest 100
            idtype="acc"
        )
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            print(f"No complete sequences found for taxid {taxid}", file=sys.stderr)
            return []
        
        all_accs = record["IdList"]
        print(f"Found {len(all_accs)} complete sequences, fetching details...", file=sys.stderr)
        
        fetched_seqs = []
        attempted = 0
        seen_viruses = set()  # Track viruses we've already fetched
        max_attempts = len(all_accs)
        
        while len(fetched_seqs) < num_seqs and attempted < max_attempts:
            # Pick random accession
            if not all_accs:
                break
            
            acc = all_accs.pop(random.randint(0, len(all_accs) - 1))
            attempted += 1
            
            try:
                # Fetch GenBank record
                handle = Entrez.efetch(db="nucleotide", id=acc, rettype="gb", retmode="text")
                seq_record = SeqIO.read(handle, "genbank")
                handle.close()
               
                # Check sequence length
                seq_len = len(seq_record.seq)
                if seq_len < min_length:
                    continue
                
                # Extract virus info from description or organism
                organism = seq_record.annotations.get("organism", "")
                description = seq_record.description.upper()
                virus_short = seq_record.annotations.get("source", organism)
                # source is often formatted like "enterovirus D68 (EV-D68)" - prefer the
                # parenthetical short name when present, otherwise use it as-is (many
                # records, e.g. generic "Enterovirus D", have no parenthetical at all)
                if "(" in virus_short and ")" in virus_short:
                    virus_short = virus_short.split("(")[1].split(")")[0]

                # Skip if we've already fetched this virus species
                if organism in seen_viruses:
                    continue
                
                # Verify it's complete
                if "COMPLETE" not in description:
                    continue
                
                # Skip if it matches our dataset virus
                if virus_name.lower() in organism.lower() or virus_name.lower() in description.lower():
                    continue
                               
                fetched_seqs.append((acc, virus_short, str(seq_record.seq)))
                seen_viruses.add(organism)
                print(f"✓ {acc} | {virus_short} ({seq_len} bp, complete)", file=sys.stderr)
                
                # Rate limiting
                time.sleep(0.35)  # 3 req/sec without API key
                
            except Exception as e:
                continue
        
        print(f"Successfully fetched {len(fetched_seqs)}/{num_seqs} sequences", file=sys.stderr)
        return fetched_seqs
        
    except Exception as e:
        print(f"Error searching NCBI: {e}", file=sys.stderr)
        return []


def write_entrez_sequences(sequences, output_file):
    """Write fetched sequences to FASTA with formatted headers."""
    with open(output_file, "w") as f:
        for accession, virus_short, seq in sequences:
            header = f"{accession} | {virus_short}"
            f.write(f">{header}\n{seq}\n")
    return len(sequences)

def eligible_seqs(records, min_length):
    """Filter records by minimum length."""
    return [r for r in records if len(r) >= min_length]

def parse_gff_gene_coords(gff_file):
    """Parse gene/CDS coordinates from a GFF3 file into 0-based half-open (start, end) tuples."""
    gene_coords = {}
    if not gff_file or not gff_file.exists():
        return gene_coords
    with open(gff_file) as gff:
        for line in gff:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in ("gene", "CDS"):
                continue
            start, end, attributes = parts[3], parts[4], parts[8]
            attrs = dict(item.split("=", 1) for item in attributes.split(";") if "=" in item)
            gene_name = attrs.get("Name") or attrs.get("gene") or attrs.get("ID")
            if gene_name:
                gene_coords[gene_name] = (int(start) - 1, int(end))
    return gene_coords


# Default fragment design: log-ish length spacing, denser below 1000nt where the
# tools actually diverge, x 3 regions x DEFAULT_REPLICATES = a set small enough to
# push through the RIVM web tool in one sitting.
DEFAULT_LENGTHS = [100, 200, 300, 500, 700, 1000, 1500, 2500]
DEFAULT_REPLICATES = 10
GENOME_REGION = "genome"

# The RIVM typing tool truncates sequence names to 30 characters, so fragment
# headers carry an opaque id only and all metadata lives in a sidecar TSV.
MAX_FRAGMENT_ID_LEN = 30


def random_fragment(seq, length, max_tries=10):
    """Pick a random `length`-nt window of `seq`, avoiding gap/N-heavy windows.

    Returns (start, end, subsequence) with coordinates relative to `seq`, or None
    if `seq` is too short. Windows that are more than 10% gaps/Ns are retried, so a
    fragment is not scored as a tool failure when it is really just missing data.
    """
    if len(seq) <= length:
        return None
    best = None
    for _ in range(max_tries):
        start = random.randint(0, len(seq) - length)
        sub = seq[start:start + length]
        ambiguous = sub.count("-") + sub.count("N") + sub.count("n")
        if best is None or ambiguous < best[0]:
            best = (ambiguous, start, sub)
        if ambiguous <= 0.1 * length:
            break
    _, start, sub = best
    return start, start + length, sub


def stratified_picks(records_by_clade, n, prefer=None):
    """Pick `n` records, round-robin over clades so rare clades are represented.

    `random.choice` over the whole pool would let the few heavily-sequenced clades
    (C4, B5) supply almost every replicate, which makes the benchmark a measurement
    of those clades rather than of the dataset.

    Clade order is shuffled per call: iterating a sorted list meant that whenever
    n was smaller than the number of clades, only the alphabetically-first clades
    were ever sampled.

    `prefer` is an optional predicate; within a clade, records satisfying it are
    drawn first. It is used to favour parents carrying both ground truths, so the
    shared-truth metric keeps as large a denominator as possible.
    """
    clades = list(records_by_clade)
    if not clades:
        return []
    random.shuffle(clades)
    picks, used = [], set()
    exhausted = 0
    i = 0
    while len(picks) < n and exhausted < len(clades):
        clade = clades[i % len(clades)]
        pool = [r for r in records_by_clade[clade] if r not in used]
        if pool:
            if prefer:
                preferred = [r for r in pool if prefer(r)]
                pool = preferred or pool
            choice = random.choice(pool)
            picks.append(choice)
            used.add(choice)
            exhausted = 0
        else:
            exhausted += 1
        i += 1
    return picks


def aligned_span(gapped, start, end):
    """Column indices in [start, end) where this aligned record has a real base.

    Working in alignment (reference) coordinates is what lets partial sequences - VP1
    amplicons, which are the bulk of EV-A71 GenBank records - contribute fragments at
    all: their covered region is read off the alignment instead of being assumed.
    """
    return [i for i in range(start, min(end, len(gapped)))
            if gapped[i] not in "-." and gapped[i] not in "Nn"]


def cut_aligned_fragment(gapped, columns, length):
    """Cut `length` real bases from an aligned record, within the given columns.

    Returns (ref_start, ref_end, ungapped_fragment) in reference coordinates, or None
    if the record does not carry `length` bases in that region.
    """
    if len(columns) < length:
        return None
    offset = random.randint(0, len(columns) - length)
    chosen = columns[offset:offset + length]
    fragment = "".join(gapped[i] for i in chosen)
    return chosen[0], chosen[-1] + 1, fragment


def generate_fragments(aligned_file, rivm_file, clades_file, nextstrain_file, output_file,
                       metadata_file, lengths=None, genes=None, gff_file=None,
                       replicates=DEFAULT_REPLICATES, matched_replicates=0):
    """Generate replicated, clade-stratified fragment test sequences.

    Parents are any sequence with a clade call from Nextstrain or RIVM that is not in
    the Nextclade reference tree, taken from a reference alignment so that partial
    sequences (mostly VP1 amplicons) can contribute fragments in the regions they
    actually cover.

    Two arms are produced. The `main` arm lets every region draw from all eligible
    parents, maximising power per region. The `matched` arm restricts all regions to
    parents covering every region, so that comparisons *across* regions are not
    confounded by the parent populations differing between them.

    Writes FASTA records named `{accession}_frag00001` and a sidecar TSV holding the
    parent accession, region, replicate, arm, true reference coordinates and both
    ground truths for each fragment.
    """
    if lengths is None:
        lengths = DEFAULT_LENGTHS
    if genes is None:
        genes = ["VP1", "3D"]

    gene_coords = parse_gff_gene_coords(gff_file)

    aligned = {r.id: str(r.seq) for r in SeqIO.parse(aligned_file, "fasta")}
    aln_len = max((len(s) for s in aligned.values()), default=0)
    print(f"{len(aligned)} aligned candidate sequences (alignment length {aln_len})",
          file=sys.stderr)

    # --- ground truths -------------------------------------------------------------
    nextstrain_df = pd.read_csv(nextstrain_file, sep="\t")
    ns_map = {k: v for k, v in nextstrain_df.set_index("accession")["clade_membership"].items()
              if pd.notna(v)}

    rivm_map = {}
    if rivm_file:
        rivm = pd.read_csv(rivm_file, sep=",")
        rivm.dropna(axis=0, inplace=True)
        if "accession" in rivm:
            rivm_map = rivm.set_index("accession")["RIVM"].to_dict()
        elif "name" in rivm:
            rivm_map = rivm.set_index("name")["VP1 subgenogroup"].to_dict()
        else:
            print("neither accession nor name in ", rivm_file, ". Please check!")

    # Exclude sequences that are in the Nextclade reference tree: testing on them would
    # measure Nextclade against its own training data. NB `in` on a pandas Series tests
    # the index, not the values, so this must go through a set.
    tree_accessions = set(pd.read_csv(clades_file, sep="\t").accession)

    pool = [acc for acc in aligned
            if acc not in tree_accessions and (acc in ns_map or acc in rivm_map)]
    print(f"{len(pool)} candidate parents with a clade call, after excluding "
          f"{len(tree_accessions)} reference-tree sequences", file=sys.stderr)

    # --- region definitions ---------------------------------------------------------
    regions = [(g, gene_coords[g]) for g in genes if g in gene_coords]
    for g in genes:
        if g not in gene_coords:
            print(f"Gene {g} not found in GFF3 annotation, skipping.", file=sys.stderr)
    regions.append((GENOME_REGION, (0, aln_len)))

    # Precompute each parent's covered columns per region, once.
    coverage = {region: {acc: aligned_span(aligned[acc], start, end) for acc in pool}
                for region, (start, end) in regions}

    # Parents usable in the matched arm cover every region at the largest length asked of it.
    matched_pool = [acc for acc in pool
                    if all(len(coverage[region][acc]) >= min(max(lengths), end - start)
                           for region, (start, end) in regions)]
    print(f"{len(matched_pool)} parents cover every region (matched arm)", file=sys.stderr)

    def group_by_clade(accessions):
        grouped = {}
        for acc in accessions:
            clade = ns_map.get(acc) or rivm_map.get(acc)
            if clade and str(clade) != "nan":
                grouped.setdefault(str(clade), []).append(acc)
        return grouped

    rows = []
    counter = 0
    with open(output_file, "w") as out_handle:
        for arm, arm_pool, arm_reps in (("main", pool, replicates),
                                        ("matched", matched_pool, matched_replicates)):
            if not arm_reps or not arm_pool:
                continue
            for length in lengths:
                for region, (start, end) in regions:
                    eligible = [a for a in arm_pool if len(coverage[region][a]) >= length]
                    if not eligible:
                        continue
                    both_truths = lambda a: a in ns_map and a in rivm_map
                    picks = stratified_picks(group_by_clade(eligible), arm_reps,
                                             prefer=both_truths)
                    for replicate, acc in enumerate(picks, start=1):
                        cut = cut_aligned_fragment(aligned[acc], coverage[region][acc], length)
                        if cut is None:
                            continue
                        ref_start, ref_end, fragment = cut
                        counter += 1
                        frag_id = f"{acc}_frag{counter:05d}"
                        out_handle.write(f">{frag_id}\n{fragment}\n")
                        rows.append({
                            "fragment_id": frag_id,
                            "accession": acc,
                            "length": length,
                            "gene": region,
                            "arm": arm,
                            "replicate": replicate,
                            # reference coordinates, read off the alignment rather than
                            # inferred later from Nextclade's own alignment of the fragment
                            "start": ref_start,
                            "end": ref_end,
                            "parent_covered_nt": len(coverage[GENOME_REGION][acc]),
                            "nextstrain_clade": ns_map.get(acc, "NA"),
                            "rivm_wg_clade": rivm_map.get(acc, "NA"),
                        })

    meta = pd.DataFrame(rows).fillna("NA")
    meta["has_both_truths"] = ((meta["nextstrain_clade"] != "NA")
                               & (meta["rivm_wg_clade"] != "NA")).astype(int)
    ids = meta["fragment_id"]
    assert ids.is_unique, "fragment ids are not unique"
    too_long = ids[ids.str.len() > MAX_FRAGMENT_ID_LEN]
    assert too_long.empty, (
        f"fragment ids exceed the RIVM {MAX_FRAGMENT_ID_LEN}-char name limit: {list(too_long)}"
    )
    meta.to_csv(metadata_file, sep="\t", index=False)
    print(f"Wrote {len(meta)} fragments to {output_file} and metadata to {metadata_file}",
          file=sys.stderr)

    return len(meta)

def generate_recombinants(sequences_file, metadata, clades_file, ev_file, 
                          output_file, inter_count=10, intra_count=10, min_length=3500):
    """Generate inter- and intra-recombinant test sequences."""
    seqs = list(SeqIO.parse(sequences_file, "fasta"))
    ns_ids = set(pd.read_csv(metadata, sep="\t").accession)
    seqs = eligible_seqs([r for r in seqs if r.id in ns_ids], min_length)
    
    clade_map = pd.read_csv(clades_file, sep="\t").set_index("accession")["clade"].to_dict()
    clade2seqs = {}
    for r in seqs:
        clade = clade_map.get(r.id, "NA")
        if clade != "NA":
            clade2seqs.setdefault(clade, []).append(r)
    
    clades = [c for c in clade2seqs if len(clade2seqs[c]) > 0]
    ev_seqs = eligible_seqs(list(SeqIO.parse(ev_file, "fasta")), min_length)
    
    recombinants = []
    
    # Intra-typic recombinants (between clades)
    for i in range(intra_count):
        if len(clades) < 2:
            break
        c1, c2 = random.sample(clades, 2)
        p1, p2 = random.choice(clade2seqs[c1]), random.choice(clade2seqs[c2])
        minlen = min(len(p1.seq), len(p2.seq))
        if minlen < min_length:
            continue
        breakpoint = random.randint(1, minlen - 1)
        header = f"intra_{p1.id}_{c1}_{breakpoint}_{p2.id}_{c2}"
        recomb_seq = str(p1.seq[:breakpoint]) + str(p2.seq[breakpoint:])
        recombinants.append((header, recomb_seq))
    
    # Inter-typic recombinants (target virus x related species)
    for j in range(inter_count):
        p1 = random.choice(seqs)
        p2 = random.choice(ev_seqs)
        minlen = min(len(p1.seq), len(p2.seq))
        if minlen < min_length:
            continue
        breakpoint = random.randint(1, minlen - 1)
        header = f"inter_{p1.id}_ virus_{breakpoint}_{p2.id}_{p2.description.split('|')[1].strip()}"
        recomb_seq = str(p1.seq[:breakpoint]) + str(p2.seq[breakpoint:])
        recombinants.append((header, recomb_seq))
    
    # Write recombinants
    with open(output_file, "w") as f:
        for header, seq in recombinants:
            f.write(f">{header}\n{seq}\n")
    
    return len(recombinants)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test sequences")
    parser.add_argument("--sequences", required=True, help="Input sequences file")
    parser.add_argument("--nextstrain", required=True, help="Nextstrain metadata TSV")
    parser.add_argument("--clades", required=True, help="Clades metadata TSV")
    parser.add_argument("--rivm", required=False, help = "RIVM Clade Assignment")
    parser.add_argument("--output-fragments", required=True, help="Output fragments FASTA")
    parser.add_argument("--output-fragment-metadata", required=False,
                        help="Output fragment metadata TSV (default: alongside the fragments FASTA)")
    parser.add_argument("--lengths", required=False, nargs="+", type=int, default=None,
                        help=f"Fragment lengths to generate (default: {DEFAULT_LENGTHS})")
    parser.add_argument("--replicates", required=False, type=int, default=DEFAULT_REPLICATES,
                        help="Fragments per length x region cell in the main arm")
    parser.add_argument("--matched-replicates", required=False, type=int, default=0,
                        help="Fragments per cell in the matched arm, where every region draws "
                             "from parents covering all regions (0 disables the arm)")
    parser.add_argument("--aligned", required=False,
                        help="Reference alignment of candidate parent sequences; required for "
                             "fragment generation so partial sequences can be placed")
    parser.add_argument("--output-recombinants", required=True, help="Output recombinants FASTA")
    parser.add_argument("--output-ev", required=False, help="Output EV species sequences FASTA (if fetching from NCBI)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ev", required=False, help="EV species sequences file (skip Entrez if provided)")
    parser.add_argument("--taxid", type=str, required=False, help="NCBI Taxon ID for EV species (if not provided as file)")
    parser.add_argument("--virus", type=str, required=False, default="EV", help="Name of dataset virus to exclude from Entrez fetch")
    parser.add_argument("--genes", required=False, nargs='+', default=None, help="Genes for which fragments are created")
    parser.add_argument("--email", required=False, help="Email for NCBI Entrez access")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    Path(args.output_fragments).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_recombinants).parent.mkdir(parents=True, exist_ok=True)

    # Handle EV species sequences: either use provided file or fetch from NCBI
    if args.ev:
        ev_file = args.ev
        print(f"Using provided EV species file: {ev_file}", file=sys.stderr)
    elif args.taxid:
        ev_file = args.output_ev or Path(args.output_fragments).parent / "EV_fetched.fasta"
        fetched = fetch_sequences_from_entrez(
            args.taxid,
            args.virus,
            num_seqs=10,
            email=args.email,
            min_length=7000
        )
        
        if fetched:
            num_ev = write_entrez_sequences(fetched, ev_file)
            print(f"Fetched and saved {num_ev} EV species sequences to {ev_file}", file=sys.stderr)
        else:
            print("Failed to fetch EV species sequences, skipping recombinant generation", file=sys.stderr)
            ev_file = None
    else:
        print("Error: Either --ev file or --taxid must be provided", file=sys.stderr)
        sys.exit(1)

    gff_file = Path("dataset/genome_annotation.gff3")

    if args.rivm:
        rivm = Path(args.rivm)
    else:
        rivm = None

    fragment_metadata = (
        Path(args.output_fragment_metadata) if args.output_fragment_metadata
        else Path(args.output_fragments).with_name("fragment_metadata.tsv")
    )

    if not args.aligned:
        sys.exit("--aligned is required: fragments are cut from a reference alignment so that "
                 "partial sequences can be placed in reference coordinates")

    n_frags = generate_fragments(args.aligned, rivm, args.clades, args.nextstrain,
                                 args.output_fragments, fragment_metadata,
                                 lengths=args.lengths, genes=args.genes, gff_file=gff_file,
                                 replicates=args.replicates,
                                 matched_replicates=args.matched_replicates)
    n_recomb = generate_recombinants(args.sequences, args.nextstrain, args.clades, ev_file, args.output_recombinants)
    
    print(f"Generated {n_frags} fragments and {n_recomb} recombinants")