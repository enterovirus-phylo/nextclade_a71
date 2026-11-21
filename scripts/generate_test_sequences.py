# scripts/generate_test_sequences.py
# clades_file="results/clades_metadata.tsv"
# nextstrain_meta_file = "test_out/nextstrain_vp1_metadata.tsv"
# filtered_seq = "test_out/fitered_nextstrain_seq.fasta"
# gff_file = "dataset/genome_annotation.gff3"
# reference_file = "dataset/reference.fasta"
# path_blastdb = "test_out/blast/"
# sequences_file="data/sequences.fasta"
# virus="Enterovirus A71"
# genes="VP1 3D"
# lengths=[100, 150, 200, 300, 500, 700, 1000, 1500, 2500, 3000]
# seed=42
# fragments_out = "test_out/fragments.fasta"

#%%
"""
Generate all test sequences: fragments, recombinants, non-EV species controls.
"""
import random
import argparse
from pathlib import Path
from Bio import SeqIO, Entrez
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys
import os
import subprocess
import generate_div as gd

# Default fragment lengths (used when --lengths is not given)
DEFAULT_LENGTHS = [100, 150, 200, 300, 500, 700, 1000, 1500, 2500, 3000]


## Fetch genotypes from same species group from NCBI Entrez, excluding the dataset virus, to use as negative controls.
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
    if not gff_file or not Path(gff_file).exists():
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


# filter all sequences for accession in nextstrain metadata
def download_nextstrain_tree(virus, output_tree):
    """Download Nextstrain tree for a given genotype, translate it to metadata and save to TSV."""
    import requests
    gt = virus.replace(" ", "/").lower()
    gt = gt.replace("/c", "/C").replace("/d", "/D").replace("/a", "/A").replace("/b", "/B")  # ensure proper case for URL
    url = f"https://nextstrain.org/charon/getDataset?prefix=groups/hodcroftlab/{gt}/vp1"
    print(url)
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_tree, "w") as f:
            f.write(response.text)
        print(f"Downloaded Nextstrain tree for {virus} to {output_tree}", file=sys.stderr)
    else:
        print(f"Failed to download Nextstrain tree for {output_tree}: HTTP {response.status_code}", file=sys.stderr)

    return output_tree

def json_to_meta(virus, nextstrain_meta_file):
    """Convert Nextstrain JSON tree to a tabular format and save to TSV."""
    output_tree = f"test_out/{virus}.tree.json"

    tree = download_nextstrain_tree(virus, output_tree)

    import json
    import pandas as pd
    import sys
    import auspice_tree_to_table as at

    with open(output_tree, "r", encoding="utf-8") as fh:
            tree_json = json.load(fh)
    
    tree = at.json_to_tree(tree_json)

    if nextstrain_meta_file:
        # Collect attributes per node from the tree to export.
        records = []
        attributes = ["clade_membership"]
        include_internal_nodes = False

        for node in tree.find_clades():
            if node.is_terminal() or include_internal_nodes:
                record = {
                    "name": node.name
                }

                for attribute in attributes:
                    if attribute in node.node_attrs:
                        value = node.node_attrs[attribute]
                    elif attribute in node.branch_attrs:
                        value = node.branch_attrs[attribute]
                    else:
                        print(f"Could not find attribute '{attribute}' for node '{node.name}'.", file=sys.stderr)
                        value = None

                    if value is not None:
                        if isinstance(value, dict) and "value" in value:
                            value = value["value"]

                    record[attribute] = value

                record["parent_name"] = getattr(node.parent, "name", "")
                records.append(record)

        # Convert records to a data frame and save as a tab-delimited file.
        df = pd.DataFrame(records)
        df.to_csv(
            nextstrain_meta_file,
            sep="\t",
            header=True,
            index=False,
            columns=["name", "parent_name"] + list(attributes),
        )

        os.remove(output_tree)

## filter sequences for accession in nextstrain metadata
def filter_seq_in_nextstrain(sequences_file, nextstrain_meta_file, filtered_seq):
    """Filter sequences for accessions present in Nextstrain VP1 metadata and save to a new FASTA file."""
    meta = pd.read_csv(nextstrain_meta_file, sep="\t")
    accessions = set(meta["name"])

    filtered_records = []
    for record in SeqIO.parse(sequences_file, "fasta"):
        if record.id in accessions:
            filtered_records.append(record)

    SeqIO.write(filtered_records, filtered_seq, "fasta")

    filtered_count = len(filtered_records)
    print(f"Filtered {filtered_count} sequences present in Nextstrain metadata and saved to {filtered_seq}", file=sys.stderr)

    # length distribution of filtered sequences
    plt.figure(figsize=(10, 6))
    plt.hist([len(record.seq) for record in filtered_records], bins=100, color='darkgrey', edgecolor='black')
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.title("Filtered Sequences Length Distribution")
    plt.savefig("test_out/filtered_sequences_length_distribution.png")
    plt.close()

def random_select(records, n, seed=None):
    """Randomly select n records from a list of SeqRecord objects."""
    if n >= len(records):
        return records
    if seed is not None:
        random.seed(seed)
    return random.sample(records, n)

def run_blast(gff_file, reference_file, genes, filtered_seq, path_blastdb):
    gene_coords = parse_gff_gene_coords(gff_file)

    ref_seq = SeqIO.read(reference_file, "fasta")

    refs = {"WG":"test_out/ref_WG.fasta"}
    for g in genes:
        if g not in gene_coords:
            print(f"Gene {g} not found in GFF3 annotation, skipping.", file=sys.stderr)
            continue
        seq = ref_seq.seq[gene_coords[g][0]:gene_coords[g][1]]

        refs[g] = f"test_out/ref_{g}.fasta"

        SeqIO.write(SeqRecord(seq, id=f"{g}_reference", description=""), refs[g], "fasta")
    
    SeqIO.write(SeqRecord(ref_seq.seq, id="WG_reference", description=""), refs["WG"], "fasta")

    # Remove gaps
    records = []
    no_gap_seq= "test_out/seq_nogaps.fasta"

    for rec in SeqIO.parse(filtered_seq, "fasta"):
        rec.seq = rec.seq.replace("-", "")
        records.append(rec)
    SeqIO.write(records, no_gap_seq, "fasta")

    for r,p in refs.items():
        Path(path_blastdb).mkdir(parents=True, exist_ok=True)
        blastdb_path = os.path.join(path_blastdb, f"blast_db_{r}")
        blast_out = f"test_out/blast_out_{r}.csv"

        # Build BLAST database
        subprocess.run(
            ["makeblastdb", "-in", p, "-out", blastdb_path, "-dbtype", "nucl"],
            check=True,
        )

        # Run blastn
        outfmt = "10 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovs"
        subprocess.run(
            ["blastn", "-task", "blastn", "-query", no_gap_seq, "-db", blastdb_path,
            "-outfmt", outfmt, "-out", blast_out, "-evalue", "0.0005"],
            check=True,
        )

    subprocess.run(["rm", no_gap_seq], check=True)

def random_fragment(seq, length, max_tries=3):
    """Pick a random `length`-nt window of `seq`, avoiding gap/N-heavy windows.

    Returns (start, end, subsequence) with coordinates relative to `seq`, or None
    if `seq` is too short. Windows that are more than 10% gaps/Ns are retried, so a
    fragment is not scored as a tool failure when it is really just missing data.
    """
    length  = int(length)
    if len(seq) <= length:
        return None
    best = None
    for _ in range(max_tries):
        start = random.randint(0, len(seq) - length)
        # round start position to avoid cutting in the middle of a codon (for coding regions)
        start = start - (start % 3)
        sub = seq[start:start + length]
        ambiguous = sub.count("-") + sub.count("N") + sub.count("n")
        if best is None or ambiguous < best[0]:
            best = (ambiguous, start, sub)
        if ambiguous <= 0.1 * length:
            break
    _, start, sub = best
    return start, start + length, sub


#%%

BLAST_COLUMNS = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qcovs",
]


def get_min_lengths(genes, reference_file, gff_file, fraction=0.8):
    """Minimum hit length per region (fraction of the reference region length)."""
    ref_len = len(SeqIO.read(reference_file, "fasta").seq)
    min_length = {"WG": int(ref_len * fraction)}

    gene_coords = parse_gff_gene_coords(gff_file)
    for g in genes:
        if g not in gene_coords:
            print(f"Gene {g} not found in GFF3 annotation, skipping.", file=sys.stderr)
            continue
        start, end = gene_coords[g]
        min_length[g] = int((end - start) * fraction)

    return min_length


def load_blast_results(regions, blast_dir):
    """Read one BLAST csv per region and tag each row with its region."""
    frames = []
    for region in regions:
        df = pd.read_csv(f"{blast_dir}/blast_out_{region}.csv", names=BLAST_COLUMNS)
        df["region"] = region  # region from the file name, not from sseqid
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extract_regions(blast_results, sequences, min_length):
    """
    For each region: keep the best hit per query, filter by length,
    and cut the aligned part out of the query sequence.
    Returns {region: [SeqRecord, ...]}.
    """
    region_seqs = {}

    for region, df in blast_results.groupby("region"):
        df = df.copy()  # avoids SettingWithCopyWarning
        df["aln_len_ref"] = (df["send"] - df["sstart"]).abs() + 1

        # Best hit per query: longest alignment, then lowest e-value
        best = (
            df.sort_values(["length", "evalue"], ascending=[False, True])
              .drop_duplicates(subset="qseqid", keep="first")
        )
        best = best[best["aln_len_ref"] >= min_length[region]]

        records = []
        for hit in best.itertuples():
            if hit.qseqid not in sequences:
                continue
            subseq = sequences[hit.qseqid].seq[hit.qstart - 1:hit.qend]  # BLAST is 1-based
            if len(subseq) >= min_length[region]:
                # region in the ID so the same accession can appear in several regions without duplicate IDs
                records.append(SeqRecord(subseq, id=f"{hit.qseqid}_{region}", description=region))

        print(f"{region}: retained {len(records)} of {len(sequences)} sequences")
        region_seqs[region] = records

    return region_seqs


def make_fragments(records, lengths, n_per_length=3, seed=42):
    """Cut one random fragment per record; each length is used n_per_length times."""
    target_lengths = [int(l) for l in lengths for _ in range(n_per_length)]

    rng = random.Random(seed)  # local generator, doesn't touch the global random state
    rng.shuffle(target_lengths)

    fragments = []
    for rec, length in zip(records, target_lengths):
        frag = random_fragment(rec.seq, length)
        if frag is None:
            print(f"Sequence {rec.id} is too short for fragment length {length}", file=sys.stderr)
            continue
        start, end, subseq = frag
        fragments.append(SeqRecord(subseq, id=rec.id, description=f"WG_{length}nt_{start}_{end}"))

    return fragments


def write_outputs(downsampled, out_dir):
    """Write one FASTA file per region."""
    for region, records in downsampled.items():
        SeqIO.write(records, f"{out_dir}/downsampled_{region}.fasta", "fasta")


def blast_sort(genes, lengths, reference_file, gff_file, filtered_seq,
               blast_dir="test_out", n_random=30, seed=42):
    """Run all steps and return {region: [SeqRecord, ...]}."""
    min_length = get_min_lengths(genes, reference_file, gff_file)
    regions = list(min_length)  # "WG" + genes found in the GFF (new list, input untouched)

    blast_results = load_blast_results(regions, blast_dir)
    sequences = SeqIO.to_dict(SeqIO.parse(filtered_seq, "fasta"))  # {id: SeqRecord}

    region_seqs = extract_regions(blast_results, sequences, min_length)

    downsampled = {}
    for region, records in region_seqs.items():
        selected = random_select(records, n_random, seed=seed)
        if region == "WG":
            selected = make_fragments(selected, lengths, seed=seed)
        downsampled[region] = selected

    return downsampled


def generate_fragments(reference_file, nextstrain_meta_file, virus, path_blastdb,
                       sequences_file, filtered_seq, fragments_out,
                       lengths=None, genes=None, gff_file=None, seed=42,
                       ):
    """
    Download metadata from Nextstrain and filter sequences for accessions present in the metadata.
    Blast sequences to correct gene regions and cut fragments of given lengths from whole-genome sequences.
    """
    # Download metadata from Nextstrain and filter sequences for accessions present in the metadata
    json_to_meta(virus, nextstrain_meta_file)
    filter_seq_in_nextstrain(sequences_file, nextstrain_meta_file, filtered_seq)

    # define fragment lengths and genes to cut fragments from
    if lengths is None:
        lengths = DEFAULT_LENGTHS
    if lengths and not isinstance(lengths, list):
        lengths = lengths.split()
    
    if genes is None:
        genes = ["VP1", "3D"]
    if not isinstance(genes, list):
        genes = genes.split()

    run_blast(gff_file, reference_file, genes, filtered_seq, path_blastdb)

    downsampled = blast_sort(genes, lengths, reference_file, gff_file, filtered_seq)
    write_outputs(downsampled, "test_out")
    
    ## mutate the sequences with generate_div script
    for r in downsampled.keys():
        gd.main(["-i", f"test_out/downsampled_{r}.fasta", "-o", f"test_out/mutated_{r}.fasta",
                        "-d", "0.01", "-n", "1", "-ka", "2.0", "-s", f"{seed}"])
        # -d sets the divergence levels. [0.01, 0.05, ...]
        # -n sets how many copies to make per sequence and level.
        # -ka sets the transition/transversion ratio.
        # -s sets the random seed.

    subprocess.run(f"cat test_out/mutated_*.fasta > {fragments_out}", shell=True, check=True)

    print(f"Done generating fragments for {virus}. Outputs are in {fragments_out}.", file=sys.stderr)
    return sum(1 for _ in SeqIO.parse(fragments_out, "fasta"))


def generate_recombinants(sequences_file, nextstrain_meta_file, clades_file, ev_file,
                          output_file, inter_count=10, intra_count=10, min_length=3500):
    """Generate inter- and intra-recombinant test sequences."""
    seqs = list(SeqIO.parse(sequences_file, "fasta"))
    ns_ids = set(pd.read_csv(nextstrain_meta_file, sep="\t").name)
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
        # related-species headers look like "ACC | EV-A125"; fall back to the ID if there is no label
        label = p2.description.split("|")[1].strip() if "|" in p2.description else p2.id
        header = f"inter_{p1.id}_virus_{breakpoint}_{p2.id}_{label.replace(' ', '_')}"
        recomb_seq = str(p1.seq[:breakpoint]) + str(p2.seq[breakpoint:])
        recombinants.append((header, recomb_seq))
    
    # Write recombinants
    with open(output_file, "w") as f:
        for header, seq in recombinants:
            f.write(f">{header}\n{seq}\n")
    
    return len(recombinants)
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test sequences")
    parser.add_argument("--sequences", "-s", required=True, help="Input sequences file")
    parser.add_argument("--clades", required=True, help="Clades metadata TSV")
    parser.add_argument("--gff", required=True, help="GFF3 genome annotation file")
    parser.add_argument("--reference", "-r", required=True, help="Reference genome FASTA")
    parser.add_argument("--ev-fasta", required=False, help="EV species sequences FASTA (if not fetching from NCBI)")

    parser.add_argument("--output-fragments", required=True, help="Output fragments FASTA")
    parser.add_argument("--filtered-seq", required=False, help="Filtered sequences FASTA (after Nextstrain accession filtering)")
    parser.add_argument("--output-recombinants", required=True, help="Output recombinants FASTA")
    parser.add_argument("--output-ev", required=False, help="Output EV species sequences FASTA (if fetching from NCBI)")
    parser.add_argument("--nextstrain-file-path", required=False, help="Nextstrain metadata TSV")
    parser.add_argument("--blastdb", required=False, help="Path to BLAST database directory")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--taxid", type=str, required=False, help="NCBI Taxon ID for EV species (if not provided as file)")
    parser.add_argument("--virus", type=str, required=False, default="EV", help="Name of dataset virus to exclude from Entrez fetch")
    parser.add_argument("--genes", required=False, nargs='+', default=None, help="Genes for which fragments are created")
    parser.add_argument("--email", required=False, help="Email for NCBI Entrez access")
    parser.add_argument("--lengths", required=False, nargs="+", type=int, default=None, help=f"Fragment lengths to generate (default: {DEFAULT_LENGTHS})")

    args = parser.parse_args()
    random.seed(args.seed)
    
    Path(args.output_fragments).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_recombinants).parent.mkdir(parents=True, exist_ok=True)

    # Handle EV species sequences: either use provided file or fetch from NCBI
    if args.ev_fasta:
        ev_file = args.ev_fasta
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

    n_frags = generate_fragments(args.reference,args.nextstrain_file_path, args.virus, args.blastdb,
                                 args.sequences, args.filtered_seq, args.output_fragments,
                                 lengths=args.lengths, genes=args.genes, gff_file=args.gff,
                                 seed=args.seed)
    
    if ev_file:
        n_recomb = generate_recombinants(args.sequences, args.nextstrain_file_path,
                                         args.clades, ev_file, args.output_recombinants)
    else:
        # still write an (empty) file so the Snakemake output exists
        open(args.output_recombinants, "w").close()
        n_recomb = 0
    
    print(f"Generated {n_frags} fragments and {n_recomb} recombinants")