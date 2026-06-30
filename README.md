# Thermodynamics Prediction via EGNNs

Synthetic ΔΔG datasets for **TCR-pMHC** interfaces, for training EGNNs to predict
binding-affinity changes upon mutation.

## Pipelines

Two generators feed the curriculum; **both emit the same ddG-label CSV** so the
tiers concatenate cleanly:

```
PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid, ddG, source
```
`source ∈ {madrax, rosetta_flex}`; ΔΔG is a **binding** ΔΔG (mutant − wild type),
obtained by chain separation (TCR vs pMHC) so it is comparable to the ATLAS/TRAIT
benchmarks. MadraX adds two analysis-only columns, `Distance_to_CDR_center` and
`Min_Interchain_Distance` (distance to the opposite partner — small ⇒ a true
inter-chain contact).

| Tier | Engine | Script | Teaches | Scale |
|------|--------|--------|---------|-------|
| 1 | MadraX (GPU, rigid) | `generate_tcr_pmhc_dataset.py` | steric clashes / immune geometry | ≥1M |
| 2 | Rosetta Flex ddG (HPC, flexible) | `rosetta_flex/` | backbone flexibility | ~thousands |

Both share interface detection (`interface_utils.py`), so they mutate comparable positions.

## Setup
```bash
uv sync                    # core deps (madrax, torch+cu126, biopython, ...)
uv sync --extra renumber   # optional ANARCI, only for fetch_stcrdab --renumber-with-anarci
```

## Step 1 — Acquire structures (`scripts/fetch_stcrdab.py`)

Downloads αβ TCR-pMHC complexes from STCRDab, collapses metadata duplicates
(peptide + MHC allele + TCR genes), and standardizes chains to **A/B/C/D/E**
(MHC1 / MHC2-β2m / peptide / TCRα / TCRβ). STCRDab serves IMGT-numbered
coordinates, so no renumbering is needed by default.

**Leakage guard:** PDBs that appear in the ATLAS/TRAIT benchmarks are hard-excluded
by default via `scripts/benchmark_blocklist.txt` (`--exclude_pdbs`; pass `''` to
disable), so training labels never overlap the evaluation sets.

```bash
python scripts/fetch_stcrdab.py --out_dir stcrdab_structures/ --max_structures 5   # dry run
python scripts/fetch_stcrdab.py --out_dir stcrdab_structures/                       # full pull
```

## Step 2 — Tier 1: MadraX (`generate_tcr_pmhc_dataset.py`)

Rigid, GPU-batched saturation mutagenesis (19 mutants per interface residue). CDR
loops are read from IMGT numbers (CDR1 27-38, CDR2 56-65, CDR3 105-117) on chains D/E,
so inputs must be IMGT-numbered — exactly what Step 1 produces. Each mutant is scored
as a **binding** ΔΔG: the full complex and the mutated residue's own chain group in
isolation are both scored, and the label is `complex_ΔΔG − partner_ΔΔG` (the
non-mutated partner cancels). Rows are flushed every batch, so an interrupted run
keeps what it computed.

```bash
# dry run (inspect the CSV)
uv run generate_tcr_pmhc_dataset.py --limit 1 --num_workers 0 --output dry_run.csv
# full run: batched forward passes, streams to CSV, resumable
nohup uv run generate_tcr_pmhc_dataset.py --output tcr_pmhc_interface_ddg.csv \
  --mutation_batch_size 75 --target_samples 1000000 > gen.log 2>&1 &
```
`--resume` skips PDB IDs already in `<output>.done`. Steric-clash blow-ups
(`|ddG| > --clash_threshold`) are filtered to `<output>_failures.csv`.

**Throughput / parallelism.** The cost is CPU-bound (`create_info_tensors`,
~0.5 s/mutant) while the GPU stays mostly idle, so run several **shards** that share
one GPU rather than enlarging the batch. Each shard takes a disjoint slice of the
PDB list and writes its own `.shardN` CSV (`merge` afterwards by concatenating and
dropping duplicate headers):
```bash
for s in $(seq 0 15); do
  CUDA_VISIBLE_DEVICES=0 uv run generate_tcr_pmhc_dataset.py \
    --num_shards 16 --shard_id "$s" --num_workers 4 \
    --output tcr_pmhc_interface_ddg.csv > gen.shard$s.log 2>&1 &
done
```

## Step 3 — Tier 2: Rosetta Flex ddG (`rosetta_flex/`)

Backbone-flexible ΔΔG via the Flex ddG protocol
([Barlow et al. 2018](https://pubs.acs.org/doi/10.1021/acs.jpcb.7b11367)), on the
**same** structures. **HPC only** — see [`rosetta_flex/README.md`](rosetta_flex/README.md).
```bash
python rosetta_flex/make_mutfiles.py --pdb_dir stcrdab_structures/ --out_dir rosetta_flex/jobs
NJOBS=$(($(wc -l < rosetta_flex/jobs/jobs.csv) - 1))
sbatch --array=0-$((NJOBS-1))%50 rosetta_flex/submit_array.sbatch
python rosetta_flex/merge_results.py
```
Graphinity's published Flex ddG data is antibody–antigen and is used only as a
*protocol* reference, not as labels.
</content>
