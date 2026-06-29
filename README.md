# Thermodynamics Prediction via EGNNs

## Synthetic data-generation pipelines

Two generators feed the curriculum; **both emit the same ddG-label CSV** so the
tiers concatenate cleanly:

```
PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid, ddG, source
```
(`source ∈ {"madrax", "rosetta_flex"}`; ΔΔG sign convention is **mutant − wild type**.
The MadraX CSV appends one extra analysis-only column, `Distance_to_CDR_center`.)

| Tier | Engine | Script | Purpose | Scale |
|------|--------|--------|---------|-------|
| 1 | MadraX (GPU, rigid physics) | `generate_tcr_pmhc_dataset.py` | baseline immune geometry / steric clashes | ≥1,000,000 |
| 2 | Rosetta Flex ddG (HPC, backbone flexibility) | `rosetta_flex/` | how backbones flex to absorb a mutation | ~thousands |

Interface detection is shared by both tiers via `interface_utils.py`, so they
mutate comparable positions.

## Step 0 — Environment
```bash
uv sync                      # core deps (madrax from Bitbucket, torch, biopython, requests, ...)
uv sync --extra renumber     # optional: ANARCI, only for fetch_stcrdab --renumber-with-anarci
```

## Step 1 — Acquire STCRDab structures (`scripts/fetch_stcrdab.py`)

Downloads αβ TCR-pMHC complexes from STCRDab, applies a light non-redundancy
filter (peptide + MHC allele + TCR genes; keep best-resolution representative),
and standardizes chains to the **A/B/C/D/E** convention (A = MHC1, B = MHC2/β2m,
C = peptide, D = TCR α, E = TCR β) the generators assume. STCRDab already serves
IMGT-numbered coordinates, so no renumbering library is needed by default.

```bash
# small dry run (5 structures + manifest.csv)
python scripts/fetch_stcrdab.py --out_dir stcrdab_structures/ --max_structures 5
# full pull (~1,000 structures -> ~1M MadraX samples)
python scripts/fetch_stcrdab.py --out_dir stcrdab_structures/
```

## Step 2 — Tier 1: MadraX dataset (`generate_tcr_pmhc_dataset.py`)

Requires PyTorch, BioPython, and a real `madrax` install. CDR loops are read
directly from each residue's IMGT number (CDR1: 27-38, CDR2: 56-65, CDR3: 105-117)
on the TCR α/β chains — so inputs must be IMGT-numbered, exactly what Step 1 produces.

1-Sample Dry Run
Use a limit block to test your setup and visually inspect the CSV output:
```bash
.venv/bin/python generate_tcr_pmhc_dataset.py --limit 1 --num_workers 0 --output dry_run_tcr_pmhc_dataset.csv
```

Full Run
Processes the complete dataset, batching many point mutants of the same structure into each MadraX forward pass for GPU throughput, and streaming rows to CSV as it goes (safe to interrupt). Use `--target_samples` to stop cleanly once enough rows are written. Run this via `nohup` if you are executing remotely:
```bash
nohup .venv/bin/python generate_tcr_pmhc_dataset.py --output tcr_pmhc_interface_ddg.csv --mutation_batch_size 75 --target_samples 1000000 > dataset_generation.log 2>&1 &
```
Add `--resume` to skip PDB IDs already recorded as done in `<output>.done` if a run gets interrupted (also extends the dataset beyond a previous `--target_samples` stop). Mutants whose MadraX energy blows up (steric clashes) are filtered out of the main CSV and logged to `<output>_failures.csv` instead.

Custom Directory Targeting
```bash
.venv/bin/python generate_tcr_pmhc_dataset.py --pdb_dir custom_pdb_batch_1/ --output batch_1_ddg.csv
```

## Step 3 — Tier 2: Rosetta Flex ddG dataset (`rosetta_flex/`)

Backbone-flexible ΔΔG via the Flex ddG protocol (Barlow et al. 2018), run on the
**same** TCR-pMHC structures. **Designed for HPC — not run locally.** See
[`rosetta_flex/README.md`](rosetta_flex/README.md) for the full workflow; in brief:
```bash
python rosetta_flex/make_mutfiles.py --pdb_dir stcrdab_structures/ --out_dir rosetta_flex/jobs
NJOBS=$(($(wc -l < rosetta_flex/jobs/jobs.csv) - 1))
sbatch --array=0-$((NJOBS-1))%50 rosetta_flex/submit_array.sbatch
python rosetta_flex/merge_results.py
```
Graphinity's published Flex ddG data is **antibody–antigen** and is used only as a
*protocol* reference, not as labels — these labels are generated on TCR-pMHC.

---

## Previous Pipelines (Generic PPIs) — deprecated

The repository also includes the legacy script `synthetic_ppi_dataset.py`. It is
**not part of the current pipeline** and is kept only for reference.


On Mac/Linux
Deterministic validation run :

```bash
cd "/path/to/repo"
"/path/to/repo/.venv/bin/python" "synthetic_ppi_dataset.py" \
	--target-samples 1 \
	--seed-pdb 1BRS \
	--output "artifacts/dryrun_seeded.pt"
	--cache-dir "artifacts/cache"
```

Standard 1-sample dry run:

```bash
cd "/path/to/repo"
"/path/to/repo/.venv/bin/python" "synthetic_ppi_dataset.py" \
	--target-samples 1 \
	--output "artifacts/dryrun.pt" \
	--cache-dir "artifacts/cache"
```

Full dataset generation:

```bash
cd "/path/to/repo"
"/path/to/repo/.venv/bin/python" "synthetic_ppi_dataset.py" \
	--target-samples 100 \
	--output "artifacts/synthetic_ppi_dataset.pt" \
	--cache-dir "artifacts/cache"
```


On Windows
Deterministic validation run:

```bash
cd "/path/to/repo"
"/path/to/repo/.venv/Scripts/python" "synthetic_ppi_dataset.py" \
	--target-samples 1 \
	--seed-pdb 1BRS \
	--output "artifacts/dryrun_seeded.pt"
	--cache-dir "artifacts/cache"
```

Standard 1-sample dry run:

```bash
cd "/path/to/repo"
"/path/to/repo/.venv/Scripts/python" "synthetic_ppi_dataset.py" \
	--target-samples 1 \
	--output "artifacts/dryrun.pt" \
	--cache-dir "artifacts/cache"
```

Full dataset generation:

```bash
cd "/path/to/repo"
"/path/to/repo/.venv/Scripts/python" "synthetic_ppi_dataset.py" \
	--target-samples 100 \
	--output "artifacts/synthetic_ppi_dataset.pt" \
	--cache-dir "artifacts/cache"
```
