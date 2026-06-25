# Thermodynamics Prediction via EGNNs

## Generating the TCR-pMHC Dataset

You can generate the Tier 1 synthetic dataset using `generate_tcr_pmhc_dataset.py`. It requires PyTorch, BioPython, and a real `madrax` install (see `pyproject.toml` / `uv sync`) — no pandas dependency. Place your **STCRDab** PDB files (IMGT-renumbered, with the standard A/B/C/D/E chain convention: MHC chain 1, MHC chain 2 or β2m, peptide, TCR alpha, TCR beta) in the `stcrdab_structures/` folder.

CDR loops are identified directly from each residue's IMGT-assigned number (CDR1: 27-38, CDR2: 56-65, CDR3: 105-117) on the TCR alpha/beta chains, not by guessing list positions — so the script depends on the input already being IMGT-numbered, exactly what STCRDab provides.

1-Sample Dry Run
Use a limit block to test your setup and visually inspect the CSV output:
```bash
.venv/bin/python generate_tcr_pmhc_dataset.py --limit 1 --num_workers 0 --output dry_run_tcr_pmhc_dataset.csv
```

Full Run
Processes the complete dataset, batching many point mutants of the same structure into each MadraX forward pass for GPU throughput, and streaming rows to CSV as it goes (safe to interrupt). Run this via `nohup` if you are executing remotely:
```bash
nohup .venv/bin/python generate_tcr_pmhc_dataset.py --output tcr_pmhc_interface_ddg.csv --mutation_batch_size 75 > dataset_generation.log 2>&1 &
```
Add `--resume` to skip PDB IDs already recorded as done in `<output>.done` if a run gets interrupted. Mutants whose MadraX energy blows up (steric clashes) are filtered out of the main CSV and logged to `<output>_failures.csv` instead.

Custom Directory Targeting
```bash
.venv/bin/python generate_tcr_pmhc_dataset.py --pdb_dir custom_pdb_batch_1/ --output batch_1_ddg.csv
```

---

## Previous Pipelines (Generic PPIs)

The repository also includes the legacy script `synthetic_ppi_dataset.py`.


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
