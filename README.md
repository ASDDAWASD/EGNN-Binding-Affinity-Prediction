# Thermodynamics Prediction via EGNNs

## Generating the TCR-pMHC Dataset

You can generate the Tier 1 synthetic dataset using `generate_tcr_pmhc_dataset.py`. Make sure your environment has PyTorch, BioPython, and pandas. Place your STCRDab PDB files in the `stcrdab_structures/` folder.

#5-Sample Dry Run
```bash
python generate_tcr_pmhc_dataset.py --limit 5 --output dry_run_tcr_pmhc_dataset.csv
```

#Full Run
```bash
nohup python generate_tcr_pmhc_dataset.py --output tcr_pmhc_interface_ddg.csv > dataset_generation.log 2>&1 &
```

#Custom Directory Targeting
```bash
python generate_tcr_pmhc_dataset.py --pdb_dir custom_pdb_batch_1/ --output batch_1_ddg.csv
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
