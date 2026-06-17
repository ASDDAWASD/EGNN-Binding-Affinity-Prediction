# ExLAI-2026

## Run

The repository currently includes one runnable script: `synthetic_ppi_dataset.py`.

Deterministic validation run:

```bash
cd "/Users/samarthkhandelwal/Documents/Thermodynamics Prediction via EGNNs"
"/Users/samarthkhandelwal/Documents/Thermodynamics Prediction via EGNNs/.venv/bin/python" "synthetic_ppi_dataset.py" \
	--target-samples 1 \
	--seed-pdb 1BRS \
	--output "artifacts/dryrun_seeded.pt" \
	--cache-dir "artifacts/cache"
```

Standard 1-sample dry run:

```bash
cd "/Users/samarthkhandelwal/Documents/Thermodynamics Prediction via EGNNs"
"/Users/samarthkhandelwal/Documents/Thermodynamics Prediction via EGNNs/.venv/bin/python" "synthetic_ppi_dataset.py" \
	--target-samples 1 \
	--output "artifacts/dryrun.pt" \
	--cache-dir "artifacts/cache"
```

Full dataset generation:

```bash
cd "/Users/samarthkhandelwal/Documents/Thermodynamics Prediction via EGNNs"
"/Users/samarthkhandelwal/Documents/Thermodynamics Prediction via EGNNs/.venv/bin/python" "synthetic_ppi_dataset.py" \
	--target-samples 100 \
	--output "artifacts/synthetic_ppi_dataset.pt" \
	--cache-dir "artifacts/cache"
```
