# Tier 2 — Rosetta Flex ddG pipeline (TCR-pMHC)

Generates the Tier-2 "plasticity" dataset: backbone-flexible ΔΔG values from the
**Flex ddG** protocol (Barlow et al., *J. Phys. Chem. B* 2018), run on **TCR-pMHC**
structures. Tier 2 teaches the EGNN how backbones flex to absorb a mutation —
the rigid MadraX physics of Tier 1 cannot.

> **This pipeline is written to run on an HPC cluster, not locally.** Nothing
> here is executed as part of the build. Rosetta is a large, separately-licensed
> C++ package; install/build it on the cluster first.

> **Note on Graphinity.** Graphinity ships a published Flex ddG set, but it is
> **antibody–antigen (SAbDab)**, a different binding problem. We do **not** use
> those numbers as labels. We reproduce the *protocol* here and run it on our own
> TCR-pMHC structures so the labels are TCR-relevant.

## Prerequisites
- A built Rosetta with `rosetta_scripts` (academic/commercial license from
  RosettaCommons). Point `ROSETTA_SCRIPTS_BIN` at the binary, or `module load rosetta`.
- The standardized TCR-pMHC PDBs in `stcrdab_structures/` (produced by
  `scripts/fetch_stcrdab.py` — the same inputs as Tier 1).
- Python env from the repo root (`uv sync`).

## Files
| File | Role |
|------|------|
| `ddG_backrub.xml` | Canonical flex_ddG RosettaScripts protocol (backrub ensemble + `InterfaceDdGMover`, talaris2014). |
| `make_mutfiles.py` | Enumerates interface mutations (via shared `interface_utils`), writes per-mutation resfiles + `jobs.csv`. |
| `run_flex_ddg.py` | Runs one `jobs.csv` row through Rosetta and parses `ddG.db3` → one shared-schema CSV row. |
| `submit_array.sbatch` | SLURM array driver: one array task per `jobs.csv` row. |
| `merge_results.py` | Concatenates per-job CSVs into one Flex ddG dataset. |

## Workflow (on HPC)
```bash
# 1. Enumerate jobs (safe to run anywhere; no Rosetta needed)
python rosetta_flex/make_mutfiles.py --pdb_dir stcrdab_structures/ --out_dir rosetta_flex/jobs

# 2. Submit the array (one task per mutation)
NJOBS=$(($(wc -l < rosetta_flex/jobs/jobs.csv) - 1))
sbatch --array=0-$((NJOBS-1))%50 rosetta_flex/submit_array.sbatch

# 3. Merge per-job outputs into the final dataset
python rosetta_flex/merge_results.py
```

## Protocol parameters (cheap defaults, after Graphinity 2025)

Defaults follow the cost-reduced Flex ddG settings from Hummer et al.
([*Nat. Comput. Sci.* 2025](https://www.nature.com/articles/s43588-025-00823-8)),
which make a >20k-mutation Tier-2 dataset feasible:

- **Backrub trials: `3500`** (`--backrub_trials` / `BACKRUB_TRIALS`) — vs Barlow 2018's `35000`.
- **Ensemble size: `nstruct = 1`** (`--nstruct` / `NSTRUCT`) — vs Barlow 2018's `35`.
- Score function: `talaris2014`.
- `chains_to_move`: `DE` (TCR α/β separated from pMHC `ABC`) — set in `make_mutfiles.py`.

Together these are ~**350× cheaper per mutation** than the full Barlow protocol
(`10× fewer backrub steps × 35× smaller ensemble`) for **near-identical ΔΔG**.
Why it holds up:
- ΔΔG is a **difference** (mutant − WT) scored in the *same* locally-relaxed
  backbone context, so most single-model/limited-sampling noise **cancels** — one
  relaxed model (`nstruct=1`) already captures the signal.
- 3,500 backrub steps already relax the *local* backbone enough to relieve the
  strain a point mutation introduces; extending to 35,000 mostly re-samples the
  same energy basin (diminishing returns).
- The accuracy ceiling here is Flex ddG's intrinsic correlation to experiment
  (~0.46), **not** ensemble size — so spending 350× compute to shave ensemble
  noise barely moves it. For an EGNN training set, **volume + diversity beats
  per-label precision** (the paper's central finding), so that compute is far
  better spent on *more* mutations.

Pass `--backrub_trials 35000 --nstruct 35` (or `BACKRUB_TRIALS`/`NSTRUCT` env
vars) to recover the full Barlow protocol.

- **GAM reweighting (recommended).** In Hummer et al., GAM-reweighted Flex ddG
  reached Pearson 0.46 vs experiment, beating both non-GAM Flex ddG (0.42) and
  FoldX (0.20). Pass `--gam-coeffs <json>` to `run_flex_ddg.py` with the official
  `{score_type: weight}` coefficients from the
  [flex_ddG tutorial](https://github.com/Kortemme-Lab/flex_ddG_tutorial) (drop the
  file in e.g. `rosetta_flex/gam_coeffs.json`). Without it, ΔΔG uses Rosetta's raw
  `total_score` (the uncalibrated "nogam" variant). We do not ship invented coefficients.

- **Insertion codes.** `make_mutfiles.py` writes the PDB insertion code into each
  resfile (`<resnum><icode> <chain> PIKAA ...`), so IMGT CDR3 insertion-coded
  positions are mutated correctly in Rosetta (which distinguishes them, unlike the
  MadraX integer-only parser).

## Output schema (shared with Tier 1)
```
PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid, ddG, source
```
`source = "rosetta_flex"`. ΔΔG sign convention is **mutant − wild type**, matching
the MadraX generator.

## Local validation without Rosetta
```bash
python rosetta_flex/run_flex_ddg.py --self-test          # synthetic db3 → parser check
python rosetta_flex/run_flex_ddg.py --parse-only path/to/ddG.db3   # parse a real db3
python rosetta_flex/make_mutfiles.py --pdb_dir stcrdab_structures/ --limit 1   # resfiles + jobs.csv
```
