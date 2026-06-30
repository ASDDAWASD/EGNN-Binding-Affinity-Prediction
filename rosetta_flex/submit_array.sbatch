#!/bin/bash
#SBATCH --job-name=flex_ddg
#SBATCH --array=0-99%50          # OVERRIDE on submit: --array=0-<N-1> from jobs.csv (one task per row); %50 caps concurrency
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00          # ~minutes/job at the cheap 3500 backrub trials x nstruct 1 settings (bump to 08:00:00 for full 35000 x 35)
#SBATCH --output=rosetta_flex/logs/%A_%a.out
#SBATCH --error=rosetta_flex/logs/%A_%a.err
#
# Tier-2 Rosetta Flex ddG array driver. ONE array task == ONE jobs.csv row.
#
# Usage (after `python rosetta_flex/make_mutfiles.py` has produced jobs.csv):
#   NJOBS=$(($(wc -l < rosetta_flex/jobs/jobs.csv) - 1))     # minus header
#   sbatch --array=0-$((NJOBS-1))%50 rosetta_flex/submit_array.sbatch
#
# Each task writes its own per-job CSV under results/parts/ to avoid write
# contention; merge_results.py concatenates them afterwards.

set -euo pipefail

module load rosetta 2>/dev/null || true          # site-specific; or set ROSETTA_SCRIPTS_BIN below
export ROSETTA_SCRIPTS_BIN="${ROSETTA_SCRIPTS_BIN:-rosetta_scripts.default.linuxgccrelease}"

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p rosetta_flex/logs rosetta_flex/results/parts

python rosetta_flex/run_flex_ddg.py \
  --jobs_csv rosetta_flex/jobs/jobs.csv \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --rosetta_bin "${ROSETTA_SCRIPTS_BIN}" \
  --protocol_xml rosetta_flex/ddG_backrub.xml \
  --backrub_trials "${BACKRUB_TRIALS:-3500}" \
  --nstruct "${NSTRUCT:-1}" \
  --workdir "rosetta_flex/results/work/job_${SLURM_ARRAY_TASK_ID}" \
  --output "rosetta_flex/results/parts/job_${SLURM_ARRAY_TASK_ID}.csv"
