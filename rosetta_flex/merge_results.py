"""
Concatenate the per-job Rosetta Flex ddG CSVs (from the SLURM array) into one
shared-schema dataset, de-duplicating and reporting coverage.

Each array task writes ``rosetta_flex/results/parts/job_<i>.csv`` (one ddG row).
This merges them into a single ``Synthetic_FlexddG_TCR_pMHC.csv`` with the same
columns the MadraX generator emits, so the two tiers concatenate cleanly:

    PDB_ID, Chain, Residue_Position, WT_Amino_Acid, Mutant_Amino_Acid, ddG, source
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("merge_results")

FIELDS = ["PDB_ID", "Chain", "Residue_Position", "WT_Amino_Acid", "Mutant_Amino_Acid", "ddG", "source"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-job Flex ddG CSVs into one dataset.")
    ap.add_argument("--parts_dir", type=str, default="rosetta_flex/results/parts")
    ap.add_argument("--output", type=str, default="rosetta_flex/results/Synthetic_FlexddG_TCR_pMHC.csv")
    args = ap.parse_args()

    parts = sorted(glob.glob(str(Path(args.parts_dir) / "*.csv")))
    if not parts:
        LOGGER.error("No part files found in %s.", args.parts_dir)
        return

    seen = set()
    n_rows = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for part in parts:
            with open(part, newline="") as fh:
                for row in csv.DictReader(fh):
                    key = (row["PDB_ID"], row["Chain"], row["Residue_Position"], row["WT_Amino_Acid"], row["Mutant_Amino_Acid"])
                    if key in seen:
                        continue
                    seen.add(key)
                    writer.writerow({k: row[k] for k in FIELDS})
                    n_rows += 1

    LOGGER.info("Merged %d part files into %d unique ddG rows -> %s", len(parts), n_rows, out_path)


if __name__ == "__main__":
    main()
