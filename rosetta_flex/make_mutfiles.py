"""
Enumerate Rosetta Flex ddG jobs for the Tier-2 dataset.

For every structure in ``--pdb_dir`` we reuse the *exact same* interface
detection as the MadraX generator (``interface_utils.find_interface_targets``)
so Tier 1 and Tier 2 mutate comparable positions, then write one Rosetta resfile
per (residue, mutant amino acid) and a single ``jobs.csv`` describing every job.

A SLURM array (see submit_array.sbatch) then runs one ``jobs.csv`` line per task
through ``run_flex_ddg.py``. Nothing here invokes Rosetta -- it only prepares
inputs, so it is safe to run anywhere (including to dry-run the pipeline).

Resfile format (Rosetta): default all positions to repack-as-native (NATAA),
then force the single target position to the mutant identity with PIKAA.

    NATAA
    start
    <resnum> <chain> PIKAA <mut_one_letter>
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path

# Allow `python rosetta_flex/make_mutfiles.py` from the repo root to import the
# shared helper that lives one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Bio.PDB import PDBParser

from interface_utils import (
    DEFAULT_COMPLEX_CHAINS,
    DEFAULT_TCR_CHAINS,
    THREE_TO_ONE,
    find_interface_targets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("make_mutfiles")

ONE_LETTER = [THREE_TO_ONE[a] for a in THREE_TO_ONE]

JOBS_FIELDS = [
    "job_id",
    "pdb_id",
    "pdb_path",
    "chain",
    "resnum",
    "wt_aa",      # 1-letter
    "mut_aa",     # 1-letter
    "resfile_path",
    "chains_to_move",
]


def write_resfile(path: Path, chain: str, resnum: int, mut_one: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"NATAA\nstart\n{resnum} {chain} PIKAA {mut_one}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Rosetta Flex ddG resfiles + jobs.csv for TCR-pMHC interfaces.")
    ap.add_argument("--pdb_dir", type=str, default="stcrdab_structures/", help="Directory of standardized TCR-pMHC PDBs (default: the same Tier-1 STCRDab set).")
    ap.add_argument("--out_dir", type=str, default="rosetta_flex/jobs", help="Where resfiles + jobs.csv are written.")
    ap.add_argument("--interface_radius", type=float, default=10.0, help="Interface radius (must match the MadraX run for comparable positions).")
    ap.add_argument("--tcr_chains", type=str, default=",".join(DEFAULT_TCR_CHAINS))
    ap.add_argument("--complex_chains", type=str, default=",".join(DEFAULT_COMPLEX_CHAINS))
    ap.add_argument("--chains_to_move", type=str, default="".join(DEFAULT_TCR_CHAINS),
                    help="Chains separated to compute binding ddG (default 'DE' = the TCR moves off the pMHC).")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N PDBs (dry runs).")
    args = ap.parse_args()

    tcr_chains = tuple(c.strip() for c in args.tcr_chains.split(",") if c.strip())
    complex_chains = tuple(c.strip() for c in args.complex_chains.split(",") if c.strip())
    out_dir = Path(args.out_dir)
    resfile_dir = out_dir / "resfiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(glob.glob(str(Path(args.pdb_dir) / "*.pdb")))
    if args.limit:
        pdb_files = pdb_files[: args.limit]
    if not pdb_files:
        LOGGER.error("No PDB files found in %s.", args.pdb_dir)
        return

    parser = PDBParser(QUIET=True)
    jobs_path = out_dir / "jobs.csv"
    job_id = 0
    n_struct = 0
    with open(jobs_path, "w", newline="") as jf:
        writer = csv.DictWriter(jf, fieldnames=JOBS_FIELDS)
        writer.writeheader()
        for pdb_path in pdb_files:
            pdb_id = Path(pdb_path).stem.lower()
            try:
                structure = parser.get_structure(pdb_id, pdb_path)
                targets = find_interface_targets(structure, tcr_chains, complex_chains, args.interface_radius)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("%s: interface detection failed (%s); skipping.", pdb_id, exc)
                continue
            if not targets:
                LOGGER.warning("%s: no interface targets found; skipping.", pdb_id)
                continue
            n_struct += 1
            for t in targets:
                wt_one = t.wt_aa_one
                for mut_one in ONE_LETTER:
                    if mut_one == wt_one:
                        continue
                    resfile = resfile_dir / pdb_id / f"{t.chain}{t.resnum}{wt_one}{mut_one}.resfile"
                    write_resfile(resfile, t.chain, t.resnum, mut_one)
                    writer.writerow(
                        {
                            "job_id": job_id,
                            "pdb_id": pdb_id,
                            "pdb_path": pdb_path,
                            "chain": t.chain,
                            "resnum": t.resnum,
                            "wt_aa": wt_one,
                            "mut_aa": mut_one,
                            "resfile_path": str(resfile),
                            "chains_to_move": args.chains_to_move,
                        }
                    )
                    job_id += 1

    LOGGER.info("Wrote %d jobs across %d structures to %s", job_id, n_struct, jobs_path)
    LOGGER.info("Submit with: sbatch --array=0-%d rosetta_flex/submit_array.sbatch", max(0, job_id - 1))


if __name__ == "__main__":
    main()
