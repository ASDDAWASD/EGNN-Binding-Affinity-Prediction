"""
Tier 1 synthetic TCR-pMHC ddG dataset generator.

Pipeline
--------
1. Ingest IMGT-renumbered STCRDab structures (chains conventionally:
   A = MHC chain 1, B = MHC chain 2 / beta2-microglobulin, C = peptide,
   D = TCR alpha chain, E = TCR beta chain).
2. Identify the 6 CDR loops on the TCR alpha/beta chains using their IMGT
   residue numbers (CDR1: 27-38, CDR2: 56-65, CDR3: 105-117).
3. Identify the "binding interface" as every residue (on the TCR, peptide,
   or MHC) within `--interface_radius` Angstrom of any CDR atom.
4. Run saturation mutagenesis (19 point mutations per interface residue)
   through the real MadraX PyTorch force field, batching many mutants of
   the same structure into a single forward pass for GPU throughput.
5. Stream results to CSV incrementally, logging/skipping structures or
   mutants that blow up (steric clashes -> NaN/Inf/huge energies).

This replaces a previous version of this script that called a
`force_field.compute_energy(...)` method that does not exist in MadraX and
silently fell back to a random-number mock, identified CDR loops by
positional list slicing (wrong whenever a structure has missing residues or
IMGT insertion codes), and required `pandas`, which was never declared as a
project dependency. All of that is fixed here using MadraX's real API
(`madrax.utils.parsePDB`, `madrax.dataStructures.create_info_tensors`,
`madrax.mutate.mutatingEngine.mutate`, `madrax.ForceField.ForceField`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from Bio.PDB import NeighborSearch, PDBIO, PDBParser, Select, Selection
from torch.utils.data import DataLoader, Dataset

try:
    from madrax.ForceField import ForceField
    from madrax import dataStructures
    from madrax import utils as madrax_utils
    from madrax.mutate.mutatingEngine import mutate as madrax_mutate
except ImportError as exc:  # MadraX is a hard requirement, not optional.
    raise SystemExit(
        "MadraX is required to run this script but could not be imported. "
        "Install it with `uv sync` (see pyproject.toml) or "
        "`pip install git+https://bitbucket.org/grogdrinker/madrax/`. "
        f"Original error: {exc}"
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("tcr_pmhc_dataset")

# Standard 20 amino acids, 3-letter codes (matches both BioPython resnames and
# the residue-name tokens MadraX expects in its "resNum_chain_RESNAME" mutation
# directives, e.g. "39_D_GLY").
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AMINO_ACID_SET = set(AMINO_ACIDS)

# STCRDab's standardized chain identifiers after IMGT renumbering.
DEFAULT_TCR_CHAINS = ("D", "E")  # TCR alpha, TCR beta
DEFAULT_COMPLEX_CHAINS = ("A", "B", "C", "D", "E")  # MHC1, MHC2/B2M, peptide, TCR a/b

# IMGT V-domain CDR residue-number ranges (inclusive). These index the actual
# IMGT-assigned residue number stored in each residue's PDB sequence number,
# not the residue's position in a Python list, so missing N-terminal residues
# or gaps elsewhere in the loop do not shift the window.
IMGT_CDR_RANGES = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}

CSV_FIELDNAMES = [
    "PDB_ID",
    "Chain",
    "Residue_Position",
    "WT_Amino_Acid",
    "Mutant_Amino_Acid",
    "Distance_to_CDR_center",
    "MadraX_ddG",
]

# MadraX's plain-text PDB parser keys atoms purely by integer residue number
# (madrax/utils.py: `resnum = int(line[22:26])`), so IMGT insertion codes
# (e.g. "111A", "111B") collapse onto the same integer position. We mirror
# that limitation deliberately: every step below identifies residues by
# (chain, integer resseq) only, so mutation directives sent to MadraX always
# refer to a residue MadraX can actually distinguish.


class InterfaceChainSelect(Select):
    """BioPython Select that keeps only standard residues on whitelisted chains."""

    def __init__(self, allowed_chains: Sequence[str]):
        self.allowed_chains = set(allowed_chains)

    def accept_chain(self, chain):
        return chain.id in self.allowed_chains

    def accept_residue(self, residue):
        return residue.id[0] == " "  # drop waters / heteroatoms / ligands


@dataclass
class MutationTarget:
    chain: str
    resnum: int
    wt_aa: str
    distance_to_cdr: float


@dataclass
class PdbJob:
    pdb_id: str
    clean_pdb_path: Optional[str]
    targets: List[MutationTarget] = field(default_factory=list)
    error: Optional[str] = None


def clean_structure(source_path: Path, dest_path: Path, allowed_chains: Sequence[str]) -> None:
    """Write a copy of `source_path` containing only standard residues on `allowed_chains`."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(source_path.stem, str(source_path))
    io = PDBIO()
    io.set_structure(structure)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(dest_path), select=InterfaceChainSelect(allowed_chains))


def identify_cdr_loops(structure, tcr_chain_ids: Sequence[str]):
    """Identify CDR1/2/3 residues on the TCR alpha/beta chains using IMGT numbering.

    Assumes the input structure has already been IMGT-renumbered (as STCRDab
    structures are), so a residue's CDR membership can be read directly off
    its PDB residue number instead of guessed from list position.
    """
    cdr_residues = []
    model = next(structure.get_models())
    for chain in model:
        if chain.id not in tcr_chain_ids:
            continue
        for residue in chain:
            if residue.id[0] != " ":
                continue
            resnum = residue.id[1]
            if any(lo <= resnum <= hi for lo, hi in IMGT_CDR_RANGES.values()):
                cdr_residues.append(residue)
    return cdr_residues


def get_interface_residues(structure, cdr_residues, radius: float, allowed_chains: Sequence[str]):
    """All residues (on any of `allowed_chains`) within `radius` Angstrom of any CDR atom."""
    if not cdr_residues:
        return []
    model = next(structure.get_models())
    searchable_atoms = [
        atom
        for chain in model
        if chain.id in allowed_chains
        for residue in chain
        if residue.id[0] == " "
        for atom in residue.get_atoms()
    ]
    if not searchable_atoms:
        return []
    cdr_atoms = Selection.unfold_entities(cdr_residues, "A")
    ns = NeighborSearch(searchable_atoms)
    interface_residues = set()
    for atom in cdr_atoms:
        interface_residues.update(ns.search(atom.coord, radius, level="R"))
    return list(interface_residues)


def cdr_center_of_mass(cdr_residues) -> np.ndarray:
    coords = [atom.coord for res in cdr_residues for atom in res.get_atoms()]
    return np.mean(coords, axis=0)


def representative_atom(residue):
    if "CB" in residue:
        return residue["CB"]
    if "CA" in residue:
        return residue["CA"]
    atoms = list(residue.get_atoms())
    return atoms[0] if atoms else None


def build_mutation_targets(interface_residues, cdr_com: np.ndarray) -> List[MutationTarget]:
    """Deduplicate interface residues by (chain, integer resnum) and build targets.

    If IMGT insertion codes produced multiple residues with the same integer
    resnum on the same chain (MadraX cannot tell them apart - see module
    docstring), only the first one encountered is kept and the rest are
    dropped with a warning, since MadraX's own parser would silently merge
    them into a single residue anyway.
    """
    seen: Dict[Tuple[str, int], MutationTarget] = {}
    for residue in interface_residues:
        wt_aa = residue.get_resname().upper()
        if wt_aa not in AMINO_ACID_SET:
            continue
        chain_id = residue.get_parent().id
        resnum = residue.id[1]
        key = (chain_id, resnum)
        if key in seen:
            LOGGER.warning(
                "Duplicate IMGT integer position %s%s (insertion code variant) "
                "- MadraX cannot distinguish these; keeping the first one only.",
                chain_id, resnum,
            )
            continue
        rep_atom = representative_atom(residue)
        distance = float(np.linalg.norm(rep_atom.coord - cdr_com)) if rep_atom is not None else -1.0
        seen[key] = MutationTarget(chain=chain_id, resnum=resnum, wt_aa=wt_aa, distance_to_cdr=distance)
    return sorted(seen.values(), key=lambda t: (t.chain, t.resnum))


class TcrPmhcDataset(Dataset):
    """CPU-only stage: parse a PDB, find CDRs + interface residues, write a clean copy.

    Kept free of CUDA/madrax-forward-pass work so it can run safely inside
    multiple `DataLoader` worker processes while the GPU forward passes stay
    serialized in the main process.
    """

    def __init__(
        self,
        pdb_files: Sequence[str],
        clean_dir: Path,
        tcr_chains: Sequence[str],
        complex_chains: Sequence[str],
        interface_radius: float,
    ):
        self.pdb_files = list(pdb_files)
        self.clean_dir = clean_dir
        self.tcr_chains = tuple(tcr_chains)
        self.complex_chains = tuple(complex_chains)
        self.interface_radius = interface_radius

    def __len__(self) -> int:
        return len(self.pdb_files)

    def __getitem__(self, idx: int) -> PdbJob:
        pdb_path = Path(self.pdb_files[idx])
        pdb_id = pdb_path.stem.lower()
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_id, str(pdb_path))

            cdr_residues = identify_cdr_loops(structure, self.tcr_chains)
            if not cdr_residues:
                return PdbJob(pdb_id, None, error="No CDR residues found (check chain IDs / IMGT numbering)")

            interface_residues = get_interface_residues(
                structure, cdr_residues, self.interface_radius, self.complex_chains
            )
            if not interface_residues:
                return PdbJob(pdb_id, None, error="No interface residues found within radius")

            cdr_com = cdr_center_of_mass(cdr_residues)
            targets = build_mutation_targets(interface_residues, cdr_com)
            if not targets:
                return PdbJob(pdb_id, None, error="Interface residues were all non-standard")

            clean_path = self.clean_dir / f"{pdb_id}.pdb"
            clean_structure(pdb_path, clean_path, self.complex_chains)

            return PdbJob(pdb_id, str(clean_path), targets)
        except Exception as exc:  # noqa: BLE001 - report and keep the pipeline alive
            return PdbJob(pdb_id, None, error=f"{type(exc).__name__}: {exc}")


def load_madrax_inputs(clean_pdb_path: str):
    """Parse a single cleaned PDB into MadraX's (coords, atom_names, pdb_names) batch-of-1 form."""
    clean_path = Path(clean_pdb_path)
    with tempfile.TemporaryDirectory(prefix="madrax_input_") as tmp_dir:
        staged = Path(tmp_dir) / clean_path.name
        shutil.copy2(clean_path, staged)
        coords, atom_names, pdb_names = madrax_utils.parsePDB(tmp_dir)
    return coords, atom_names, pdb_names


def run_mutation_batch(
    coords: torch.Tensor,
    atom_names: List[List[str]],
    directives: List[str],
    force_field: ForceField,
    device: torch.device,
) -> torch.Tensor:
    """Mutate one structure into `len(directives)` independent point mutants and
    score WT + all mutants in a single MadraX forward pass.

    Returns a 1D tensor of length `len(directives)` with MadraX_ddG (mutant
    total energy - WT total energy) for each directive, in the same order.
    """
    mutation_list = [[[directive] for directive in directives]]
    mutated_coords, mutated_atom_names = madrax_mutate(coords, atom_names, mutation_list)
    info_tensors = dataStructures.create_info_tensors(mutated_atom_names, device=str(device))
    with torch.no_grad():
        energy = force_field(mutated_coords.to(device), info_tensors)
    # energy shape: (Batch=1, nChains, nResi, nAlt, nEnergyComponents)
    # alt index 0 is always the wild type; 1..N are the requested mutants, in order.
    total_per_alt = energy[0].sum(dim=(0, 1, 3))
    ddg = total_per_alt[1:] - total_per_alt[0]
    return ddg.detach().cpu()


def chunked(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def process_job(
    job: PdbJob,
    force_field: ForceField,
    device: torch.device,
    mutation_batch_size: int,
    clash_threshold: float,
    writer: csv.DictWriter,
    failure_log: csv.DictWriter,
) -> int:
    if job.error or job.clean_pdb_path is None:
        LOGGER.warning("Skipping %s: %s", job.pdb_id, job.error)
        failure_log.writerow({"PDB_ID": job.pdb_id, "Reason": job.error or "unknown"})
        return 0

    try:
        coords, atom_names, _ = load_madrax_inputs(job.clean_pdb_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to parse %s for MadraX: %s", job.pdb_id, exc)
        failure_log.writerow({"PDB_ID": job.pdb_id, "Reason": f"MadraX parse failure: {exc}"})
        return 0

    # Flatten every (residue, mutant) pair for the whole structure, then feed
    # them through MadraX in large batched forward passes instead of one
    # forward pass per single mutation -- this is the throughput-critical
    # step for reaching ~1M datapoints in a reasonable wall-clock time.
    flat_jobs: List[Tuple[MutationTarget, str]] = []
    for target in job.targets:
        for mutant_aa in AMINO_ACIDS:
            if mutant_aa == target.wt_aa:
                continue
            directive = f"{target.resnum}_{target.chain}_{mutant_aa}"
            flat_jobs.append((target, mutant_aa, directive))

    written = 0
    batch_size = mutation_batch_size
    pending = list(flat_jobs)
    while pending:
        batch = pending[:batch_size]
        directives = [item[2] for item in batch]
        try:
            ddg_values = run_mutation_batch(coords, atom_names, directives, force_field, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 1:
                LOGGER.error("OOM on %s even at batch size 1; skipping remaining mutants.", job.pdb_id)
                failure_log.writerow({"PDB_ID": job.pdb_id, "Reason": "CUDA OOM at batch size 1"})
                pending = pending[1:]
                continue
            batch_size = max(1, batch_size // 2)
            LOGGER.warning("CUDA OOM on %s; reducing mutation batch size to %d and retrying.", job.pdb_id, batch_size)
            continue
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("MadraX forward pass failed for %s batch starting at %s: %s", job.pdb_id, directives[0], exc)
            failure_log.writerow({"PDB_ID": job.pdb_id, "Reason": f"forward pass failure: {exc}"})
            pending = pending[len(batch) :]
            continue

        for (target, mutant_aa, _directive), ddg_tensor in zip(batch, ddg_values):
            ddg = float(ddg_tensor.item())
            if not np.isfinite(ddg) or abs(ddg) > clash_threshold:
                LOGGER.warning(
                    "Severe steric clash for %s %s%d %s->%s (ddG=%s); filtering out.",
                    job.pdb_id, target.chain, target.resnum, target.wt_aa, mutant_aa, ddg,
                )
                failure_log.writerow(
                    {"PDB_ID": job.pdb_id, "Reason": f"clash at {target.chain}{target.resnum}{target.wt_aa}->{mutant_aa} (ddG={ddg})"}
                )
                continue
            writer.writerow(
                {
                    "PDB_ID": job.pdb_id,
                    "Chain": target.chain,
                    "Residue_Position": target.resnum,
                    "WT_Amino_Acid": target.wt_aa,
                    "Mutant_Amino_Acid": mutant_aa,
                    "Distance_to_CDR_center": round(target.distance_to_cdr, 3),
                    "MadraX_ddG": round(ddg, 4),
                }
            )
            written += 1

        pending = pending[len(batch) :]
        del ddg_values
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return written


def load_done_pdb_ids(done_path: Path) -> set:
    if not done_path.exists():
        return set()
    return {line.strip() for line in done_path.read_text().splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic TCR-pMHC ddG dataset using MadraX.")
    parser.add_argument("--pdb_dir", type=str, default="stcrdab_structures/", help="Directory containing STCRDab PDB files.")
    parser.add_argument("--output", type=str, default="tcr_pmhc_interface_ddg.csv", help="Output CSV file path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of PDB files to process (useful for dry runs).")
    parser.add_argument("--interface_radius", type=float, default=10.0, help="Angstrom radius defining the binding interface around CDR atoms.")
    parser.add_argument("--tcr_chains", type=str, default=",".join(DEFAULT_TCR_CHAINS), help="Comma-separated chain IDs for the TCR alpha/beta chains.")
    parser.add_argument("--complex_chains", type=str, default=",".join(DEFAULT_COMPLEX_CHAINS), help="Comma-separated chain IDs that make up the full TCR-pMHC complex.")
    parser.add_argument("--mutation_batch_size", type=int, default=75, help="Number of point mutants scored per MadraX forward pass.")
    parser.add_argument("--clash_threshold", type=float, default=1000.0, help="|ddG| (kcal/mol) above which a mutant is treated as a steric clash and filtered out.")
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="CPU worker processes for PDB parsing / interface detection.")
    parser.add_argument("--clean_dir", type=str, default=None, help="Directory to cache cleaned, chain-filtered PDBs. Defaults to <pdb_dir>/_clean.")
    parser.add_argument("--resume", action="store_true", help="Skip PDB IDs already recorded as done in <output>.done.")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    output_csv = Path(args.output)
    tcr_chains = tuple(c.strip() for c in args.tcr_chains.split(",") if c.strip())
    complex_chains = tuple(c.strip() for c in args.complex_chains.split(",") if c.strip())
    clean_dir = Path(args.clean_dir) if args.clean_dir else pdb_dir / "_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    if not pdb_dir.exists():
        pdb_dir.mkdir(parents=True)
        LOGGER.info("Directory %s created. Please populate it with STCRDab PDB files.", pdb_dir)

    pdb_files = sorted(glob.glob(str(pdb_dir / "*.pdb")))

    done_path = output_csv.with_suffix(output_csv.suffix + ".done")
    done_ids = load_done_pdb_ids(done_path) if args.resume else set()
    if done_ids:
        pdb_files = [f for f in pdb_files if Path(f).stem.lower() not in done_ids]
        LOGGER.info("Resuming: skipping %d already-completed PDB files.", len(done_ids))

    if args.limit:
        pdb_files = pdb_files[: args.limit]
        LOGGER.info("Limiting to %d PDB files for this run.", args.limit)

    LOGGER.info("Found %d PDB files to process.", len(pdb_files))
    if not pdb_files:
        LOGGER.error("No PDB files left to process. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        LOGGER.warning("CUDA is not available; running on CPU. This will be far too slow for the full 1M-datapoint run.")
    LOGGER.info("Using device: %s", device)

    force_field = ForceField(device=str(device)).to(device)
    force_field.eval()

    dataset = TcrPmhcDataset(pdb_files, clean_dir, tcr_chains, complex_chains, args.interface_radius)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0],
    )

    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    failure_log_path = output_csv.with_name(output_csv.stem + "_failures.csv")
    failure_write_header = not failure_log_path.exists() or failure_log_path.stat().st_size == 0

    total_written = 0
    with open(output_csv, "a", newline="") as out_f, open(failure_log_path, "a", newline="") as fail_f, open(done_path, "a") as done_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        failure_writer = csv.DictWriter(fail_f, fieldnames=["PDB_ID", "Reason"])
        if failure_write_header:
            failure_writer.writeheader()

        for i, job in enumerate(loader):
            LOGGER.info("[%d/%d] Processing %s (%d interface residues)", i + 1, len(dataset), job.pdb_id, len(job.targets))
            written = process_job(job, force_field, device, args.mutation_batch_size, args.clash_threshold, writer, failure_writer)
            total_written += written
            out_f.flush()
            fail_f.flush()
            done_f.write(job.pdb_id + "\n")
            done_f.flush()
            LOGGER.info("%s -> %d ddG points (running total: %d)", job.pdb_id, written, total_written)

    LOGGER.info("Successfully generated %d synthetic ddG points.", total_written)
    LOGGER.info("Results written to %s (failures logged to %s).", output_csv, failure_log_path)


if __name__ == "__main__":
    main()
