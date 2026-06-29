"""
Shared TCR-pMHC interface-detection utilities.

This module is the single source of truth for "which residues sit at the TCR
binding interface and are therefore worth mutating". Both data-generation
pipelines import from here so that the MadraX (Tier 1) and Rosetta Flex ddG
(Tier 2) datasets cover the *same* set of interface positions:

* ``generate_tcr_pmhc_dataset.py``  (MadraX saturation mutagenesis)
* ``rosetta_flex/make_mutfiles.py``  (Rosetta Flex ddG job enumeration)

The geometry is identical in both cases:

1. Identify the 6 CDR loops on the TCR alpha/beta chains using IMGT residue
   numbers (CDR1: 27-38, CDR2: 56-65, CDR3: 105-117). This assumes the input
   structure has already been IMGT-renumbered (as STCRDab structures are), so
   CDR membership is read directly off a residue's PDB number rather than
   guessed from list position (which breaks on missing residues / insertion
   codes).
2. Define the binding interface as every standard residue (on the TCR, peptide,
   or MHC) within ``radius`` Angstrom of any CDR atom.
3. Deduplicate by (chain, integer resnum). MadraX's plain-text PDB parser keys
   atoms purely by integer residue number, so IMGT insertion codes (e.g.
   "111A", "111B") collapse onto the same integer position; we mirror that
   limitation deliberately so a mutation directive always refers to a residue
   the downstream engine can actually distinguish. The kept representative's
   insertion code is preserved on ``MutationTarget.icode`` so the Rosetta tier
   (which *can* address insertion codes) can target the exact residue in its
   resfile, while MadraX keeps using the integer position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from Bio.PDB import NeighborSearch, Selection

LOGGER = logging.getLogger("interface_utils")

# Standard 20 amino acids, 3-letter codes (matches both BioPython resnames and
# the residue-name tokens MadraX expects in its "resNum_chain_RESNAME" mutation
# directives, e.g. "39_D_GLY").
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AMINO_ACID_SET = set(AMINO_ACIDS)

# 3-letter -> 1-letter, used by the Rosetta mutfile writer (Rosetta resfiles
# and flex_ddG mutfiles speak single-letter codes).
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# STCRDab's standardized chain identifiers after IMGT renumbering.
DEFAULT_TCR_CHAINS = ("D", "E")  # TCR alpha, TCR beta
DEFAULT_COMPLEX_CHAINS = ("A", "B", "C", "D", "E")  # MHC1, MHC2/B2M, peptide, TCR a/b

# IMGT V-domain CDR residue-number ranges (inclusive).
IMGT_CDR_RANGES = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}


@dataclass
class MutationTarget:
    chain: str
    resnum: int
    wt_aa: str  # 3-letter code
    distance_to_cdr: float
    icode: str = ""  # PDB insertion code of the kept residue ("" if none)

    @property
    def wt_aa_one(self) -> str:
        return THREE_TO_ONE.get(self.wt_aa, "X")


def identify_cdr_loops(structure, tcr_chain_ids: Sequence[str]):
    """CDR1/2/3 residues on the TCR alpha/beta chains, by IMGT numbering."""
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
    """All standard residues (on ``allowed_chains``) within ``radius`` A of any CDR atom."""
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
    """Deduplicate interface residues by (chain, integer resnum) and build targets."""
    seen: Dict[Tuple[str, int], MutationTarget] = {}
    for residue in interface_residues:
        wt_aa = residue.get_resname().upper()
        if wt_aa not in AMINO_ACID_SET:
            continue
        chain_id = residue.get_parent().id
        resnum = residue.id[1]
        icode = residue.id[2].strip()  # "" when there is no insertion code
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
        seen[key] = MutationTarget(
            chain=chain_id, resnum=resnum, wt_aa=wt_aa, distance_to_cdr=distance, icode=icode
        )
    return sorted(seen.values(), key=lambda t: (t.chain, t.resnum))


def find_interface_targets(
    structure,
    tcr_chains: Sequence[str] = DEFAULT_TCR_CHAINS,
    complex_chains: Sequence[str] = DEFAULT_COMPLEX_CHAINS,
    radius: float = 10.0,
) -> List[MutationTarget]:
    """Convenience wrapper: structure -> sorted list of interface MutationTargets.

    Returns an empty list (no exception) when no CDRs / interface residues are
    found, so callers can report-and-skip uniformly. Raises only on genuinely
    malformed input.
    """
    cdr_residues = identify_cdr_loops(structure, tcr_chains)
    if not cdr_residues:
        return []
    interface_residues = get_interface_residues(structure, cdr_residues, radius, complex_chains)
    if not interface_residues:
        return []
    cdr_com = cdr_center_of_mass(cdr_residues)
    return build_mutation_targets(interface_residues, cdr_com)
