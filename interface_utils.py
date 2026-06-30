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
   the downstream engine can actually distinguish.
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
    # Minimum atom-atom distance from this residue to the *opposite* binding
    # partner (TCR<->pMHC). Small (<~5 A) => a genuine inter-chain contact;
    # large => second-shell residue pulled in by the broad CDR radius. Kept as
    # an analysis column so true contacts can be filtered downstream. -1.0 means
    # "not computed / no opposite-partner atoms present".
    min_interchain_dist: float = -1.0

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


def _group_atom_coords(structure, chain_ids: Sequence[str]) -> np.ndarray:
    """Stack the coordinates of every standard-residue atom on ``chain_ids``."""
    model = next(structure.get_models())
    coords = [
        atom.coord
        for chain in model
        if chain.id in set(chain_ids)
        for residue in chain
        if residue.id[0] == " "
        for atom in residue.get_atoms()
    ]
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)


def _min_dist_to_group(residue, group_coords: np.ndarray) -> float:
    """Minimum atom-atom distance from ``residue`` to a precomputed coord array."""
    if group_coords.shape[0] == 0:
        return -1.0
    res_coords = np.asarray([atom.coord for atom in residue.get_atoms()], dtype=float)
    if res_coords.shape[0] == 0:
        return -1.0
    # (n_res_atoms, n_group_atoms) pairwise distances; residues are tiny so this
    # broadcast is cheap even against a few thousand opposite-partner atoms.
    dists = np.linalg.norm(res_coords[:, None, :] - group_coords[None, :, :], axis=2)
    return float(dists.min())


def build_mutation_targets(
    interface_residues,
    cdr_com: np.ndarray,
    tcr_chains: Sequence[str],
    tcr_coords: np.ndarray,
    pmhc_coords: np.ndarray,
) -> List[MutationTarget]:
    """Deduplicate interface residues by (chain, integer resnum) and build targets.

    For each residue, also records the minimum distance to the *opposite* binding
    partner: residues on a TCR chain are measured against the pMHC atoms and vice
    versa, so the column flags genuine inter-chain contacts vs. second-shell hits.
    """
    tcr_set = set(tcr_chains)
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
                "- downstream engines cannot distinguish these; keeping the first one only.",
                chain_id, resnum,
            )
            continue
        rep_atom = representative_atom(residue)
        distance = float(np.linalg.norm(rep_atom.coord - cdr_com)) if rep_atom is not None else -1.0
        opposite = pmhc_coords if chain_id in tcr_set else tcr_coords
        min_interchain = _min_dist_to_group(residue, opposite)
        seen[key] = MutationTarget(
            chain=chain_id,
            resnum=resnum,
            wt_aa=wt_aa,
            distance_to_cdr=distance,
            min_interchain_dist=min_interchain,
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
    # Partner groups for the inter-chain distance: TCR (tcr_chains) vs. pMHC
    # (everything else in the complex). Atom coords are gathered once per group.
    pmhc_chains = [c for c in complex_chains if c not in set(tcr_chains)]
    tcr_coords = _group_atom_coords(structure, tcr_chains)
    pmhc_coords = _group_atom_coords(structure, pmhc_chains)
    return build_mutation_targets(interface_residues, cdr_com, tcr_chains, tcr_coords, pmhc_coords)
