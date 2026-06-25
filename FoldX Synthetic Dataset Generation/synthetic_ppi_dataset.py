from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from Bio.PDB import PDBIO, PDBParser, Select, ShrakeRupley
try:
    from Bio.PDB.DSSP import DSSP
except Exception:  # pragma: no cover - optional external dependency
    DSSP = None  # type: ignore[assignment]
from torch_geometric.data import Data

try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except Exception:  # pragma: no cover - optional at import time
    AutoModelForMaskedLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from madrax.ForceField import ForceField
    from madrax import dataStructures, utils
    from madrax.mutate.StructureOptimizer import optimize
    from madrax.mutate.mutatingEngine import mutate
except Exception:  # pragma: no cover - optional at import time
    ForceField = None  # type: ignore[assignment]
    dataStructures = None  # type: ignore[assignment]
    utils = None  # type: ignore[assignment]
    optimize = None  # type: ignore[assignment]
    mutate = None  # type: ignore[assignment]


DOCKGROUND_BOUND_CSV_URL = "https://dockground.compbio.ku.edu/downloads/bound/bound_downloads/protein-pairwise-interactions.csv.gz"

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AMINO_ACIDS = list(THREE_TO_ONE.keys())
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}


@dataclass(frozen=True)
class ResidueRecord:
    chain_id: str
    resid: int
    resname: str
    center: np.ndarray
    sasa: float
    seq_index: int
    sequence_letter: str


class StandardResidueSelect(Select):
    def accept_residue(self, residue):
        return residue.id[0] == " "


def patch_torch_scheduler_compatibility() -> None:
    scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    if getattr(scheduler_cls, "_copilot_verbose_compat", False):
        return

    def _compat_reduce_lr_on_plateau(*args, verbose=None, **kwargs):
        del verbose
        return scheduler_cls(*args, **kwargs)

    _compat_reduce_lr_on_plateau._copilot_verbose_compat = True  # type: ignore[attr-defined]
    torch.optim.lr_scheduler.ReduceLROnPlateau = _compat_reduce_lr_on_plateau  # type: ignore[assignment]


def configure_logging(verbose: bool = True) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="[%(levelname)s] %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading %s", url)
    urllib.request.urlretrieve(url, destination)
    return destination


def prepare_dockground_pool(cache_dir: Path, max_candidates: int = 20) -> List[Path]:
    import gzip

    structure_dir = ensure_dir(cache_dir / "structures")
    csv_path = download_file(DOCKGROUND_BOUND_CSV_URL, ensure_dir(cache_dir / "metadata") / "protein-pairwise-interactions.csv.gz")
    candidates: List[Path] = []
    seen_codes = set()
    with gzip.open(csv_path, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    random.shuffle(rows)
    for row in rows:
        pdb_code = (row.get("PDB Code") or "").strip().lower()
        buried_area_text = (row.get("Mean area buried by each chain") or "0").strip()
        try:
            buried_area = float(buried_area_text)
        except ValueError:
            buried_area = 0.0
        if not pdb_code or pdb_code in seen_codes or buried_area < 250.0:
            continue
        seen_codes.add(pdb_code)
        destination = structure_dir / f"{pdb_code}.pdb"
        url = f"https://files.rcsb.org/download/{pdb_code.upper()}.pdb"
        try:
            download_file(url, destination)
            candidates.append(destination)
        except Exception as exc:
            logging.info("Skipped %s: %s", pdb_code, exc)
        if len(candidates) >= max_candidates:
            break
    return candidates


def prepare_seed_structure(seed_pdb: str, cache_dir: Path) -> Path:
    seed_path = Path(seed_pdb)
    if seed_path.exists():
        return seed_path
    structure_dir = ensure_dir(cache_dir / "structures")
    destination = structure_dir / f"{seed_pdb.lower()}.pdb"
    url = f"https://files.rcsb.org/download/{seed_pdb.upper()}.pdb"
    return download_file(url, destination)


def strip_hetatm(source_path: Path, destination_path: Path) -> Path:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(source_path.stem, str(source_path))
    io = PDBIO()
    io.set_structure(structure)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(destination_path), select=StandardResidueSelect())
    return destination_path


def load_structure(path: Path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def residue_center(residue) -> np.ndarray:
    coords = [atom.coord for atom in residue.get_atoms() if getattr(atom, "element", "") != "H"]
    if not coords:
        coords = [atom.coord for atom in residue.get_atoms()]
    return np.asarray(coords, dtype=np.float32).mean(axis=0)


def compute_sasa_lookup(clean_pdb_path: Path, structure) -> Dict[Tuple[str, int], float]:
    if DSSP is not None:
        try:
            model = next(structure.get_models())
            dssp = DSSP(model, str(clean_pdb_path))
            lookup: Dict[Tuple[str, int], float] = {}
            for key in dssp.keys():
                chain_id = key[0]
                resid = int(key[1][1])
                lookup[(chain_id, resid)] = float(dssp[key][3])
            return lookup
        except Exception:
            pass
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    lookup: Dict[Tuple[str, int], float] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                lookup[(chain.id, int(residue.id[1]))] = float(getattr(residue, "sasa", 0.0))
    return lookup


def identify_interface_residues(structure, cutoff: float = 8.0) -> List[Tuple[str, int]]:
    model = next(structure.get_models())
    chains = [chain for chain in model if chain.id.strip()]
    interface = set()
    for i, chain_a in enumerate(chains):
        for chain_b in chains[i + 1 :]:
            for residue_a in chain_a:
                if residue_a.id[0] != " ":
                    continue
                atoms_a = list(residue_a.get_atoms())
                if not atoms_a:
                    continue
                for residue_b in chain_b:
                    if residue_b.id[0] != " ":
                        continue
                    atoms_b = list(residue_b.get_atoms())
                    if not atoms_b:
                        continue
                    if any(atom_a - atom_b <= cutoff for atom_a in atoms_a for atom_b in atoms_b):
                        interface.add((chain_a.id, int(residue_a.id[1])))
                        interface.add((chain_b.id, int(residue_b.id[1])))
    return sorted(interface)


def build_residue_records(structure, sasa_lookup: Dict[Tuple[str, int], float]) -> List[ResidueRecord]:
    records: List[ResidueRecord] = []
    for model in structure:
        for chain in model:
            seq_index = 0
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                resname = residue.get_resname().upper()
                if resname not in THREE_TO_ONE:
                    continue
                records.append(
                    ResidueRecord(
                        chain_id=chain.id,
                        resid=int(residue.id[1]),
                        resname=resname,
                        center=residue_center(residue),
                        sasa=float(sasa_lookup.get((chain.id, int(residue.id[1])), 0.0)),
                        seq_index=seq_index,
                        sequence_letter=THREE_TO_ONE[resname],
                    )
                )
                seq_index += 1
    return records


def choose_mutant_residue(wildtype: str) -> str:
    return random.choice([res for res in AMINO_ACIDS if res != wildtype])


def mutation_code(chain_id: str, resid: int, resname: str) -> str:
    return f"{resid}_{chain_id}_{resname}"


def load_esm_model(model_name: str, device: torch.device):
    if AutoTokenizer is None or AutoModelForMaskedLM is None:
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def esm_embeddings(sequence: str, tokenizer, model, device: torch.device) -> torch.Tensor:
    if tokenizer is None or model is None:
        return torch.zeros((len(sequence), 1), dtype=torch.float32)
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
    return output.hidden_states[-1][0, 1:-1].detach().cpu().float()


def build_esm_lookup(records: List[ResidueRecord], tokenizer, model, device: torch.device) -> Dict[Tuple[str, int], torch.Tensor]:
    lookup: Dict[Tuple[str, int], torch.Tensor] = {}
    by_chain: Dict[str, List[ResidueRecord]] = {}
    for record in records:
        by_chain.setdefault(record.chain_id, []).append(record)
    for chain_id, chain_records in by_chain.items():
        ordered = sorted(chain_records, key=lambda item: item.seq_index)
        sequence = "".join(record.sequence_letter for record in ordered)
        embeddings = esm_embeddings(sequence, tokenizer, model, device)
        for idx, record in enumerate(ordered):
            if idx < embeddings.shape[0]:
                lookup[(record.chain_id, record.resid)] = embeddings[idx]
    return lookup


def rbf(distance: float, centers: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    return torch.exp(-gamma * (distance - centers) ** 2)


def local_frame_scalars(center_a: np.ndarray, center_b: np.ndarray) -> torch.Tensor:
    delta = torch.tensor(center_b - center_a, dtype=torch.float32)
    dist = torch.norm(delta)
    if dist.item() == 0:
        return torch.zeros(3, dtype=torch.float32)
    return torch.stack([dist, delta[0] / dist, delta[1] / dist])


def graph_from_mutation(
    records: List[ResidueRecord],
    mutation_record: ResidueRecord,
    mutant_resname: str,
    ddg: float,
    esm_lookup: Dict[Tuple[str, int], torch.Tensor],
) -> Data:
    selected = [record for record in records if np.linalg.norm(record.center - mutation_record.center) <= 10.0]
    if not selected:
        selected = [mutation_record]

    node_features: List[torch.Tensor] = []
    positions: List[torch.Tensor] = []
    for record in selected:
        mutation_flag = 1.0 if (record.chain_id == mutation_record.chain_id and record.resid == mutation_record.resid) else 0.0
        original_onehot = torch.zeros(len(AMINO_ACIDS), dtype=torch.float32)
        original_onehot[AA_TO_INDEX[record.resname]] = 1.0
        mutant_onehot = torch.zeros(len(AMINO_ACIDS), dtype=torch.float32)
        if mutation_flag == 1.0:
            mutant_onehot[AA_TO_INDEX[mutant_resname]] = 1.0
        esm = esm_lookup.get((record.chain_id, record.resid), torch.zeros(1, dtype=torch.float32)).flatten().float()
        feature = torch.cat(
            [
                torch.tensor([mutation_flag], dtype=torch.float32),
                original_onehot,
                mutant_onehot,
                torch.tensor([record.sasa], dtype=torch.float32),
                esm,
            ]
        )
        node_features.append(feature)
        positions.append(torch.tensor(record.center, dtype=torch.float32))

    x = torch.stack(node_features, dim=0)
    pos = torch.stack(positions, dim=0)
    centers = torch.linspace(0.0, 10.0, steps=8)
    edge_index: List[List[int]] = []
    edge_attr: List[torch.Tensor] = []
    for i in range(len(selected)):
        for j in range(len(selected)):
            if i == j:
                continue
            distance = torch.norm(pos[i] - pos[j]).item()
            seq_distance = abs(selected[i].seq_index - selected[j].seq_index)
            link_flag = 1.0 if selected[i].chain_id != selected[j].chain_id else 0.0
            frame = local_frame_scalars(selected[i].center, selected[j].center)
            attr = torch.cat([rbf(distance, centers), torch.tensor([seq_distance, link_flag], dtype=torch.float32), frame])
            edge_index.append([i, j])
            edge_attr.append(attr)

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.stack(edge_attr, dim=0)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 13), dtype=torch.float32)

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
        y=torch.tensor([ddg], dtype=torch.float32),
        u=torch.tensor([7.0, 298.15], dtype=torch.float32),
    )
    data.mutation_chain = mutation_record.chain_id
    data.mutation_resid = mutation_record.resid
    data.mutation_wildtype = mutation_record.resname
    data.mutation_mutant = mutant_resname
    return data


def pdb_to_madrax_inputs(clean_pdb_path: Path, device: torch.device):
    if utils is None or dataStructures is None:
        raise RuntimeError("madrax is not available")
    input_dir = clean_pdb_path.parent / f"{clean_pdb_path.stem}_madrax_input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clean_pdb_path, input_dir / clean_pdb_path.name)
    coords, atnames, pdb_names = utils.parsePDB(str(input_dir))
    info_tensors = dataStructures.create_info_tensors(atnames, device=str(device))
    return coords, atnames, pdb_names, info_tensors


def ddg_from_energy(energy: torch.Tensor) -> float:
    if energy.ndim < 5:
        return float(energy.mean().item())
    wt = energy[..., 0, :].mean().item()
    mut = energy[..., -1, :].mean().item()
    return float(mut - wt)


def prepare_clean_structure(source_path: Path, clean_dir: Path) -> Path:
    clean_path = clean_dir / source_path.name.replace(" ", "_")
    if clean_path.exists():
        return clean_path
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(source_path.stem, str(source_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(clean_path), select=StandardResidueSelect())
    return clean_path


def run_madrax_mutation(clean_path: Path, mutation_label: str, device: torch.device):
    if ForceField is None or dataStructures is None or mutate is None or optimize is None:
        raise RuntimeError("madrax is required for mutation evaluation")
    patch_torch_scheduler_compatibility()
    coords, atnames, pdb_names, _ = pdb_to_madrax_inputs(clean_path, device)
    mutated_coords, mutated_atnames = mutate(coords, atnames, [[[mutation_label]]])
    mutated_info = dataStructures.create_info_tensors(mutated_atnames, device=str(device))
    force_field = ForceField(device=str(device))
    energy, optimized_coords = optimize(
        force_field,
        mutated_coords.to(device),
        mutated_info,
        epochs=3,
        learning_rate=0.001,
        backbone_rotation=False,
    )
    return energy, optimized_coords, mutated_atnames, pdb_names


def generate_dataset(
    output_dir: Path,
    cache_dir: Path,
    target_samples: int = 100,
    seed: int = 7,
    interface_cutoff: float = 8.0,
    esm_model_name: str = "facebook/esm2_t12_35M_UR50D",
    seed_pdb: Optional[str] = None,
) -> List[Data]:
    del output_dir
    set_seed(seed)
    device = get_device()
    logging.info("Using device: %s", device)

    if ForceField is None or dataStructures is None or utils is None:
        raise RuntimeError("madrax must be installed for this generator")

    if seed_pdb:
        raw_pool = [prepare_seed_structure(seed_pdb, cache_dir)]
    else:
        raw_pool = prepare_dockground_pool(cache_dir)
        if not raw_pool:
            raise RuntimeError("No DOCKGROUND structures were downloaded")

    clean_dir = ensure_dir(cache_dir / "cleaned")
    mutant_dir = ensure_dir(cache_dir / "mutants")
    tokenizer, esm_model = load_esm_model(esm_model_name, device)
    dataset: List[Data] = []
    attempt = 0

    while len(dataset) < target_samples:
        attempt += 1
        source_path = random.choice(raw_pool)
        logging.info("Attempt %d | %s | accepted=%d/%d", attempt, source_path.name, len(dataset), target_samples)

        try:
            clean_path = prepare_clean_structure(source_path, clean_dir)
            structure = load_structure(clean_path)
            sasa_lookup = compute_sasa_lookup(clean_path, structure)
            interface = identify_interface_residues(structure, cutoff=interface_cutoff)
            if not interface:
                logging.info("Rejected: no interface residues found")
                continue

            records = build_residue_records(structure, sasa_lookup)
            record_map = {(record.chain_id, record.resid): record for record in records}
            candidates = [record_map[key] for key in interface if key in record_map]
            if not candidates:
                logging.info("Rejected: interface residues were not mapped to residues")
                continue

            mutation_record = random.choice(candidates)
            mutant_resname = choose_mutant_residue(mutation_record.resname)
            mutation_label = f"{mutation_record.resid}_{mutation_record.chain_id}_{mutant_resname}"
            energy, optimized_coords, mutated_atnames, pdb_names = run_madrax_mutation(clean_path, mutation_label, device)
            ddg = ddg_from_energy(energy)

            try:
                utils.writepdb(optimized_coords.cpu().data, mutated_atnames, output_folder=str(mutant_dir), pdb_names=pdb_names)
            except Exception:
                pass

            esm_lookup = build_esm_lookup(records, tokenizer, esm_model, device)
            data = graph_from_mutation(records, mutation_record, mutant_resname, ddg, esm_lookup)
            data.source_pdb = str(clean_path)
            data.ddg = ddg
            dataset.append(data)
            logging.info("Accepted %s -> %s | ddG=%.4f", mutation_label, mutant_resname, ddg)
        except Exception as exc:
            logging.exception("Rejected with error: %s", exc)
            continue

    return dataset


def save_dataset(dataset: List[Data], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 100 synthetic PPI mutation samples using MadraX")
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic_ppi_dataset.pt"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--target-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--interface-cutoff", type=float, default=8.0)
    parser.add_argument("--esm-model", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--seed-pdb", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(not args.quiet)
    dataset = generate_dataset(
        output_dir=args.output.parent,
        cache_dir=args.cache_dir,
        target_samples=args.target_samples,
        seed=args.seed,
        interface_cutoff=args.interface_cutoff,
        esm_model_name=args.esm_model,
        seed_pdb=args.seed_pdb,
    )
    save_dataset(dataset, args.output)
    logging.info("Saved %d samples to %s", len(dataset), args.output)


if __name__ == "__main__":
    main()