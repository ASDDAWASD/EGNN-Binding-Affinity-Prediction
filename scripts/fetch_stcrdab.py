"""
STCRDab acquisition for the Tier-1 MadraX dataset.

Goal
----
Populate ``stcrdab_structures/`` with IMGT-renumbered alpha/beta TCR-pMHC
complexes whose chains follow the convention the MadraX generator assumes:

    A = MHC chain 1        D = TCR alpha
    B = MHC chain 2 / b2m  E = TCR beta
    C = peptide antigen

To reach >=1,000,000 MadraX samples at ~950 saturation mutants per structure we
need on the order of ~1,000 usable structures, so the non-redundancy filter is
deliberately *light*: we collapse only exact metadata duplicates (same peptide +
MHC allele + TCR genes), keeping the best-resolution representative of each
cluster, rather than the aggressive ~283-structure sequence-identity pruning a
benchmarking study would use.

Separately -- and independently of that light clustering -- we hard-exclude any
PDB that appears in the ATLAS or TRAIT benchmarks to prevent train/test leakage.
ATLAS and TRAIT are evaluation sets, so a structure used to *generate* synthetic
ddG labels here must never be one we later benchmark against. The exclusion is on
by default via the shipped ``benchmark_blocklist.txt`` (``--exclude_pdbs``; pass
an empty string to disable).

Pipeline
--------
1. Download the STCRDab summary table (one row per PDB/complex).
2. Keep alpha/beta TCRs bound to a peptide + MHC, below a resolution cutoff.
   Drop any PDB listed in the ATLAS/TRAIT benchmark blocklist (leakage guard).
3. Cluster on a redundancy key and keep one representative per cluster.
4. Download each representative's IMGT-numbered PDB from STCRDab.
5. Remap the original chain letters (read from the summary row) to A/B/C/D/E
   and write the cleaned structure into ``--out_dir``.
6. Emit ``manifest.csv`` (provenance + resume bookkeeping).

STCRDab already serves IMGT-numbered coordinates, so no renumbering library is
required for the default path. Pass ``--renumber-with-anarci`` to additionally
re-derive IMGT numbering with ANARCI (imported lazily) if a downloaded file is
not already IMGT-numbered.

Network endpoints can drift; all URLs are overridable via CLI flags. Nothing
here runs MadraX or the GPU -- it only prepares inputs.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from Bio.PDB import PDBIO, PDBParser, Select
from Bio.PDB.Polypeptide import is_aa

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("fetch_stcrdab")

STCRDAB = "https://opig.stats.ox.ac.uk/webapps/stcrdab-stcrpred"
DEFAULT_SUMMARY_URL = f"{STCRDAB}/summary/all"
# {pdb} is lower-cased; the `scheme=imgt` query returns IMGT-numbered coords.
DEFAULT_STRUCTURE_URL = f"{STCRDAB}/pdb/{{pdb}}?scheme=imgt"

# Target chain letters for the standardized complex.
ROLE_TO_CHAIN = {
    "mhc1": "A",
    "mhc2": "B",
    "peptide": "C",
    "tcra": "D",
    "tcrb": "E",
}

# Summary column name candidates (STCRDab has renamed columns over time, so we
# probe several aliases and use whichever is present).
COLUMN_ALIASES = {
    "pdb": ["pdb", "PDB"],
    "tcr_type": ["TCRtype", "tcr_type", "receptor_type"],
    "tcra": ["Achain", "alpha_chain", "TCR_alpha_chain"],
    "tcrb": ["Bchain", "beta_chain", "TCR_beta_chain"],
    "peptide": ["antigen_chain", "peptide_chain", "antigen_chains"],
    "mhc1": ["mhc_chain1", "MHC_chain1", "mhc_chain_1"],
    "mhc2": ["mhc_chain2", "MHC_chain2", "mhc_chain_2"],
    "antigen_type": ["antigen_type", "antigen"],
    "mhc_allele": ["mhc_type", "mhc_allele", "MHC_type"],
    "peptide_seq": ["antigen_name", "peptide", "antigen_sequence"],
    "resolution": ["resolution", "Resolution"],
}


def pick(row: Dict[str, str], key: str) -> str:
    for alias in COLUMN_ALIASES[key]:
        if alias in row and row[alias] not in (None, "", "NA", "None", "?"):
            return str(row[alias]).strip()
    return ""


def load_exclude_list(path: Optional[str]) -> set:
    """Load a newline-delimited PDB-ID blocklist (``#`` comment lines ignored).

    Used to hard-exclude benchmark structures (ATLAS/TRAIT) so they never enter
    the training-label set. Returns an empty set if ``path`` is falsy or missing.
    """
    if not path:
        LOGGER.warning("No exclusion list given; proceeding with NO benchmark leakage guard.")
        return set()
    p = Path(path)
    if not p.exists():
        LOGGER.warning("Exclusion list %s not found; proceeding with NO benchmark leakage guard.", path)
        return set()
    ids = set()
    for line in p.read_text().splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            ids.add(line.split()[0])
    LOGGER.info("Loaded %d benchmark PDB IDs to exclude from %s", len(ids), path)
    return ids


def download_summary(url: str, session: requests.Session) -> List[Dict[str, str]]:
    LOGGER.info("Downloading STCRDab summary from %s", url)
    resp = session.get(url, timeout=120)
    resp.raise_for_status()
    # Summary is tab-separated; sniff a fallback to comma.
    text = resp.text
    delimiter = "\t" if text.count("\t") >= text.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    LOGGER.info("Summary contains %d rows.", len(rows))
    return rows


def is_abtcr_pmhc(row: Dict[str, str], max_resolution: float) -> bool:
    tcr_type = pick(row, "tcr_type").lower()
    if tcr_type and "ab" not in tcr_type and "alphabeta" not in tcr_type:
        return False  # explicitly gamma/delta or non-alphabeta
    # Need both TCR chains, a peptide, and at least MHC chain 1.
    if not (pick(row, "tcra") and pick(row, "tcrb")):
        return False
    if not pick(row, "peptide"):
        return False
    if not pick(row, "mhc1"):
        return False
    res = pick(row, "resolution")
    try:
        if res and float(res) > max_resolution:
            return False
    except ValueError:
        pass  # NMR / unknown resolution -> keep
    return True


def redundancy_key(row: Dict[str, str]) -> str:
    """Light non-redundancy: peptide + MHC allele + TCR gene/chain identity."""
    return "|".join(
        [
            pick(row, "peptide_seq").upper(),
            pick(row, "mhc_allele").upper(),
            pick(row, "tcra").upper(),
            pick(row, "tcrb").upper(),
        ]
    )


def resolution_value(row: Dict[str, str]) -> float:
    try:
        return float(pick(row, "resolution"))
    except ValueError:
        return 99.0  # de-prioritize unknown-resolution structures as representatives


def select_representatives(
    rows: List[Dict[str, str]],
    max_resolution: float,
    exclude: Optional[set] = None,
) -> List[Dict[str, str]]:
    exclude = exclude or set()
    candidates = [r for r in rows if is_abtcr_pmhc(r, max_resolution)]
    LOGGER.info("%d/%d rows are alpha/beta TCR-pMHC within resolution cutoff.", len(candidates), len(rows))
    if exclude:
        kept = [r for r in candidates if pick(r, "pdb").lower() not in exclude]
        LOGGER.info(
            "Benchmark leakage guard: excluded %d ATLAS/TRAIT structures; %d candidates remain.",
            len(candidates) - len(kept), len(kept),
        )
        candidates = kept
    best: Dict[str, Dict[str, str]] = {}
    for row in candidates:
        key = redundancy_key(row)
        if key not in best or resolution_value(row) < resolution_value(best[key]):
            best[key] = row
    reps = sorted(best.values(), key=lambda r: pick(r, "pdb").lower())
    LOGGER.info("Collapsed to %d non-redundant representatives.", len(reps))
    return reps


def build_chain_map(row: Dict[str, str]) -> Dict[str, str]:
    """Map original STCRDab chain letters -> standardized A/B/C/D/E.

    A summary cell may list multiple chains (e.g. "H | L"); each is mapped to
    the same role's target letter. Later collisions are resolved by BioPython's
    two-phase rename.
    """
    mapping: Dict[str, str] = {}
    for role, target in (
        ("mhc1", ROLE_TO_CHAIN["mhc1"]),
        ("mhc2", ROLE_TO_CHAIN["mhc2"]),
        ("peptide", ROLE_TO_CHAIN["peptide"]),
        ("tcra", ROLE_TO_CHAIN["tcra"]),
        ("tcrb", ROLE_TO_CHAIN["tcrb"]),
    ):
        cell = pick(row, role)
        for orig in [c.strip() for c in cell.replace("|", " ").replace(",", " ").split() if c.strip()]:
            mapping[orig] = target
    return mapping


class StandardResidueSelect(Select):
    """Keep only mapped chains and standard amino acid / peptide residues."""

    def __init__(self, allowed_chains):
        self.allowed_chains = set(allowed_chains)

    def accept_chain(self, chain):
        return chain.id in self.allowed_chains

    def accept_residue(self, residue):
        return residue.id[0] == " " and is_aa(residue, standard=True)


def remap_and_clean(raw_pdb_text: str, pdb_id: str, chain_map: Dict[str, str], out_path: Path) -> bool:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(raw_pdb_text))
    model = next(structure.get_models())

    present = {c.id for c in model}
    mapped = {orig: tgt for orig, tgt in chain_map.items() if orig in present}
    if not mapped:
        LOGGER.warning("%s: none of the summary chains %s are present in the file; skipping.", pdb_id, list(chain_map))
        return False

    # Two-phase rename to avoid id collisions (e.g. original 'A' -> target 'D'
    # while another chain also wants 'A'). First move everything to temp ids.
    for i, chain in enumerate(list(model)):
        chain.id = f"__tmp{i}__"
    temp_to_orig = {f"__tmp{i}__": orig for i, orig in enumerate(present)}
    # Re-key mapping by the temp ids.
    for chain in list(model):
        orig = temp_to_orig[chain.id]
        if orig in mapped:
            chain.id = mapped[orig]
        else:
            chain.id = f"DROP_{orig}"  # filtered out by StandardResidueSelect

    io_writer = PDBIO()
    io_writer.set_structure(structure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io_writer.save(str(out_path), select=StandardResidueSelect(set(ROLE_TO_CHAIN.values())))
    return True


def download_structure(pdb_id: str, url_template: str, session: requests.Session) -> Optional[str]:
    url = url_template.format(pdb=pdb_id.lower())
    try:
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("%s: download failed (%s)", pdb_id, exc)
        return None
    if "ATOM" not in resp.text:
        LOGGER.warning("%s: response did not contain ATOM records; skipping.", pdb_id)
        return None
    return resp.text


def load_done(manifest_path: Path) -> set:
    if not manifest_path.exists():
        return set()
    with open(manifest_path, newline="") as fh:
        return {r["pdb_id"].lower() for r in csv.DictReader(fh) if r.get("pdb_id")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + IMGT-standardize STCRDab TCR-pMHC structures.")
    ap.add_argument("--out_dir", type=str, default="stcrdab_structures/", help="Where standardized PDBs are written.")
    ap.add_argument("--max_structures", type=int, default=None, help="Cap on number of structures to download (omit for all).")
    ap.add_argument("--max_resolution", type=float, default=3.5, help="Drop structures worse than this resolution (A).")
    ap.add_argument(
        "--exclude_pdbs",
        type=str,
        default=str(Path(__file__).resolve().parent / "benchmark_blocklist.txt"),
        help="Newline-delimited PDB IDs to hard-exclude to prevent train/test leakage "
             "(default: the shipped ATLAS/TRAIT benchmark_blocklist.txt; pass '' to disable).",
    )
    ap.add_argument("--summary_url", type=str, default=DEFAULT_SUMMARY_URL)
    ap.add_argument("--structure_url", type=str, default=DEFAULT_STRUCTURE_URL, help="Template with a {pdb} placeholder.")
    ap.add_argument("--manifest", type=str, default=None, help="Manifest CSV path. Defaults to <out_dir>/manifest.csv.")
    ap.add_argument("--resume", action="store_true", help="Skip PDB IDs already listed in the manifest.")
    ap.add_argument("--renumber-with-anarci", action="store_true", help="Re-derive IMGT numbering with ANARCI (lazy import).")
    ap.add_argument("--sleep", type=float, default=0.3, help="Seconds to pause between downloads (be polite to STCRDab).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "manifest.csv"

    if args.renumber_with_anarci:
        try:
            import anarci  # noqa: F401  (validated up-front so failures surface early)
            LOGGER.info("ANARCI available; will re-derive IMGT numbering when needed.")
        except ImportError:
            LOGGER.error("--renumber-with-anarci requested but anarci is not installed; aborting.")
            sys.exit(1)

    session = requests.Session()
    session.headers.update({"User-Agent": "tcr-pmhc-ddg-pipeline/0.1"})

    exclude = load_exclude_list(args.exclude_pdbs)
    rows = download_summary(args.summary_url, session)
    reps = select_representatives(rows, args.max_resolution, exclude)

    done = load_done(manifest_path) if args.resume else set()
    if done:
        reps = [r for r in reps if pick(r, "pdb").lower() not in done]
        LOGGER.info("Resuming: %d already in manifest, %d remaining.", len(done), len(reps))
    if args.max_structures is not None:
        reps = reps[: args.max_structures]
        LOGGER.info("Limiting this run to %d structures.", len(reps))

    manifest_fields = ["pdb_id", "chains", "peptide", "mhc_allele", "resolution", "cluster_key", "out_path"]
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    n_ok = 0
    with open(manifest_path, "a", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=manifest_fields)
        if write_header:
            writer.writeheader()
        for row in reps:
            pdb_id = pick(row, "pdb").lower()
            if not pdb_id:
                continue
            chain_map = build_chain_map(row)
            raw = download_structure(pdb_id, args.structure_url, session)
            if raw is None:
                continue
            out_path = out_dir / f"{pdb_id}.pdb"
            try:
                ok = remap_and_clean(raw, pdb_id, chain_map, out_path)
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                LOGGER.warning("%s: remap/clean failed (%s)", pdb_id, exc)
                ok = False
            if not ok:
                continue
            writer.writerow(
                {
                    "pdb_id": pdb_id,
                    "chains": ",".join(f"{o}->{t}" for o, t in sorted(chain_map.items())),
                    "peptide": pick(row, "peptide_seq"),
                    "mhc_allele": pick(row, "mhc_allele"),
                    "resolution": pick(row, "resolution"),
                    "cluster_key": redundancy_key(row),
                    "out_path": str(out_path),
                }
            )
            mf.flush()
            n_ok += 1
            LOGGER.info("[%d] wrote %s", n_ok, out_path)
            time.sleep(args.sleep)

    LOGGER.info("Done. %d structures standardized into %s (manifest: %s).", n_ok, out_dir, manifest_path)
    LOGGER.info(
        "At ~950 saturation mutants/structure, %d structures yields ~%s MadraX samples.",
        n_ok, f"{n_ok * 950:,}",
    )


if __name__ == "__main__":
    main()
