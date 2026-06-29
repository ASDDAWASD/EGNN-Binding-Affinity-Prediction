"""Count candidate ddG samples a full run would generate, using the pipeline's
own CPU-only interface-detection stage (no GPU / MadraX forward passes)."""
import glob
import statistics
from pathlib import Path

from torch.utils.data import DataLoader
from generate_tcr_pmhc_dataset import (
    TcrPmhcDataset,
    AMINO_ACIDS,
    DEFAULT_TCR_CHAINS,
    DEFAULT_COMPLEX_CHAINS,
)

pdb_dir = Path("stcrdab_structures")
clean_dir = pdb_dir / "_clean"
clean_dir.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(str(pdb_dir / "*.pdb")))
ds = TcrPmhcDataset(files, clean_dir, DEFAULT_TCR_CHAINS, DEFAULT_COMPLEX_CHAINS, 10.0)
loader = DataLoader(ds, batch_size=1, num_workers=8, collate_fn=lambda b: b[0])

muts_per = len(AMINO_ACIDS) - 1  # skip wildtype -> 19
total_targets = total_muts = ok = 0
counts = []
errs = {}
for i, job in enumerate(loader):
    if getattr(job, "targets", None):
        n = len(job.targets)
        ok += 1
        total_targets += n
        total_muts += n * muts_per
        counts.append(n)
    else:
        errs[job.error] = errs.get(job.error, 0) + 1
    if (i + 1) % 25 == 0:
        print(f"...processed {i+1}/{len(ds)}  ok={ok}  cand_muts={total_muts}", flush=True)

print("\n================ FULL-RUN SAMPLE COUNT ================")
print(f"structures_total            : {len(ds)}")
print(f"structures_with_targets     : {ok}")
print(f"structures_skipped          : {len(ds) - ok}")
print(f"total_interface_residues    : {total_targets}")
print(f"mutants_per_residue         : {muts_per}")
print(f"CANDIDATE_SAMPLES (x19)     : {total_muts}")
if counts:
    print(f"targets/structure min/med/mean/max : "
          f"{min(counts)}/{int(statistics.median(counts))}/"
          f"{round(statistics.mean(counts),1)}/{max(counts)}")
if errs:
    print("skip_reasons:")
    for k, v in sorted(errs.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")
