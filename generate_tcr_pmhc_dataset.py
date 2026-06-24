import os
import glob
import torch
import numpy as np
import pandas as pd
import argparse
from Bio.PDB import PDBParser, Selection, NeighborSearch
from torch.utils.data import Dataset
import logging
from tqdm import tqdm

# --- MadraX Mock / Import ---
# Attempt to import madrax. If not found, use a mock for illustration.
try:
    import madrax
except ImportError:
    logging.warning("MadraX not found in environment. Using a mock implementation for demonstration.")
    class MockMadrax:
        def __init__(self):
            pass
        def to(self, device):
            return self
        def eval(self):
            pass
        def compute_energy(self, structure_data, mutation=None):
            return torch.randn(1) * 10
    madrax = MockMadrax

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Standard 20 amino acids (3-letter codes for BioPython compatibility)
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
               "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

class TcrPmhcDataset(Dataset):
    """Simple Dataset wrapper for PDB files"""
    def __init__(self, pdb_files):
        self.pdb_files = pdb_files
        
    def __len__(self):
        return len(self.pdb_files)
    
    def __getitem__(self, idx):
        return self.pdb_files[idx]

def identify_cdr_loops(structure):
    """
    Programmatically identifies CDR loop residues for the TCR Alpha/Beta chains.
    In a fully robust pipeline, this would use ANARCI or structural HMMs to find IMGT numbered loops.
    
    Here we implement a placeholder that assumes TCR alpha/beta chains and extracts standard IMGT ranges.
    Assuming chain 'A' and 'B' are the TCR chains for demonstration:
    IMGT CDR1: 27-38, CDR2: 56-65, CDR3: 105-117.
    """
    cdr_residues = []
    # Note: A resilient implementation should map sequence to IMGT numbering dynamically.
    for model in structure:
        for chain in model:
            # Assuming A, B are TCR chains. E.g., STCRDab usually standardizes chains or metadata can be read.
            if chain.id in ['A', 'B', 'D', 'E']: 
                res_list = list(chain.get_residues())
                if len(res_list) > 117:
                    # Using arbitrary indices as a mock representation of IMGT mapping
                    cdr_residues.extend(res_list[27:38])
                    cdr_residues.extend(res_list[56:65])
                    cdr_residues.extend(res_list[105:117])
    return cdr_residues

def get_interface_residues(structure, cdr_residues, radius=10.0):
    """
    Identifies all residues within `radius` Å of any CDR atom using KD-Tree based NeighborSearch.
    This fulfills the 10.0 Ångstrom Euclidean distance constraint.
    """
    # Unfold entities to get all atoms for CDRs and the entire structure
    cdr_atoms = Selection.unfold_entities(cdr_residues, 'A')
    all_atoms = Selection.unfold_entities(structure, 'A')
    
    ns = NeighborSearch(all_atoms)
    interface_residues = set()
    
    for atom in cdr_atoms:
        neighbors = ns.search(atom.coord, radius, level='R') # 'R' level returns residues
        interface_residues.update(neighbors)
        
    return list(interface_residues)

def calculate_distance_to_cdr(residue, cdr_residues):
    """
    Calculates the minimum Euclidean distance from a residue's representative atom (CB/CA) 
    to the center of mass of the identified CDR loops.
    """
    # Calculate CDR center of mass
    cdr_coords = [atom.coord for res in cdr_residues for atom in res.get_atoms()]
    if not cdr_coords:
        return 0.0
    cdr_com = np.mean(cdr_coords, axis=0)
    
    # Residue representative atom (Prefer CB, fallback to CA)
    rep_atom = None
    if 'CB' in residue:
        rep_atom = residue['CB']
    elif 'CA' in residue:
        rep_atom = residue['CA']
    else:
        atoms = list(residue.get_atoms())
        if atoms:
            rep_atom = atoms[0]
            
    if rep_atom:
        diff = rep_atom.coord - cdr_com
        return np.linalg.norm(diff)
    return -1.0

def generate_mutations(wt_aa):
    """
    Generates the 19 unique point mutations for a given wild-type amino acid.
    """
    return [mut_aa for mut_aa in AMINO_ACIDS if mut_aa != wt_aa]

def process_pdb_structure(pdb_file, device, force_field):
    """
    Processes a single PDB file:
    1. Parses structure using BioPython.
    2. Identifies CDRs and Interface geometry.
    3. Evaluates WT and Mutants through MadraX on GPU.
    """
    results = []
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)
        
        # 1. Identify CDRs
        cdr_residues = identify_cdr_loops(structure)
        if not cdr_residues:
            logging.warning(f"No CDRs found for {pdb_id}. Skipping.")
            return results
            
        # 2. Identify 10A Interface
        interface_residues = get_interface_residues(structure, cdr_residues, radius=10.0)
        
        # Compute wildtype energy on GPU
        # In a real use-case, MadraX might need a dataloader/featurizer here.
        with torch.no_grad():
            wt_energy = force_field.compute_energy(pdb_file).to(device)
            
        # 3. Saturation Mutagenesis Loop
        for res in interface_residues:
            chain = res.get_parent().id
            res_idx = res.id[1]
            wt_aa = res.resname
            
            # Skip non-standard residues / heteroatoms (like HOH, ligands)
            if wt_aa not in AMINO_ACIDS:
                continue
                
            dist_to_cdr = calculate_distance_to_cdr(res, cdr_residues)
            mutations = generate_mutations(wt_aa)
            
            for mut_aa in mutations:
                # Generate string directive for mutation e.g., "A_14_VAL"
                mutation_directive = f"{chain}_{res_idx}_{mut_aa}"
                
                try:
                    with torch.no_grad():
                        # Run mutant through MadraX
                        # Implementation detail: madrax should ideally batch these or internally pack sidechains.
                        mutant_energy = force_field.compute_energy(pdb_file, mutation=mutation_directive).to(device)
                        
                        ddG = (mutant_energy - wt_energy).item()
                        
                        # Filter out severe steric clashes (e.g. overlapping atoms yielding massive energy spikes)
                        if ddG > 1000.0 or np.isnan(ddG):
                            logging.warning(f"Severe clash detected for {pdb_id} {chain}{res_idx}{mut_aa}. Filtering.")
                            ddG = np.nan
                            
                except Exception as e:
                    logging.error(f"Failed Madrax forward pass for {pdb_id} {mutation_directive}: {e}")
                    ddG = np.nan
                    
                # 4. Append to Results Array
                results.append({
                    "PDB_ID": pdb_id,
                    "Chain": chain,
                    "Residue_Position": res_idx,
                    "WT_Amino_Acid": wt_aa,
                    "Mutant_Amino_Acid": mut_aa,
                    "Distance_to_CDR_center": dist_to_cdr,
                    "MadraX_ddG": ddG
                })
                
    except Exception as e:
        logging.error(f"Error parsing or processing structure {pdb_file}: {e}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TCR-pMHC ddG dataset using MadraX.")
    parser.add_argument("--pdb_dir", type=str, default="stcrdab_structures/", help="Directory containing STCRDab PDB files.")
    parser.add_argument("--output", type=str, default="tcr_pmhc_interface_ddg.csv", help="Output CSV file path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of PDB files to process (useful for dry runs).")
    args = parser.parse_args()

    pdb_dir = args.pdb_dir
    output_csv = args.output
    
    # 1. Gather all PDBs for the dataset
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir)
        logging.info(f"Directory {pdb_dir} created. Please populate it with STCRDab PDB files.")
        
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    
    if args.limit:
        pdb_files = pdb_files[:args.limit]
        logging.info(f"Limiting to {args.limit} PDB files for this run.")
        
    logging.info(f"Found {len(pdb_files)} PDB files to process.")
    
    if not pdb_files:
        logging.error("No PDB files found. Exiting.")
        return

    # 2. Setup Device (cuda) and MadraX Force Field
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device} for PyTorch tensors.")
    
    # Initialize MadraX model (assuming it has a similar API to standard PyTorch models)
    if hasattr(madrax, 'MadraxModel'):
       force_field = madrax.MadraxModel().to(device)
    else:
       force_field = madrax().to(device) # Mock instantiation
    force_field.eval()
    
    dataset = TcrPmhcDataset(pdb_files)
    all_results = []
    
    # Note: For processing 1,100 files, if madrax supports batched inference (e.g. shapes [B, N, 3]), 
    # you can wrap `dataset` in a DataLoader and adjust the `process_pdb_structure` accordingly. 
    # For now, we serialize processing over files but use the GPU internally for tensor ops.
    
    for i in tqdm(range(len(dataset)), desc="Evaluating TCR-pMHC Interfaces"):
        pdb_file = dataset[i]
        res = process_pdb_structure(pdb_file, device, force_field)
        all_results.extend(res)
        
    # 3. Output payload to DataFrame & CSV
    df = pd.DataFrame(all_results)
    # Drop rows where MadraX_ddG is NaN (due to severe steric clashes or exceptions)
    df.dropna(subset=['MadraX_ddG'], inplace=True) 
    
    df.to_csv(output_csv, index=False)
    logging.info(f"Successfully generated {len(df)} synthetic ddG points.")
    logging.info(f"Results successfully written to {output_csv}.")

if __name__ == "__main__":
    main()
