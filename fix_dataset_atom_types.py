"""
Rebuilds tadf_dataset.pt by splitting index 10 (which lumped Se, Si and I together
without distinguishing them) into three new, correct indices: 10=Se, 11=Si, 12=I.

Previously verified: indices 0-9 DO match atom_decoder exactly across the 452
molecules / 29183 atoms (there is no need to touch them). Only the original index 10
was built incorrectly.

It uses RDKit's atom ordering (Chem.MolFromSmiles(d.smiles).GetAtoms()) to know which
real element corresponds to each position marked as 10 - that ordering matches the one
in Data.x for this dataset (confirmed: 452/452 molecules with the same number of atoms
on both sides).

Usage:
    python fix_dataset_atom_types.py --dataset data/tadf_dataset.pt --out data/tadf_dataset_fixed.pt
"""
import argparse
import torch
from rdkit import Chem

OLD_ATOM_DECODER = ['C', 'N', 'O', 'F', 'S', 'P', 'B', 'Br', 'Cl', 'H']
NEW_ELEMENTS = ['Se', 'Si', 'I']  # added as indices 10, 11, 12
NEW_ATOM_DECODER = OLD_ATOM_DECODER + NEW_ELEMENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/tadf_dataset.pt")
    ap.add_argument("--out", default="data/tadf_dataset_fixed.pt")
    args = ap.parse_args()

    dataset = torch.load(args.dataset, weights_only=False)
    symbol_to_new_idx = {sym: i for i, sym in enumerate(NEW_ATOM_DECODER)}

    n_fixed_atoms = 0
    n_fixed_molecules = 0
    n_unresolved = 0

    for d in dataset:
        mol = Chem.MolFromSmiles(d.smiles)
        if mol is None or mol.GetNumAtoms() != d.x.size(0):
            print(f"Could not re-verify (SMILES/RDKit size mismatch), left unchanged: {d.smiles[:60]}")
            continue

        touched = False
        for i, atom in enumerate(mol.GetAtoms()):
            if int(d.x[i]) == 10:  # the original ambiguous index
                real_symbol = atom.GetSymbol()
                if real_symbol in symbol_to_new_idx:
                    d.x[i] = symbol_to_new_idx[real_symbol]
                    n_fixed_atoms += 1
                    touched = True
                else:
                    n_unresolved += 1
                    print(f"Element '{real_symbol}' is not covered by NEW_ELEMENTS, check manually: {d.smiles[:60]}")
        if touched:
            n_fixed_molecules += 1

    print(f"Atoms reassigned: {n_fixed_atoms}  |  Molecules touched: {n_fixed_molecules}  |  unresolved: {n_unresolved}")

    torch.save(dataset, args.out)
    print(f"Saved to {args.out}")

    # Final verification: walk through the WHOLE dataset and confirm that every index
    # matches the real RDKit symbol, including the new 10/11/12.
    print("\nFinal verification...")
    mismatches = 0
    checked = 0
    for d in dataset:
        mol = Chem.MolFromSmiles(d.smiles)
        if mol is None or mol.GetNumAtoms() != d.x.size(0):
            continue
        for i, atom in enumerate(mol.GetAtoms()):
            idx = int(d.x[i])
            expected = NEW_ATOM_DECODER[idx] if idx < len(NEW_ATOM_DECODER) else None
            checked += 1
            if expected != atom.GetSymbol():
                mismatches += 1
    print(f"Atoms verified: {checked}  |  remaining mismatches: {mismatches}")
    if mismatches == 0:
        print("All atom indices now match RDKit, including Se/Si/I.")


if __name__ == "__main__":
    main()
