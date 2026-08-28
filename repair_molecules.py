"""
Repairs the generated graphs by removing bonds until every valence is satisfied,
and produces valid SMILES.

WHAT THIS IS (and what it is NOT):
- It is an emergency post-processing step, so there are valid molecules to work
  with WHILE the model finishes training.
- It does NOT replace training. Removing bonds changes the structure with respect
  to what the model generated, so the guidance toward delta E_ST and toward the
  favorable fragments gets diluted. The repaired molecule is NOT the guided
  molecule.
- The better trained the model is, the fewer bonds have to be removed and the more
  faithful the repair. In the current state (~43% of the atoms violating valence,
  median of 3 excess bonds) the repair is aggressive.

STRATEGY:
For every atom that exceeds its allowed valence, remove bonds starting with the
highest order ones (triple > double > aromatic > single) and with the neighbors
that are also saturated - that way each removal fixes two problems at once when
possible. Repeat until no violations remain. Finally, keep the largest connected
fragment (the same thing valid_mol_can_with_seg does in rdkit_functions.py).

Usage:
    python repair_molecules.py --tensors generated_tensorsII.pt --out repaired_smiles.csv
"""
import argparse
import csv
import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

ATOM_DECODER = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']
# Maximum valence allowed per element (the highest one when several are possible)
MAX_VALENCE = {'C': 4, 'N': 3, 'O': 2, 'F': 1, 'S': 6, 'P': 5, 'B': 3,
               'Br': 1, 'Cl': 1, 'H': 1, 'Se': 6, 'Si': 4, 'I': 1}
BOND_ORDER = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
BOND_TYPES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
              3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}


def current_valences(atom_syms, E):
    """Sum of bond orders per atom (E is the dense, symmetric matrix of bond types)."""
    n = len(atom_syms)
    val = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                val[i] += BOND_ORDER.get(int(E[i, j]), 0.0)
    return val


def repair_graph(atom_syms, E, verbose=False):
    """Removes bonds until no atom exceeds its valence. Returns the repaired E
    and how many bonds were removed."""
    E = E.clone()
    n = len(atom_syms)
    removed = 0

    for _ in range(n * 20):  # safety bound
        val = current_valences(atom_syms, E)
        # atoms that violate their valence, worst one first
        violators = [(val[i] - MAX_VALENCE.get(atom_syms[i], 4), i)
                     for i in range(n)
                     if val[i] > MAX_VALENCE.get(atom_syms[i], 4) + 1e-6]
        if not violators:
            break
        violators.sort(reverse=True)
        _, i = violators[0]

        # candidates: neighbors of i, prioritizing (a) higher bond order,
        # (b) a neighbor that is also saturated (kill two birds with one stone)
        cands = []
        for j in range(n):
            if j == i:
                continue
            b = int(E[i, j])
            if b == 0:
                continue
            order = BOND_ORDER.get(b, 0.0)
            j_excess = val[j] - MAX_VALENCE.get(atom_syms[j], 4)
            cands.append((order, j_excess, j))
        if not cands:
            break
        cands.sort(reverse=True)
        _, _, j = cands[0]
        E[i, j] = 0
        E[j, i] = 0
        removed += 1

    return E, removed


def graph_to_mol(atom_syms, E):
    mol = Chem.RWMol()
    for s in atom_syms:
        mol.AddAtom(Chem.Atom(s))
    n = len(atom_syms)
    for i in range(n):
        for j in range(i + 1, n):
            b = int(E[i, j])
            if b in BOND_TYPES:
                mol.AddBond(i, j, BOND_TYPES[b])
    return mol.GetMol()


def largest_fragment(mol):
    try:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if not frags:
            return mol
        return max(frags, key=lambda m: m.GetNumAtoms())
    except Exception:
        return mol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", default="generated_tensorsII.pt")
    ap.add_argument("--out", default="repaired_smiles.csv")
    args = ap.parse_args()

    data = torch.load(args.tensors, map_location='cpu', weights_only=False)
    # the file stores a list of (atom_types, edge_types) per molecule
    molecules = data if isinstance(data, list) else data.get('molecules', data)

    results = []
    n_valid = 0
    for idx, item in enumerate(molecules, start=1):
        atom_types, edge_types = item[0], item[1]
        atom_types = atom_types.cpu()
        edge_types = edge_types.cpu()

        # drop padding (-1) if there is any
        keep = atom_types >= 0
        atom_types = atom_types[keep]
        edge_types = edge_types[keep][:, keep]
        edge_types = edge_types.clamp(min=0)

        atom_syms = [ATOM_DECODER[int(a)] if int(a) < len(ATOM_DECODER) else 'C'
                     for a in atom_types]

        n_bonds_before = int((edge_types > 0).sum().item() // 2)
        E_fixed, removed = repair_graph(atom_syms, edge_types)
        n_bonds_after = int((E_fixed > 0).sum().item() // 2)

        mol = graph_to_mol(atom_syms, E_fixed)
        mol = largest_fragment(mol)

        smiles = None
        try:
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            # second attempt: aromatic -> single (usually fixes kekulization)
            try:
                rw = Chem.RWMol(mol)
                for b in rw.GetBonds():
                    if b.GetBondType() == Chem.BondType.AROMATIC:
                        b.GetBeginAtom().SetIsAromatic(False)
                        b.GetEndAtom().SetIsAromatic(False)
                        b.SetBondType(Chem.BondType.SINGLE)
                m2 = rw.GetMol()
                Chem.SanitizeMol(m2)
                smiles = Chem.MolToSmiles(m2)
            except Exception:
                smiles = None

        ok = smiles is not None
        n_valid += int(ok)
        results.append({
            'index': idx,
            'atoms': len(atom_syms),
            'bonds_before': n_bonds_before,
            'bonds_after': n_bonds_after,
            'bonds_removed': removed,
            'pct_bonds_removed': f"{100*removed/max(n_bonds_before,1):.1f}%",
            'is_valid': ok,
            'smiles': smiles or '',
        })
        status = f"OK {smiles[:60]}" if ok else "not repairable"
        print(f"Mol {idx:3d}: {len(atom_syms):3d} atoms | bonds {n_bonds_before:4d}->{n_bonds_after:4d} "
              f"(-{removed}, {100*removed/max(n_bonds_before,1):.0f}%) | {status}")

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print()
    print(f"Valid after repair: {n_valid}/{len(results)} ({100*n_valid/len(results):.1f}%)")
    avg_removed = sum(r['bonds_removed'] for r in results) / len(results)
    avg_pct = sum(100*r['bonds_removed']/max(r['bonds_before'],1) for r in results) / len(results)
    print(f"   Bonds removed on average: {avg_removed:.0f} per molecule ({avg_pct:.0f}% of the original)")
    print(f"   Warning: the higher that %, the less faithful the repaired molecule is to what the model generated.")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
