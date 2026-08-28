"""
Reindexes tadf_dataset_fixed.pt to the EXACT atom vocabulary of GuacaMol.

WHY:
Your GuacaMol checkpoint already knows how to generate fused aromatic rings (it was
trained on ~1.1M molecules). But since your atom vocabulary had a different order and
an extra 'H', the mlp_in_X and mlp_out_X layers were discarded on load and randomly
reinitialized. That forced the model to re-learn from scratch how to build molecules -
and with only 452 molecules that is not enough (hence the aliphatic chains with no
rings).

KEY FINDING:
Your dataset has NO explicit hydrogens (0 out of 29183 atoms). Removing the 'H' from
the atom_decoder leaves exactly the same 12 elements GuacaMol uses:
    B, Br, C, Cl, F, I, N, O, P, S, Se, Si
Only the ORDER differs. After reindexing, the dimensions become mlp_out_X=12 and
mlp_in_X=20, which is precisely what the checkpoint expects.

EXPECTED RESULT:
The model inherits all of GuacaMol's structural knowledge (rings, aromaticity,
valences) and the fine-tuning with your 452 molecules only has to specialize it toward
TADF chemistry, instead of learning chemistry from scratch.

Usage:
    python fix_dataset_to_guacamol_vocab.py \
        --dataset data/tadf_dataset_fixed.pt \
        --out data/tadf_dataset_guacamol.pt \
        --guacamol_dataset_py ../datasets/guacamol_dataset.py
"""
import argparse
import os
import re
import torch
from rdkit import Chem

# Current order of your dataset (13 types, with an unused H)
TADF_DECODER = ['C', 'N', 'O', 'F', 'S', 'P', 'B', 'Br', 'Cl', 'H', 'Se', 'Si', 'I']

# GuacaMol's order in DiGress. We try to read it from the real file in the repo; this is
# only the fallback if it cannot be found.
GUACAMOL_DECODER_FALLBACK = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']


def read_guacamol_decoder(path):
    """Extracts atom_decoder from the real guacamol_dataset.py in the repo (the
    authoritative source, because it is the one matching your checkpoint)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        src = f.read()
    m = re.search(r"atom_decoder\s*=\s*\[([^\]]+)\]", src)
    if not m:
        return None
    items = re.findall(r"['\"]([A-Za-z]{1,2})['\"]", m.group(1))
    return items or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/tadf_dataset_fixed.pt")
    ap.add_argument("--out", default="data/tadf_dataset_guacamol.pt")
    ap.add_argument("--guacamol_dataset_py", default="src/datasets/guacamol_dataset.py",
                     help="Path to guacamol_dataset.py in your DiGress repo, relative to the "
                          "folder you run this from. If it exists, the real atom_decoder is "
                          "read from there (recommended).")
    args = ap.parse_args()

    guaca = read_guacamol_decoder(args.guacamol_dataset_py)
    if guaca:
        print(f"atom_decoder read from '{args.guacamol_dataset_py}': {guaca}")
    else:
        guaca = GUACAMOL_DECODER_FALLBACK
        print(f"Could not read '{args.guacamol_dataset_py}'. Using the default list: {guaca}")
        print("   CHECK IT: if it does not match the one from your checkpoint, the atoms will come out wrong.")

    tadf_used = [a for a in TADF_DECODER if a != 'H']
    if set(tadf_used) != set(guaca):
        raise SystemExit(
            f"The element sets do NOT match.\n"
            f"   TADF (without H): {sorted(tadf_used)}\n"
            f"   GuacaMol        : {sorted(guaca)}\n"
            f"   Missing in GuacaMol: {sorted(set(tadf_used) - set(guaca))}\n"
            f"   Without an exact match this approach does not apply."
        )
    print(f"Element sets are identical ({len(guaca)} types). Only the order changes.")

    # Map: old index (TADF) -> new index (GuacaMol)
    guaca_idx = {s: i for i, s in enumerate(guaca)}
    remap = {}
    for old_i, sym in enumerate(TADF_DECODER):
        if sym == 'H':
            continue  # not used in the dataset
        remap[old_i] = guaca_idx[sym]

    print("\nReindexing map (old -> new):")
    for old_i in sorted(remap):
        print(f"  {old_i:2d} ({TADF_DECODER[old_i]:2s}) -> {remap[old_i]:2d} ({guaca[remap[old_i]]:2s})")

    dataset = torch.load(args.dataset, weights_only=False)
    n_h_found = 0
    for d in dataset:
        new_x = d.x.clone()
        for old_i, new_i in remap.items():
            new_x[d.x == old_i] = new_i
        n_h_found += int((d.x == TADF_DECODER.index('H')).sum().item())
        d.x = new_x

    if n_h_found:
        raise SystemExit(f"Found {n_h_found} explicit hydrogens. This approach assumes there "
                          f"are none (there were 0 when verified). Check the dataset.")
    print(f"\n{len(dataset)} molecules reindexed. Explicit hydrogens found: 0")

    torch.save(dataset, args.out)
    print(f"Saved to {args.out}")

    # Final verification against RDKit, atom by atom
    print("\nFinal verification against RDKit...")
    checked = mism = 0
    for d in dataset:
        mol = Chem.MolFromSmiles(d.smiles)
        if mol is None or mol.GetNumAtoms() != d.x.size(0):
            continue
        for i, atom in enumerate(mol.GetAtoms()):
            idx = int(d.x[i])
            checked += 1
            if not (0 <= idx < len(guaca)) or guaca[idx] != atom.GetSymbol():
                mism += 1
    print(f"  Atoms verified: {checked} | mismatches: {mism}")
    if mism == 0:
        print("  All indices match the real symbol according to RDKit.")
        print("\nNEXT STEP: update tadf_dataset.py to use this atom_decoder:")
        print(f"  self.atom_decoder = {guaca}")
        valencies = {'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Br': 1, 'Cl': 1,
                      'I': 1, 'P': 3, 'S': 2, 'Se': 2, 'Si': 4}
        print(f"  self.valencies    = {[valencies[s] for s in guaca]}")
    else:
        print("  There are mismatches: check that the GuacaMol atom_decoder is the right one.")


if __name__ == "__main__":
    main()
