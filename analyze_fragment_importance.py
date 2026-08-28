"""
Extracts, from your already trained guidance_mlp (TADF_MLP), which bits of the
Morgan fingerprint (= which fragments) influence delta E_ST the most, and WITH
WHICH SIGN.

It uses exactly the same method you already use in daugfinger_randomIII.py:
Captum Integrated Gradients against a zero baseline, target=0 (the single
regression output). This is intentional: that way 'fragment_importance.pt' is
consistent with the same numbers you already see in your attribution plots.

IMPORTANT (unlike an earlier version of this script): the attribution is SIGNED,
not absolute value.
- NEGATIVE attribution -> the fragment LOWERS the predicted delta E_ST (favors TADF).
- POSITIVE attribution -> the fragment RAISES the predicted delta E_ST (disfavors TADF).
Using |attribution| would treat a good fragment and a bad one the same way, which
is exactly the opposite of what we want to guide toward.

It does not need real delta E_ST labels (the .pt does not carry them): Integrated
Gradients only compares the model output against a zero baseline, just like in
your original pipeline.

Usage:
    python analyze_fragment_importance.py --dataset data/tadf_dataset.pt \
        --guidance_mlp tadf_mlp_model.pt --out fragment_importance.pt --top_k 15
"""
import argparse
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from captum.attr import IntegratedGradients
import sys
import os


def real_fingerprint_with_bitinfo(smiles, n_bits=2048, radius=2, solvent_eps=2.38):
    """Same format as prepare_features() in daugfinger_randomIII.py:
    2048 Morgan bits + 1 solvent feature (eps/12.0). tadf_dataset.pt does not
    carry a per-molecule solvent, so we use the same default as your own code
    (toluene, eps=2.38) when the value is missing."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    bit_info = {}
    fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
    fp = np.array(fp_vect, dtype=np.float32)
    eps_scaled = np.array([solvent_eps / 12.0], dtype=np.float32)
    return torch.tensor(np.concatenate([fp, eps_scaled]), dtype=torch.float32), bit_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/tadf_dataset.pt")
    ap.add_argument("--guidance_mlp", default="tadf_mlp_model.pt")
    ap.add_argument("--out", default="fragment_importance.pt")
    ap.add_argument("--top_k", type=int, default=15)
    args = ap.parse_args()

    # FIX: this used to count a fixed number of '..' levels, which breaks if the file is
    # moved to another depth. We now search upward for the folder that CONTAINS
    # 'trying_one', so the project can be relocated without editing this.
    def _find_ancestor_containing(dirname, start):
        current = start
        while True:
            if os.path.isdir(os.path.join(current, dirname)):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent

    here = os.path.abspath(os.path.dirname(__file__))
    trying_one_path = os.environ.get("TRYING_ONE_PARENT") or _find_ancestor_containing("trying_one", here)
    if trying_one_path is None:
        raise RuntimeError(
            "Could not locate the folder containing 'trying_one/'. "
            "Set the TRYING_ONE_PARENT environment variable."
        )
    if trying_one_path not in sys.path:
        sys.path.insert(0, trying_one_path)
    from trying_one.daugfinger_randomIII import TADF_MLP  # your original class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guidance_mlp = TADF_MLP(input_dim=2049, hidden_dim1=256, hidden_dim2=64).to(device)
    guidance_mlp.load_state_dict(torch.load(args.guidance_mlp, map_location=device))
    guidance_mlp.eval()

    dataset = torch.load(args.dataset, weights_only=False)

    fps = []
    bit_info_examples = {}  # bit_idx -> (smiles, atom_idx, radius) of the first example seen
    for d in dataset:
        if not hasattr(d, 'smiles'):
            continue
        fp, bit_info = real_fingerprint_with_bitinfo(d.smiles)
        if fp is None:
            continue
        fps.append(fp)
        for bit_idx, envs in bit_info.items():
            if bit_idx not in bit_info_examples:
                atom_idx, radius = envs[0]
                bit_info_examples[bit_idx] = (d.smiles, atom_idx, radius)

    X = torch.stack(fps).to(device)
    print(f"{X.size(0)} molecules with a valid SMILES used for Integrated Gradients")

    baseline = torch.zeros(1, 2049).to(device)
    ig = IntegratedGradients(guidance_mlp)
    # Same as in daugfinger_randomIII.py: target=0 (the single regression output)
    attributions, delta = ig.attribute(X, baseline, target=0, return_convergence_delta=True)

    mean_attr = attributions.mean(dim=0).detach().cpu().numpy()  # SIGNED, no abs()
    fp_scores = mean_attr[:2048]
    solvent_score = float(mean_attr[2048])
    print(f"Average solvent attribution (eps): {solvent_score:.6f} eV")

    # We normalize by the maximum |attribution| so it lands in [-1, 1] and
    # lambda_frag in cond_fn has a manageable scale; the sign is preserved.
    max_abs = np.abs(fp_scores).max()
    fp_scores_norm = fp_scores / max_abs if max_abs > 0 else fp_scores

    torch.save({
        "fragment_importance": torch.tensor(fp_scores_norm, dtype=torch.float32),  # SIGNED
        "solvent_attribution": solvent_score,
        "n_molecules_used": X.size(0),
    }, args.out)
    print(f"Saved to {args.out}")

    order = np.argsort(fp_scores)  # ascending: most negative (favor TADF) first
    top_favor = order[:args.top_k]
    top_disfavor = order[-args.top_k:][::-1]

    print(f"\nTop {args.top_k} fragments that MOST FAVOR TADF (they lower delta E_ST):")
    for b in top_favor:
        b = int(b)
        info = bit_info_examples.get(b)
        extra = f" | example: atom {info[1]} (radius {info[2]}) in {info[0][:40]}" if info else " | (not seen in dataset)"
        print(f"  bit {b:4d}  attribution={fp_scores[b]:+.6f}{extra}")

    print(f"\nTop {args.top_k} fragments that MOST DISFAVOR TADF (they raise delta E_ST):")
    for b in top_disfavor:
        b = int(b)
        info = bit_info_examples.get(b)
        extra = f" | example: atom {info[1]} (radius {info[2]}) in {info[0][:40]}" if info else " | (not seen in dataset)"
        print(f"  bit {b:4d}  attribution={fp_scores[b]:+.6f}{extra}")


if __name__ == "__main__":
    main()
