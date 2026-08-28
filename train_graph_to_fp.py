"""
Trains GraphToFP so it learns to map (X, E) -> real fingerprint (Morgan 2048 bits
+ 1 solvent feature), using the 452 real molecules in tadf_dataset.pt. Once
trained, cond_fn() in the guidance script can use this network instead of the
random linear projection, and the gradient computed w.r.t. X_t, E_t will point
toward structures that really do change the fingerprint / the prediction of your
guidance_mlp in a way that is consistent with the real chemistry.

Usage:
    python train_graph_to_fp.py --dataset data/tadf_dataset.pt --epochs 200 --out graph_to_fp.pt
"""
import argparse
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

from graph_to_fp import GraphToFP

ATOM_DECODER = ['C', 'N', 'O', 'F', 'S', 'P', 'B', 'Br', 'Cl', 'H']


def build_dense_graph(data, num_atom_types, num_bond_types, max_n):
    """Converts a Data object (x, edge_index, directed edge_attr) into dense,
    padded tensors, in the SAME format the diffusion model uses:
    X: (max_n, num_atom_types) one-hot, E: (max_n, max_n, num_bond_types) one-hot
    (with 0 = "no bond" for the padding and for unconnected pairs)."""
    n = data.x.size(0)
    x_idx = data.x.long().clamp(max=num_atom_types - 1)  # in case there are out-of-range indices
    X = torch.zeros(max_n, num_atom_types)
    X[:n] = F.one_hot(x_idx, num_classes=num_atom_types).float()

    E_idx = torch.zeros(max_n, max_n, dtype=torch.long)  # 0 = no bond
    if data.edge_index.numel() > 0:
        row, col = data.edge_index[0], data.edge_index[1]
        E_idx[row, col] = data.edge_attr.long()
    E = F.one_hot(E_idx, num_classes=num_bond_types).float()

    node_mask = torch.zeros(max_n, dtype=torch.bool)
    node_mask[:n] = True
    return X, E, node_mask


def real_fingerprint(smiles, n_bits=2048, radius=2):
    """Real fingerprint via RDKit, in EXACTLY the same format as
    extract_fingerprints() in the guidance script (2048 bits + 1 dummy solvent
    feature)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    fp = np.array(fp_vect, dtype=np.float32)
    solvent_dummy = np.array([2.38 / 12.0], dtype=np.float32)
    return torch.tensor(np.concatenate([fp, solvent_dummy]), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/tadf_dataset.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="graph_to_fp.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = torch.load(args.dataset, weights_only=False)

    num_atom_types = max(len(ATOM_DECODER), max(int(d.x.max().item()) for d in dataset) + 1)
    num_bond_types = 5
    max_n = max(d.x.size(0) for d in dataset)
    print(f"num_atom_types={num_atom_types}, num_bond_types={num_bond_types}, max_n={max_n}")

    Xs, Es, Masks, Fps = [], [], [], []
    skipped = 0
    for d in dataset:
        if not hasattr(d, 'smiles'):
            skipped += 1
            continue
        fp = real_fingerprint(d.smiles)
        if fp is None:
            skipped += 1
            continue
        X, E, mask = build_dense_graph(d, num_atom_types, num_bond_types, max_n)
        Xs.append(X); Es.append(E); Masks.append(mask); Fps.append(fp)

    print(f"Molecules used: {len(Xs)} (skipped: {skipped})")
    X = torch.stack(Xs).to(device)
    E = torch.stack(Es).to(device)
    mask = torch.stack(Masks).to(device)
    Y = torch.stack(Fps).to(device)

    model = GraphToFP(num_atom_types, num_bond_types, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_val = max(1, int(0.1 * X.size(0)))
    perm = torch.randperm(X.size(0))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    batch_size = args.batch_size

    for epoch in range(args.epochs):
        model.train()
        train_perm = train_idx[torch.randperm(train_idx.size(0))]
        epoch_loss = 0.0
        for i in range(0, train_perm.size(0), batch_size):
            b = train_perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X[b], E[b], mask[b])
            loss = F.binary_cross_entropy(pred, Y[b].clamp(0, 1))
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * b.size(0)
        epoch_loss /= train_perm.size(0)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for i in range(0, val_idx.size(0), batch_size):
                    b = val_idx[i:i + batch_size]
                    val_pred = model(X[b], E[b], mask[b])
                    val_losses.append(F.binary_cross_entropy(val_pred, Y[b].clamp(0, 1)).item() * b.size(0))
                val_loss = sum(val_losses) / val_idx.size(0)
            print(f"epoch {epoch:4d}  train_bce={epoch_loss:.4f}  val_bce={val_loss:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "num_atom_types": num_atom_types,
        "num_bond_types": num_bond_types,
        "hidden_dim": args.hidden_dim,
        "max_n_nodes": max_n,
    }, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
