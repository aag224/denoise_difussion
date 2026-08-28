"""
GraphToFP: differentiable, permutation-invariant network that maps the raw graph
tensors (X: nodes, E: edges, both "soft"/one-hot) into the same 2049-dimensional
space expected by TADF_MLP (2048 Morgan fingerprint bits + 1 solvent feature).

Why this and not a plain nn.Linear over the flattened graph:
- The dataset contains molecules with up to 272 atoms. Flattening E (N x N x 5)
  gives 272*272*5 = 369,920 values per graph. A dense Linear from there to 2049
  would have ~760M parameters in a single layer: infeasible and physically
  meaningless.
- An encoder of the form "one graph convolution layer + global pooling" scales
  reasonably (linearly/quadratically), is invariant to node permutation
  (important: the atom ordering in X, E has no canonical meaning) and is fully
  differentiable, so the gradient that cond_fn() computes w.r.t. X_t, E_t
  remains valid.
"""
import torch
import torch.nn as nn


class GraphToFP(nn.Module):
    def __init__(self, num_atom_types, num_bond_types, hidden_dim=128, fp_dim=2049):
        super().__init__()
        self.node_embed = nn.Linear(num_atom_types, hidden_dim)
        self.edge_embed = nn.Linear(num_bond_types, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fp_dim),
        )

    def forward(self, X, E, node_mask=None):
        """
        X: (bs, n, num_atom_types)  -- soft or one-hot
        E: (bs, n, n, num_bond_types)
        node_mask: (bs, n) bool, optional
        returns: (bs, fp_dim) in [0,1] (sigmoid) -> compatible with fingerprint + solvent feature
        """
        bs, n, _ = X.shape
        if node_mask is None:
            node_mask = torch.ones(bs, n, dtype=torch.bool, device=X.device)
        mask_f = node_mask.float().unsqueeze(-1)  # bs, n, 1

        h = self.node_embed(X) * mask_f  # bs, n, hidden

        # Message from neighbors weighted by bond type: for each node i we sum
        # over j the projection of the bond type E[i,j], which acts as a
        # single-layer graph convolution.
        edge_h = self.edge_embed(E)  # bs, n, n, hidden
        edge_mask = (mask_f.unsqueeze(2) * mask_f.unsqueeze(1))  # bs, n, n, 1
        msg = (edge_h * edge_mask).sum(dim=2)  # bs, n, hidden  (sum over neighbors j)

        h = self.update(h + msg) * mask_f  # bs, n, hidden

        # Permutation-invariant global pooling
        pooled = h.sum(dim=1)  # bs, hidden

        out = torch.sigmoid(self.readout(pooled))  # bs, fp_dim, in [0,1]
        return out
