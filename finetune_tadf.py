"""
Fine-tunes the model on your TADF dataset BEFORE generating.

Why this is needed: when loading the GuacaMol checkpoint you had to resize
mlp_in_X/mlp_in_y (new atom vocabulary, with Se/Si/I). Those two layers were left
with random, UNTRAINED weights. That is the first layer every piece of information
goes through before reaching the rest of the transformer (which is pretrained) -
without training it, everything downstream receives noise, which is why you got
0/10 valid molecules regardless of the guidance.

This script:
1. Rebuilds the model exactly as in your main() (loads GuacaMol, resizes
   mlp_in_X/mlp_in_y).
2. Adds the missing 'training_step' (your class already has self.train_loss and
   self.apply_noise, but they are never used anywhere because there is no
   training_step wiring them together).
3. Runs a few fine-tuning epochs with pytorch_lightning.Trainer over the 452
   molecules in tadf_dataset_fixed.pt.
4. Saves the fine-tuned checkpoint so main() can load it instead of the plain
   GuacaMol one next time.

Usage:
    python finetune_tadf.py --dataset data/tadf_dataset_fixed.pt --epochs 50 --out finetuned_tadf.ckpt
"""
import argparse
import os
import types
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.loader import DataLoader

# FIX: library code (src/metrics/train_metrics.py) calls wandb.log(...) without checking
# whether there is an active session, which blows up with
# "You must call wandb.init() before wandb.log()". We initialize wandb in 'disabled' mode
# so that ANY wandb.log() anywhere in the code (not just here) becomes a silent no-op,
# without having to patch every file of the library.
import wandb
wandb.init(mode="disabled")


def dense_batch_from_pyg(data, num_atom_types, num_bond_types, max_n_nodes, device):
    """Converts a sparse PyG Geometric batch into dense, one-hot (X, E, y, node_mask),
    in the format the diffusion model expects."""
    data = data.to(device)
    dense_X_idx, node_mask = to_dense_batch(data.x, data.batch, max_num_nodes=max_n_nodes)
    X = F.one_hot(dense_X_idx, num_classes=num_atom_types).float()
    X = X * node_mask.unsqueeze(-1)

    dense_E_idx = to_dense_adj(data.edge_index, data.batch, edge_attr=data.edge_attr,
                                max_num_nodes=max_n_nodes)
    E = F.one_hot(dense_E_idx.long(), num_classes=num_bond_types).float()
    E = E * node_mask.unsqueeze(1).unsqueeze(-1) * node_mask.unsqueeze(2).unsqueeze(-1)

    y = torch.zeros(X.size(0), 0, device=device)  # dataset_infos.output_dims['y'] == 0
    return X, E, y, node_mask


def make_training_step(num_atom_types, num_bond_types, max_n_nodes):
    """Creates the training_step that the class is missing, and returns it so it can
    be bound to the model instance."""

    def training_step(self, data, batch_idx):
        X, E, y, node_mask = dense_batch_from_pyg(
            data, num_atom_types, num_bond_types, max_n_nodes, self.device
        )
        noisy_data = self.apply_noise(X, E, y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(
            masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
            true_X=X, true_E=E, true_y=y,
            log=(batch_idx % 10 == 0),
        )
        self.log('train_loss', loss, prog_bar=True, batch_size=X.size(0))
        return loss

    return training_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/tadf_dataset_fixed.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2,
                     help="Parallel processes used to load data. 0 = everything in the main "
                          "thread (what you had). On Windows, 2-4 is usually best.")
    ap.add_argument("--max_atoms", type=int, default=None,
                     help="Discards molecules with more atoms than this limit. Very effective: "
                          "the whole batch is padded up to the largest molecule, and the "
                          "dominant cost (eigh in extra_features) is O(n^3). Your dataset has a "
                          "median of 56 but a maximum of 272 atoms, so --max_atoms 100 keeps "
                          "90%% of the molecules and cuts the cost by roughly 20x.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="finetuned_tadf.ckpt")
    ap.add_argument("--resume", default=None,
                     help="Path to a previous checkpoint produced by this same script "
                          "(e.g. finetune_autosave_last.ckpt) to CONTINUE the fine-tuning "
                          "instead of starting over from GuacaMol.")
    args = ap.parse_args()

    # Imports the function we already factored out in your own file: it builds the model
    # exactly like your main() (dataset_infos, cfg, GuacaMol checkpoint, resizing of
    # mlp_in_X/mlp_in_y), without duplicating that logic here.
    import importlib.util
    v6_path = os.environ.get("V5_SCRIPT_PATH", "guidance_diffusion_model_discrete_v6.py")
    spec = importlib.util.spec_from_file_location("guidance_v5", v6_path)
    guidance_v6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guidance_v6)  # runs the module WITHOUT running its __main__ (normal Python guard)

    model, dataset_infos, device = guidance_v6.build_model()

    # NEW: if --resume was passed, we load the weights from a previous run of THIS script
    # instead of starting from GuacaMol again. This avoids losing hours of compute if you
    # need to stop and pick the fine-tuning back up across several sessions.
    if args.resume:
        print(f"Resuming fine-tuning from '{args.resume}'...")
        resume_ckpt = torch.load(args.resume, map_location=device)
        resume_state = resume_ckpt.get("state_dict", resume_ckpt)
        model.load_state_dict(resume_state)
        print("Previous weights loaded, continuing training.")

    # NEW: optional size filter. IMPORTANT: besides discarding the large molecules, the
    # effective max_n_nodes has to be recomputed - if we filtered but kept padding up to
    # 272, we would save absolutely nothing.
    train_dataset = dataset_infos.dataset
    effective_max_n = dataset_infos.max_n_nodes
    if args.max_atoms is not None:
        original_n = len(train_dataset)
        train_dataset = [d for d in train_dataset if d.x.size(0) <= args.max_atoms]
        if len(train_dataset) == 0:
            raise ValueError(f"--max_atoms {args.max_atoms} discarded ALL the molecules.")
        effective_max_n = max(d.x.size(0) for d in train_dataset)
        print(f"Filter --max_atoms {args.max_atoms}: {len(train_dataset)}/{original_n} molecules "
              f"kept ({100*len(train_dataset)/original_n:.0f}%). "
              f"Effective max_n_nodes: {dataset_infos.max_n_nodes} -> {effective_max_n}")

    # --- from here on nothing needs to be touched ---
    model.train()
    model.training_step = types.MethodType(
        make_training_step(dataset_infos.num_atom_types, dataset_infos.num_bond_types,
                           effective_max_n),
        model,
    )
    # simple, direct optimizer (we do not rely on configure_optimizers, which uses
    # self.args.train.lr - an attribute with its own history of bugs in this file)
    model.configure_optimizers = types.MethodType(
        lambda self: torch.optim.AdamW(self.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-12),
        model,
    )

    # NEW: num_workers>0 loads the batches in parallel processes, overlapping data
    # preparation with compute (Lightning warns about this explicitly). On Windows each
    # worker spawns a new process, so very high values can be counterproductive; 2-4 is
    # usually the sweet spot. persistent_workers avoids recreating them every epoch.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    # NEW: autosave at the end of EVERY epoch. If the process crashes, the power goes out,
    # or you close the console by accident, you never lose more than one epoch of progress.
    # It is saved in the same {"state_dict": ...} format used by --resume and GUACAMOL_CKPT_PATH.
    class AutosaveEveryEpoch(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            path = args.out.replace(".ckpt", "_autosave.ckpt")
            torch.save({"state_dict": pl_module.state_dict()}, path)
            print(f"Autosaved epoch {trainer.current_epoch} to '{path}'")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
        callbacks=[AutosaveEveryEpoch()],
    )

    # NEW: if you interrupt with Ctrl+C, the current state of the model still gets saved
    # (on top of the per-epoch autosave, in case you want the EXACT state at the moment you
    # stopped, not just the one from the last completed epoch).
    try:
        trainer.fit(model, train_dataloaders=train_loader)
    except KeyboardInterrupt:
        print("\nTraining interrupted manually. Saving the current state before exiting...")

    torch.save({"state_dict": model.state_dict()}, args.out)
    print(f"Fine-tuned model saved to {args.out}")
    print(f"Set GUACAMOL_CKPT_PATH to point at this file the next time you generate.")
    print(f"To CONTINUE this training later: python finetune_tadf.py --resume {args.out} ...")


if __name__ == "__main__":
    main()
