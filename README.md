# Guided TADF molecule generation with DiGress

Conditional generation of TADF molecules with a discrete diffusion model (DiGress),
guided by an MLP that predicts the singlet–triplet gap (ΔE_ST) from Morgan fingerprints.

The pipeline has four moving parts:

| Component | File | What it is |
|---|---|---|
| Property predictor | `tadf_mlp_model.pt` | MLP that maps a 2049-dim fingerprint (2048 Morgan bits + solvent feature) to ΔE_ST. Trained separately in `daugfinger_randomIII.py`. |
| Graph → fingerprint bridge | `graph_to_fp.pt` | Differentiable model that turns a DiGress graph into an approximate Morgan fingerprint, so gradients can flow from the MLP back to the graph. |
| Fragment importance | `fragment_importance.pt` | Signed per-bit importance (Integrated Gradients). Optional but recommended. |
| Diffusion backbone | `checkpoint/guacamol_last.ckpt` or `finetuned_tadf.ckpt` | Pretrained DiGress, optionally fine-tuned on the TADF dataset. |

---

## 1. Configuration

All the hardcoded absolute paths were replaced by environment variables. Nothing needs
to be edited inside the scripts anymore — set the variables that differ from the defaults.

| Variable | Default | Used by |
|---|---|---|
| `TADF_DATASET_PATH` | `data/tadf_dataset_fixed.pt` | `setup_tadf_components()` |
| `PROJECT_ROOT` | auto-detected (folder containing `src/`) | sys.path setup |
| `TRYING_ONE_PARENT` | auto-detected (folder containing `trying_one/`) | sys.path setup |
| `TADF_MLP_PATH` | `tadf_mlp_model.pt` | guidance MLP loading |
| `GUACAMOL_CKPT_PATH` | `checkpoint/guacamol_last.ckpt` | diffusion backbone weights |
| `GRAPH_TO_FP_CKPT` | `graph_to_fp.pt` | `cond_fn` |
| `FRAGMENT_IMPORTANCE_CKPT` | `fragment_importance.pt` | `cond_fn` (optional) |
| `LAMBDA_FRAG` | `1.0` | weight of the fragment term in the guidance loss |

**Linux / macOS**

```bash
export TADF_DATASET_PATH="data/tadf_dataset_fixed.pt"
export TADF_MLP_PATH="tadf_mlp_model.pt"
export GUACAMOL_CKPT_PATH="checkpoint/guacamol_last.ckpt"
```

**Windows PowerShell**

```powershell
$env:TADF_DATASET_PATH = "data\tadf_dataset_fixed.pt"
$env:TADF_MLP_PATH     = "tadf_mlp_model.pt"
$env:GUACAMOL_CKPT_PATH = "checkpoint\guacamol_last.ckpt"
```

These only last for the current shell session. To make them permanent on Windows use
`setx TADF_MLP_PATH "C:\ruta\a\tadf_mlp_model.pt"` and open a new terminal.

If a required file is missing, the script fails immediately with a message telling you
exactly which variable to set — it no longer fails halfway through generation.

---

## 2. Pipeline

Run from the folder containing the scripts (`diffguidance/`). See `MIGRATION.md` if you
are moving them there from `src/guidance/`.

### Step 0 - Prepare the dataset

Two one-off preprocessing steps, only needed if you do not already have the `.pt` files:

```bash
# split the ambiguous atom index 10 into Se / Si / I
python fix_dataset_atom_types.py --dataset data/tadf_dataset.pt --out data/tadf_dataset_fixed.pt

# reindex to GuacaMol's exact atom vocabulary (recommended: lets the checkpoint's
# mlp_in_X / mlp_out_X weights be reused instead of randomly reinitialized)
python fix_dataset_to_guacamol_vocab.py --dataset data/tadf_dataset_fixed.pt --out data/tadf_dataset_guacamol.pt
```

If you run the second one, remember to point `TADF_DATASET_PATH` at
`data/tadf_dataset_guacamol.pt` and to update `atom_decoder` / `valencies` in
`tadf_dataset.py` as the script instructs.

### Step 1 — Train the graph → fingerprint bridge

Required. Without it, generation aborts with a `FileNotFoundError`.

```bash
python train_graph_to_fp.py --dataset data/tadf_dataset_fixed.pt --out graph_to_fp.pt
```

### Step 2 — Compute fragment importance

Optional. If skipped, guidance falls back to the property MSE alone and prints a warning.

```bash
python analyze_fragment_importance.py \
    --dataset data/tadf_dataset_fixed.pt \
    --guidance_mlp "$TADF_MLP_PATH" \
    --out fragment_importance.pt
```

On Windows PowerShell, `"$env:TADF_MLP_PATH"`.

### Step 3 — Fine-tune the backbone on the TADF dataset

Optional but strongly recommended: the GuacaMol checkpoint has not seen TADF chemistry.

```bash
# from scratch (from the GuacaMol weights)
python finetune_tadf.py --dataset data/tadf_dataset_fixed.pt --epochs 5 --out finetuned_tadf.ckpt

# resume from where it stopped
python finetune_tadf.py --dataset data/tadf_dataset_fixed.pt --resume finetuned_tadf.ckpt --epochs 20 --out finetuned_tadf.ckpt
```

To generate with the fine-tuned weights instead of the GuacaMol ones, point the checkpoint
variable at them:

```bash
export GUACAMOL_CKPT_PATH="finetuned_tadf.ckpt"
```

### Step 4 — Guided generation

```bash
python guidance_diffusion_model_discrete_v6.py
```

Note the version bump: the entry point is now **v6**, not `guidance_diffusion_model_discrete_v5.py`.

---

## 3. Generation parameters

Edited at the top of the `if __name__ == "__main__":` block:

```python
num_samples        = 100    # molecules to generate
target_E_ST_value  = 0.1    # target ΔE_ST in eV
guidance_scale     = 1.5    # gamma: guidance strength
```

`guidance_scale` is the main knob. Higher values push harder toward the target but degrade
chemical validity; 1.5 / 2.5 / 5.0 are reasonable points to sweep. `LAMBDA_FRAG` weights the
fragment term against the property MSE; set it to `0` to disable the fragment term without
deleting the file.

## 4. Outputs

| File | Content |
|---|---|
| `generated_tensorsIII.pt` | Raw generated graphs (atom and bond tensors) |
| `generated_smiles_results.csv` | Per-molecule index, SMILES, validity flag and status |
| `chains/<exp_name>/` | GIF animations of the denoising trajectory |
| `graphs/<model>/epoch<N>_b<batch>/` | 2D renders of the final molecules |
| `psi4_output.dat` | Psi4 log, only when running the DFT evaluation path |

The console prints the overall chemical validity at the end, e.g.
`Overall chemical validity: 37 / 100 (37.0%)`.

## 5. Notes and known issues

- All comments and console messages in `guidance_diffusion_model_discrete_v6.py` are in
  English and ASCII-only.
- Psi4 is optional. If it is not installed the module prints `PSI4 not found` on import and
  keeps working; only `cond_sample_metric` (the DFT-based MAE evaluation) needs it.
- Low validity on a first run usually means the backbone has not been fine-tuned on TADF
  chemistry, or `guidance_scale` is too high.
- `guidance_diffusion_model_discrete_v6.py:388` still contains an invalid escape sequence
  (`f"\ Warning ..."`), which emits a `SyntaxWarning` on import. It was probably meant to be
  `\n` and is left as-is pending confirmation.
