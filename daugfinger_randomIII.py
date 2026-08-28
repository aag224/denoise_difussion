import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from captum.attr import IntegratedGradients
from PIL import Image
import io

# ==========================================
# 1. DATA LOADING AND CLEANING WITH RDKIT
# ==========================================
def load_and_clean_real_dataset(csv_path, smiles_col="name.smiles", target_col="raw_value"):
    """
    Loads the CSV, drops null values and checks that the SMILES strings
    can be parsed by RDKit.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[smiles_col, target_col]).copy()
    
    valid_rows = []
    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is not None:
            valid_rows.append(row)
            
    clean_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    return clean_df

# ==========================================
# 2. MORGAN FINGERPRINT EXTRACTION
# ==========================================
# Flexible mapping to tolerate name variants in Spanish and English
SOLVENT_EPSILON = {
    'C1=CC=CC=C1': 2.38, 'C1=CC=CC=C2': 2.38,'C1=CC=CC=C3': 2.38, 'C1=CC=CC=C4': 2.38,
    'CC1CCOC1': 6.97, 'CC1OCCC1': 6.97, 'CC1CCCO1': 6.97,
    'C1CCCO2': 7.58, 'tC1CCCO3"': 7.58, 'C1CCCO4': 7.58, 'C1CCCO5': 7.58, 'C1CCCO6': 7.58,
    'ClC([H])([H])Cl': 8.93,
    'CC(Cl)Cl': 10.36
}

def prepare_features(smiles_list, solvent_list, radius=2, n_bits=2048):
    """
    Combines the 2048 Morgan fingerprint bits with the normalized dielectric
    constant of the solvent (total dimension = 2049).
    """
    X_features = []
    
    for sm, sol in zip(smiles_list, solvent_list):
        # 1. Extract the fingerprint
        mol = Chem.MolFromSmiles(str(sm))
        if mol is not None:
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)
        else:
            fp = np.zeros(n_bits, dtype=np.float32)
            
        # 2. Extract and scale the dielectric constant (eps / 12.0 for a ~0 to 1 scale)
        sol_clean = str(sol).strip().lower()
        eps = SOLVENT_EPSILON.get(sol_clean, 2.38) # toluene by default if the value is missing
        eps_scaled = np.array([eps / 12.0], dtype=np.float32)
        
        # 3. Concatenate: 2049-dimensional vector
        combined_vector = np.concatenate([fp, eps_scaled])
        X_features.append(combined_vector)
        
    return np.array(X_features, dtype=np.float32)
# ==========================================
# 3. PYTORCH DATASET
# ==========================================
class FingerprintDataset(Dataset):
    """
    Dataset for numeric tabular vectors (fingerprints).
    """
    def __init__(self, fps_matrix, targets):
        self.X = torch.tensor(fps_matrix, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}

# ==========================================
# 4. DENSE NEURAL NETWORK ARCHITECTURE (MLP)
# ==========================================
class TADF_MLP(nn.Module):
    """
    Multilayer perceptron with batch normalization and dropout to
    regularize on small datasets.
    """
    def __init__(self, input_dim=2048, hidden_dim1=256, hidden_dim2=64, dropout_rate=0.20):
        super(TADF_MLP, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer -> hidden 1
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden 1 -> hidden 2
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer (regression)
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# ==========================================
# 5. PARITY PLOT AND EVALUATION
# ==========================================
def evaluate_and_plot_parity(y_true, y_pred, title="ΔE_ST-Parity Plot (MLP/Fingerprint)"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    plt.figure(figsize=(7, 6), dpi=120)
    
    # 1. MARKER SIZE: 's' raised from 50 to 95
    plt.scatter(y_true, y_pred, alpha=0.75, color='#2b5c8f', edgecolors='k', linewidth=0.5, s=95, label='Test samples')
    
    min_val = min(min(y_true), min(y_pred)) - 0.02
    max_val = max(max(y_true), max(y_pred)) + 0.02
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
    plt.axvspan(min_val, 0.2, color='green', alpha=0.12, label='Optimal TADF region (≤ 0.20 eV)')
    
    plt.xlabel('Actual ΔE_ST (eV)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted ΔE_ST (eV)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=12, pad=12)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # --- Metrics box ---
    metrics_box = (
        f"MAE  = {mae:.4f} eV\n"
        f"RMSE = {rmse:.4f} eV\n"
        f"R²   = {r2:.4f}"
    )
    
    # 2. METRICS BOX SIZE: 'fontsize' raised to 11.5 and 'pad' to 0.7
    plt.gca().text(
        0.95, 0.05, metrics_box, transform=plt.gca().transAxes,
        fontsize=14, horizontalalignment='right', verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 3. LEGEND SIZE: 'fontsize=11' added
    plt.legend(loc='upper left', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTest set evaluation summary (MLP + fingerprints):")
    print(f"   - Mean absolute error (MAE):    {mae:.4f} eV")
    print(f"   - Root mean squared error (RMSE): {rmse:.4f} eV")
    print(f"   - R2 coefficient:               {r2:.4f}")
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def evaluate_model_on_test(model, test_loader, device):
    model.eval()
    y_reales = []
    y_predichos = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_batch = batch["x"].to(device)
            y_batch = batch["y"].to(device)
            
            preds = model(x_batch)
            
            y_reales.extend(y_batch.cpu().numpy().flatten())
            y_predichos.extend(preds.cpu().numpy().flatten())
            
    return evaluate_and_plot_parity(y_reales, y_predichos)

# ==========================================
# 6. MAIN FUNCTION (PIPELINE)
# ==========================================
def find_structure_for_bit(df, smiles_col, bit_id, prefix="FRAGMENT", radius=2, n_bits=2048):
   """
    Tracks down the first molecule containing bit_id and saves the fragment
    as a PNG image ready to open in Windows.
    """
   for sm in df[smiles_col].dropna():
        mol = Chem.MolFromSmiles(str(sm))
        if mol is not None:
            bit_info = {}
            _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
            
            if bit_id in bit_info:
                # 1. Generate the image with RDKit
                img = Draw.DrawMorganBit(mol, bit_id, bit_info)
                filename_png = f"{prefix}_bit_{bit_id}.png"
                
                # 2. If it is a PIL image (RDKit's standard behavior)
                if isinstance(img, Image.Image):
                    img.save(filename_png)
                    print(f"   -> Found in SMILES: {sm[:30]}... | Saved: {filename_png}")
                    return
                
                # 3. If RDKit returns an SVG/string object, clean up the XML and save a valid .svg
                elif isinstance(img, str) or hasattr(img, 'data'):
                    svg_text = img.data if hasattr(img, 'data') else str(img)
                    if "<svg" in svg_text:
                        # Trim any leading text so it starts exactly at <svg
                        clean_svg = svg_text[svg_text.find("<svg"):]
                        filename_svg = f"{prefix}_bit_{bit_id}.svg"
                        with open(filename_svg, "w", encoding="utf-8") as f:
                            f.write(clean_svg)
                        print(f"   -> Found in SMILES: {sm[:30]}... | Saved: {filename_svg}")
                        return
                        
                # 4. General fallback: save through 'save' if it exists
                elif hasattr(img, 'save'):
                    img.save(filename_png)
                    print(f"   -> Found in SMILES: {sm[:30]}... | Saved: {filename_png}")
                    return
        print(f"   -> Bit {bit_id} not present in the molecules analyzed.")
def extract_and_save_top_bits(df, smiles_col, fp_scores, top_k=5, radius=2, n_bits=2048):
    """
    Sorts the bits by their attribution value and exports the images of the top fragments.
    """
    # Bits with the MOST NEGATIVE attribution (they lower delta E_ST -> favor TADF)
    top_tadf_bits = np.argsort(fp_scores)[:top_k]
    
    # Bits with the MOST POSITIVE attribution (they raise delta E_ST -> disfavor TADF)
    top_anti_bits = np.argsort(fp_scores)[-top_k:][::-1]
    
    print("\nTOP FRAGMENTS THAT FAVOR TADF (they lower delta E_ST):")
    print("-" * 65)
    for rank, bit_id in enumerate(top_tadf_bits, 1):
        score = fp_scores[bit_id]
        print(f"Rank {rank} | Bit ID: {bit_id} | Attribution: {score:.6f} eV")
        find_structure_for_bit(df, smiles_col, bit_id, prefix="FAVORS_TADF", radius=radius, n_bits=n_bits)

    print("\nTOP FRAGMENTS THAT DISFAVOR TADF (they raise delta E_ST):")
    print("-" * 65)
    for rank, bit_id in enumerate(top_anti_bits, 1):
        score = fp_scores[bit_id]
        print(f"Rank {rank} | Bit ID: {bit_id} | Attribution: {score:.6f} eV")
        find_structure_for_bit(df, smiles_col, bit_id, prefix="DISFAVORS_TADF", radius=radius, n_bits=n_bits)


# ==========================================
# MODIFIED MAIN PIPELINE
# ==========================================
def run_tadf_mlp_pipeline():
    CSV_PATH = "C:\\Users\\mafo_\\Desktop\\mk_predictions\\SolutionData.csv"
    SMILES_COL = "name.smiles"
    SOLVENT_COL = "state"
    TARGET_COL = "raw_value"
    
    # 1. Load the data
    df = load_and_clean_real_dataset(CSV_PATH, smiles_col=SMILES_COL, target_col=TARGET_COL)
    
    # 2. Random split
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    # 3. Extract features
    print("\n3. Building input vectors (2048 fingerprint bits + 1 solvent feature = 2049)...")
    X_train = prepare_features(train_df[SMILES_COL].tolist(), train_df[SOLVENT_COL].tolist())
    X_test  = prepare_features(test_df[SMILES_COL].tolist(), test_df[SOLVENT_COL].tolist())
    
    y_train = train_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    # 4. DataLoaders
    train_dataset = FingerprintDataset(X_train, y_train)
    test_dataset  = FingerprintDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    # 5. Initialize the MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TADF_MLP(input_dim=2049, hidden_dim1=256, hidden_dim2=64, dropout_rate=0.20).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # 6. Training loop
    NUM_EPOCHS = 80
    print(f"\nTraining the MLP model for {NUM_EPOCHS} epochs...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            x_batch = batch["x"].to(device)
            y_batch = batch["y"].to(device)
            
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x_batch.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoch [{epoch}/{NUM_EPOCHS}] - MSE Loss: {epoch_loss:.5f}")

    print("\nTraining completed!")

    torch.save(model.state_dict(), "tadf_mlp_model.pt") # save information about the trained model

    # =========================================================
    # CAPTUM AND XAI SECTION (THIS IS WHERE THE INTEGRATION LIVES)
    # =========================================================
    print("\nComputing attributions with Captum (Integrated Gradients)...")
    
    # 1. Make sure X_test is a tensor on the correct device (CPU/CUDA)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    baseline = torch.zeros(1, 2049).to(device)
    
    model.eval()
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(X_test_tensor, baseline, target=0, return_convergence_delta=True)

    # 2. Average of the attributions
    promedio_atrb = attributions.mean(dim=0).cpu().detach().numpy()
    
    # Separate the 2048 molecule bits from bit 2048 (solvent)
    fp_scores = promedio_atrb[:2048]
    solvent_score = promedio_atrb[2048]
    
    print(f"   -> Average solvent attribution (eps): {solvent_score:.6f} eV")

    # 3. Plot the global bit attribution
    plt.figure(figsize=(10, 4), dpi=120)
    plt.bar(range(len(fp_scores)), fp_scores, color=np.where(fp_scores < 0, '#2ca02c', '#d62728'), width=1.5)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('Morgan fingerprint bit index (0 - 2047)', fontsize=11, fontweight='bold')
    plt.ylabel('Average attribution to ΔE_ST (eV)', fontsize=11, fontweight='bold')
    plt.title('Global bit importance (Captum Integrated Gradients)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 4. Extract and save the PNG images of the top fragments
    extract_and_save_top_bits(df, smiles_col=SMILES_COL, fp_scores=fp_scores, top_k=5)

    # =========================================================
    # 7. Final evaluation
    # =========================================================
    print("\nEvaluating the MLP on the test set...")
    metrics = evaluate_model_on_test(model, test_loader, device)

if __name__ == "__main__":
    run_tadf_mlp_pipeline()
