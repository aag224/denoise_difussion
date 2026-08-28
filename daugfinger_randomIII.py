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
# 1. CARGA Y LIMPIEZA DE DATOS CON RDKIT
# ==========================================
def load_and_clean_real_dataset(csv_path, smiles_col="name.smiles", target_col="raw_value"):
    """
    Carga el CSV, elimina valores nulos y verifica que las cadenas SMILES 
    sean interpretables por RDKit.
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
# 2. EXTRACCIÓN DE MORGAN FINGERPRINTS
# ==========================================
# Mapeo flexible para tolerar variantes de nombres en español e inglés
SOLVENT_EPSILON = {
    'C1=CC=CC=C1': 2.38, 'C1=CC=CC=C2': 2.38,'C1=CC=CC=C3': 2.38, 'C1=CC=CC=C4': 2.38,
    'CC1CCOC1': 6.97, 'CC1OCCC1': 6.97, 'CC1CCCO1': 6.97,
    'C1CCCO2': 7.58, 'tC1CCCO3"': 7.58, 'C1CCCO4': 7.58, 'C1CCCO5': 7.58, 'C1CCCO6': 7.58,
    'ClC([H])([H])Cl': 8.93,
    'CC(Cl)Cl': 10.36
}

def prepare_features(smiles_list, solvent_list, radius=2, n_bits=2048):
    """
    Combina los 2048 bits de Morgan Fingerprints con la constante 
    dieléctrica del disolvente normalizada (Dimensión total = 2049).
    """
    X_features = []
    
    for sm, sol in zip(smiles_list, solvent_list):
        # 1. Extraer Fingerprint
        mol = Chem.MolFromSmiles(str(sm))
        if mol is not None:
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)
        else:
            fp = np.zeros(n_bits, dtype=np.float32)
            
        # 2. Extraer y escalar constante dieléctrica (ε / 12.0 para escala ~0 a 1)
        sol_clean = str(sol).strip().lower()
        eps = SOLVENT_EPSILON.get(sol_clean, 2.38) # Tolueno por defecto si falta algún dato
        eps_scaled = np.array([eps / 12.0], dtype=np.float32)
        
        # 3. Concatenar: Vector de 2049 dimensiones
        combined_vector = np.concatenate([fp, eps_scaled])
        X_features.append(combined_vector)
        
    return np.array(X_features, dtype=np.float32)
# ==========================================
# 3. DATASET DE PYTORCH
# ==========================================
class FingerprintDataset(Dataset):
    """
    Dataset para vectores tabulares numéricos (Fingerprints).
    """
    def __init__(self, fps_matrix, targets):
        self.X = torch.tensor(fps_matrix, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {"x": self.X[idx], "y": self.y[idx]}

# ==========================================
# 4. ARQUITECTURA RED NEURONAL DENSA (MLP)
# ==========================================
class TADF_MLP(nn.Module):
    """
    Perceptrón Multicapa con Normalización de Lote y Dropout 
    para regularizar en datasets pequeños.
    """
    def __init__(self, input_dim=2048, hidden_dim1=256, hidden_dim2=64, dropout_rate=0.20):
        super(TADF_MLP, self).__init__()
        
        self.network = nn.Sequential(
            # Capa de Entrada -> Oculta 1
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Capa Oculta 1 -> Oculta 2
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Capa de Salida (Regresión)
            nn.Linear(hidden_dim2, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# ==========================================
# 5. GRÁFICA DE PARIDAD Y EVALUACIÓN
# ==========================================
def evaluate_and_plot_parity(y_true, y_pred, title="ΔE_ST-Parity Plot (MLP/Fingerprint)"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    plt.figure(figsize=(7, 6), dpi=120)
    
    # 1. TAMAÑO DE PUNTOS: Se incrementa 's' de 50 a 95
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
    
    # --- Recuadro de métricas ---
    metrics_box = (
        f"MAE  = {mae:.4f} eV\n"
        f"RMSE = {rmse:.4f} eV\n"
        f"R²   = {r2:.4f}"
    )
    
    # 2. TAMAÑO DEL RECUADRO DE MÉTRICAS: Se incrementa 'fontsize' a 11.5 y el relleno 'pad' a 0.7
    plt.gca().text(
        0.95, 0.05, metrics_box, transform=plt.gca().transAxes,
        fontsize=14, horizontalalignment='right', verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 3. TAMAÑO DE LEYENDA: Se agrega 'fontsize=11'
    plt.legend(loc='upper left', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📈 Resumen de Evaluación en Test (MLP + Fingerprints):")
    print(f"   - Error Absoluto Medio (MAE):   {mae:.4f} eV")
    print(f"   - Error Cuadrático Medio (RMSE): {rmse:.4f} eV")
    print(f"   - Coeficiente R²:               {r2:.4f}")
    
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
# 6. FUNCIÓN PRINCIPAL (PIPELINE)
# ==========================================
def find_structure_for_bit(df, smiles_col, bit_id, prefix="FRAGMENTO", radius=2, n_bits=2048):
   """
    Rastrea la primera molécula que contiene el bit_id y guarda el fragmento 
    como una imagen PNG lista para abrir en Windows.
    """
   for sm in df[smiles_col].dropna():
        mol = Chem.MolFromSmiles(str(sm))
        if mol is not None:
            bit_info = {}
            _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
            
            if bit_id in bit_info:
                # 1. Generar la imagen con RDKit
                img = Draw.DrawMorganBit(mol, bit_id, bit_info)
                filename_png = f"{prefix}_bit_{bit_id}.png"
                
                # 2. Si es una imagen PIL (comportamiento estándar de RDKit)
                if isinstance(img, Image.Image):
                    img.save(filename_png)
                    print(f"   -> Encontrado en SMILES: {sm[:30]}... | Guardado: {filename_png}")
                    return
                
                # 3. Si RDKit devuelve un objeto SVG/String, limpiamos el XML y guardamos .svg válido
                elif isinstance(img, str) or hasattr(img, 'data'):
                    svg_text = img.data if hasattr(img, 'data') else str(img)
                    if "<svg" in svg_text:
                        # Cortar cualquier texto previo para que empiece exactamente en <svg
                        clean_svg = svg_text[svg_text.find("<svg"):]
                        filename_svg = f"{prefix}_bit_{bit_id}.svg"
                        with open(filename_svg, "w", encoding="utf-8") as f:
                            f.write(clean_svg)
                        print(f"   -> Encontrado en SMILES: {sm[:30]}... | Guardado: {filename_svg}")
                        return
                        
                # 4. Respaldo general para guardar mediante save si existe
                elif hasattr(img, 'save'):
                    img.save(filename_png)
                    print(f"   -> Encontrado en SMILES: {sm[:30]}... | Guardado: {filename_png}")
                    return
        print(f"   -> Bit {bit_id} no presente en las moléculas analizadas.")
def extract_and_save_top_bits(df, smiles_col, fp_scores, top_k=5, radius=2, n_bits=2048):
    """
    Ordena los bits por su valor de atribución y exporta las imágenes de los fragmentos top.
    """
    # Bits con atribución MÁS NEGATIVA (Reducen ΔE_ST -> Favorecen TADF)
    top_tadf_bits = np.argsort(fp_scores)[:top_k]
    
    # Bits con atribución MÁS POSITIVA (Aumentan ΔE_ST -> Desfavorecen TADF)
    top_anti_bits = np.argsort(fp_scores)[-top_k:][::-1]
    
    print("\n🟢 TOP FRAGMENTOS QUE FAVORECEN TADF (Reducen ΔE_ST):")
    print("-" * 65)
    for rank, bit_id in enumerate(top_tadf_bits, 1):
        score = fp_scores[bit_id]
        print(f"Rank {rank} | Bit ID: {bit_id} | Atribución: {score:.6f} eV")
        find_structure_for_bit(df, smiles_col, bit_id, prefix="FAVORECE_TADF", radius=radius, n_bits=n_bits)

    print("\n🔴 TOP FRAGMENTOS QUE DESFAVORECEN TADF (Aumentan ΔE_ST):")
    print("-" * 65)
    for rank, bit_id in enumerate(top_anti_bits, 1):
        score = fp_scores[bit_id]
        print(f"Rank {rank} | Bit ID: {bit_id} | Atribución: {score:.6f} eV")
        find_structure_for_bit(df, smiles_col, bit_id, prefix="DESFAVORECE_TADF", radius=radius, n_bits=n_bits)


# ==========================================
# PIPELINE PRINCIPAL MODIFICADO
# ==========================================
def run_tadf_mlp_pipeline():
    CSV_PATH = "C:\\Users\\mafo_\\Desktop\\mk_predictions\\SolutionData.csv"
    SMILES_COL = "name.smiles"
    SOLVENT_COL = "state"
    TARGET_COL = "raw_value"
    
    # 1. Cargar datos
    df = load_and_clean_real_dataset(CSV_PATH, smiles_col=SMILES_COL, target_col=TARGET_COL)
    
    # 2. Random Split
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    # 3. Extraer características
    print("\n3. Generando Vectores de Entrada (2048 bits Fingerprint + 1 Feature Disolvente = 2049)...")
    X_train = prepare_features(train_df[SMILES_COL].tolist(), train_df[SOLVENT_COL].tolist())
    X_test  = prepare_features(test_df[SMILES_COL].tolist(), test_df[SOLVENT_COL].tolist())
    
    y_train = train_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    # 4. DataLoaders
    train_dataset = FingerprintDataset(X_train, y_train)
    test_dataset  = FingerprintDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    # 5. Inicializar MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TADF_MLP(input_dim=2049, hidden_dim1=256, hidden_dim2=64, dropout_rate=0.20).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # 6. Bucle de Entrenamiento
    NUM_EPOCHS = 80
    print(f"\n⚡ Entrenando el modelo MLP durante {NUM_EPOCHS} épocas...")
    
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
            print(f"   Época [{epoch}/{NUM_EPOCHS}] - MSE Loss: {epoch_loss:.5f}")

    print("\n✅ ¡Entrenamiento completado!")

    torch.save(model.state_dict(), "tadf_mlp_model.pt") # save information about the trained model

    # =========================================================
    # 🔍 SECCIÓN CAPTUM Y XAI (AQUÍ ESTÁ LA INTEGRACIÓN)
    # =========================================================
    print("\n🔍 Calculando Atribuciones con Captum (Integrated Gradients)...")
    
    # 1. Asegurar que X_test esté como Tensor en el DEVICE correcto (CPU/CUDA)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    baseline = torch.zeros(1, 2049).to(device)
    
    model.eval()
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(X_test_tensor, baseline, target=0, return_convergence_delta=True)

    # 2. Promedio de atribuciones
    promedio_atrb = attributions.mean(dim=0).cpu().detach().numpy()
    
    # Separar los 2048 bits de la molécula del bit 2048 (Disolvente)
    fp_scores = promedio_atrb[:2048]
    solvent_score = promedio_atrb[2048]
    
    print(f"   -> Atribución promedio del Disolvente (ε): {solvent_score:.6f} eV")

    # 3. Graficar atribución global de bits
    plt.figure(figsize=(10, 4), dpi=120)
    plt.bar(range(len(fp_scores)), fp_scores, color=np.where(fp_scores < 0, '#2ca02c', '#d62728'), width=1.5)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('Índice del Bit Morgan Fingerprint (0 - 2047)', fontsize=11, fontweight='bold')
    plt.ylabel('Atribución Promedio a ΔE_ST (eV)', fontsize=11, fontweight='bold')
    plt.title('Importancia Global de Bits (Captum Integrated Gradients)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 4. Extraer y guardar las imágenes PNG de los Top Fragmentos
    extract_and_save_top_bits(df, smiles_col=SMILES_COL, fp_scores=fp_scores, top_k=5)

    # =========================================================
    # 7. Evaluación Final
    # =========================================================
    print("\n📊 Evaluando MLP en el Test Set...")
    metrics = evaluate_model_on_test(model, test_loader, device)

if __name__ == "__main__":
    run_tadf_mlp_pipeline()