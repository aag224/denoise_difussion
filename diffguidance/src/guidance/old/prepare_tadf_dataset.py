import torch
from torch_geometric.data import Data
from rdkit import Chem
#from rdkit.Chem import pdq
import pandas as pd
import os

CSV_PATH = "C:\\Users\\mafo_\\Desktop\\mk_predictions\\SolutionData.csv"
SMILES_COL = "name.smiles"
SOLVENT_COL = "state"
TARGET_COL = "raw_value"
    

df = pd.read_csv(CSV_PATH)

def smiles_to_digress_graph(smiles_list, atom_decoder=['C', 'N', 'O', 'F', 'S', 'P', 'B', 'Br', 'Cl', 'H']):
    """
    Convierte una lista de SMILES en objetos Data de PyTorch Geometric compatibles con DiGress.
    """
    atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}
    data_list = []

    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        
        Chem.SanitizeMol(mol)
        
        # 1. Nodos (Tipos de átomos)
        nodes = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in atom_encoder:
                nodes.append(atom_encoder[symbol])
            else:
                # Si hay un átomo no presente en tu lista
                nodes.append(len(atom_encoder)) 
        
        x = torch.tensor(nodes, dtype=torch.long)

        # 2. Aristas (Tipos de enlaces: 1=SIMPLE, 2=DOBLE, 3=TRIPLE, 4=AROMATICO)
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Enlaces bidireccionales
            edge_indices.extend([[i, j], [j, i]])
            
            b_type = bond.GetBondType()
            if b_type == Chem.rdchem.BondType.SINGLE:
                b_val = 1
            elif b_type == Chem.rdchem.BondType.DOUBLE:
                b_val = 2
            elif b_type == Chem.rdchem.BondType.TRIPLE:
                b_val = 3
            elif b_type == Chem.rdchem.BondType.AROMATIC:
                b_val = 4
            else:
                b_val = 1
                
            edge_attrs.extend([b_val, b_val])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

        # Crear el objeto de grafo
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=sm)
        data_list.append(data)

    return data_list


os.makedirs("data", exist_ok=True)
tadf_dataset = smiles_to_digress_graph(df[SMILES_COL])
torch.save(tadf_dataset, "C:\\Users\\mafo_\\Desktop\\mk_predictions\\trying_one\\diffguidance\\src\\guidance\\data\\tadf_dataset.pt")
print(f"Dataset procesado con {len(tadf_dataset)} grafos válidos guardados en data/tadf_dataset.pt")