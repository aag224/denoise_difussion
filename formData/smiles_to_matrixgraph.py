import torch
import numpy as np
from rdkit import Chem

# 1. Chem vocabulary for common atoms and bond types in TADF molecules
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br']  # List of common atoms in TADF molecules
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]

def smiles_to_graph_matrices(smiles, max_nodes=100):
    """
    Convert a SMILES string into graph matrices (adjacency and features) suitable for GNN.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    # Kekulize the molecule to ensure proper representation of aromatic bonds
    Chem.Kekulize(mol, clearAromaticFlags=False)
    
    num_atoms = mol.GetNumAtoms()
    if num_atoms > max_nodes:
        raise ValueError(f"The molecule exceeds the maximum number of nodes ({num_atoms} > {max_nodes})")

    # ==========================================
    # 1. FORM A MATRIX X [max_nodes, num_atom_types + 1]
    # (Las channel it is for 'Padding/No atom')
    # ==========================================
    num_atom_types = len(ATOM_LIST)
    X = np.zeros((max_nodes, num_atom_types + 1), dtype=np.float32)
    
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in ATOM_LIST:
            atom_idx = ATOM_LIST.index(symbol)
            X[i, atom_idx] = 1.0
        else:
            # atoms not in the list are treated as 'No atom' (Padding)
            X[i, -1] = 1.0  
            
    # Define 'No atom' in none valid positions (padding) for the rest of the matrix
    for i in range(num_atoms, max_nodes):
        X[i, -1] = 1.0

    # ==========================================
    # 2. CONSTRUCT ADJACENCY MATRIX A [max_nodes, max_nodes, num_bond_types + 1]
    # (Channel 0 = No Bond, Channel 1..4 = Bond Types)
    # ==========================================
    num_bond_types = len(BOND_LIST)
    A = np.zeros((max_nodes, max_nodes, num_bond_types + 1), dtype=np.float32)
    
    # Initialize all as 'No Bond' (Channel 0 = 1.0)
    A[:, :, 0] = 1.0
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        
        if bond_type in BOND_LIST:
            bond_idx = BOND_LIST.index(bond_type) + 1 # +1 to account for 'No Bond' channel
        else:
            bond_idx = 1 # Default value for single bond
            
        # The adjacency matrix is symmetric (Undirected Graph)
        A[i, j, 0] = 0.0
        A[j, i, 0] = 0.0
        A[i, j, bond_idx] = 1.0
        A[j, i, bond_idx] = 1.0

    # Convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    A_tensor = torch.tensor(A, dtype=torch.float32)
    
    return X_tensor, A_tensor


# Ejemplo de uso:
smiles_ejemplo = "c1ccc2c(c1)c3ccccc3n2c4ccc(cc4)C(=O)c5ccccc5" # Molécula TADF típica
X_mat, A_mat = smiles_to_graph_matrices(smiles_ejemplo, max_nodes=100)

print("Dimensiones de X (Nodos):", X_mat.shape)      # Output: torch.Size([50, 9])
print("Dimensiones de A (Adyacencia):", A_mat.shape) # Output: torch.Size([50, 50, 5])