import torch
import torch.nn.functional as F
from rdkit import Chem

import torch
from rdkit import Chem

class TADFDatasetInfos:
    def __init__(self, dataset_path="data/tadf_dataset.pt"):
        self.dataset = torch.load(dataset_path, weights_only=False)
        
        # 1. Lista base de átomos y tipos de enlaces
        # ═══════════════════════════════════════════════════════════════════════
        # VOCABULARIO ALINEADO CON GUACAMOL (12 tipos, mismo ORDEN que el checkpoint).
        # Antes teníamos 13 tipos en otro orden (con una 'H' que nunca se usa: el
        # dataset tiene 0 hidrógenos explícitos en 29183 átomos). Ese desajuste hacía
        # que mlp_in_X y mlp_out_X se descartaran al cargar el checkpoint y se
        # reinicializaran al azar — obligando al modelo a re-aprender química desde
        # cero con solo 452 moléculas (de ahí las cadenas alifáticas sin anillos).
        # Con este orden, las dimensiones son mlp_out_X=12 y mlp_in_X=12+8=20, que es
        # exactamente lo que espera el checkpoint, así que se hereda todo su
        # conocimiento estructural (anillos, aromaticidad, valencias).
        # REQUIERE el dataset reindexado con fix_dataset_to_guacamol_vocab.py.
        # ═══════════════════════════════════════════════════════════════════════
        self.atom_decoder = ['C', 'N', 'O', 'F', 'B', 'Br', 'Cl', 'I', 'P', 'S', 'Se', 'Si']
        self.valencies = [4, 3, 2, 1, 3, 1, 1, 1, 3, 2, 2, 4]
        self.num_bond_types = 5  # 0: Sin enlace, 1: Simple, 2: Doble, 3: Triple, 4: Aromático
        self.remove_h = False

        weights_dict = {
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'B': 10.81,
            'Br': 79.904, 'Cl': 35.453, 'I': 126.904, 'P': 30.974, 'S': 32.06,
            'Se': 78.971, 'Si': 28.085
        }
        
        # Ajustar si hay más índices de átomos en los datos de los que hay en atom_decoder
        max_atom_idx = max(int(data.x.max().item()) for data in self.dataset if data.x.numel() > 0)
        self.num_atom_types = max(len(self.atom_decoder), max_atom_idx + 1)
        self.num_classes = self.num_atom_types

        # Asegurar un peso por cada índice posible de átomo
        self.atom_weights = {
            i: weights_dict.get(self.atom_decoder[i], 12.0) if i < len(self.atom_decoder) else 12.0 
            for i in range(self.num_atom_types)
        }

        # 2. Dimensiones de entrada/salida para DiGress
        self.output_dims = {
            'X': self.num_atom_types,
            'E': self.num_bond_types,
            'y': 0  
        }
        self.input_dims = self.output_dims

        # 3. Calcular estadísticas de los datos
        self._compute_statistics()

    def _compute_statistics(self):
        atom_counts = torch.zeros(self.num_atom_types)
        bond_counts = torch.zeros(self.num_bond_types)
        
        max_nodes = 0
        node_num_list = []
        
        # Mapeo de tipo de enlace a orden de valencia
        bond_orders = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
        max_valency = 10
        valency_counts = torch.zeros(max_valency + 1)

        for data in self.dataset:
            # Conteo de átomos
            for atom in data.x:
                atom_counts[atom.long()] += 1
            
            num_nodes = data.x.size(0)
            node_num_list.append(num_nodes)
            if num_nodes > max_nodes:
                max_nodes = num_nodes
                
            # Procesamiento de enlaces y valencia
            if hasattr(data, 'edge_index') and hasattr(data, 'edge_attr') and data.edge_index.numel() > 0:
                row, col = data.edge_index[0], data.edge_index[1]
                edge_attr = data.edge_attr.squeeze() if data.edge_attr.dim() > 1 else data.edge_attr
                
                node_valencies = torch.zeros(num_nodes)
                existing_edges_count = 0

                for i in range(len(edge_attr)):
                    u, v = row[i].item(), col[i].item()
                    b_type = int(edge_attr[i].item())
                    
                    bond_counts[b_type] += 1
                    
                    # Valencia corregida: Para la arista dirigida u -> v, sumamos el orden completo a u
                    order = bond_orders.get(b_type, 1.0)
                    node_valencies[u] += order
                    existing_edges_count += 1
                
                # Conteo de pares "Sin enlace" (tipo 0) en la matriz densa N x N
                total_pairs = num_nodes * (num_nodes - 1)
                no_bond_count = max(0, total_pairs - existing_edges_count)
                bond_counts[0] += no_bond_count

                # Histograma de valencias redondeado
                for val in node_valencies:
                    v_int = min(int(round(val.item())), max_valency)
                    valency_counts[v_int] += 1

        self.max_n_nodes = max_nodes

        # Peso molecular máximo
        max_mol_weight = max(
            sum(self.atom_weights[int(atom.item())] for atom in data.x) 
            for data in self.dataset
        )
        self.max_weight = max_mol_weight
        
        # Normalización a distribuciones de probabilidad
        self.atom_types = atom_counts / atom_counts.sum()
        self.node_types = self.atom_types
        self.edge_types = bond_counts / bond_counts.sum()
        
        nodes_histogram = torch.zeros(max_nodes + 1)
        for n in node_num_list:
            nodes_histogram[n] += 1
            
        self.nodes_dist = nodes_histogram / nodes_histogram.sum()
        self.n_nodes = self.nodes_dist
        
        # Distribución de valencias
        valency_sum = valency_counts.sum()
        if valency_sum > 0:
            self.valency_distribution = valency_counts / valency_sum
        else:
            self.valency_distribution = torch.ones(max_valency + 1) / (max_valency + 1)

    def to_molecule(self, X_single, E_single):
        """
        Convierte matrices de nodos (X) y aristas (E) a RDKit Mol.
        Soporta tanto tensores discretos (1D/2D) como vectores One-Hot (2D/3D).
        """
        mol = Chem.RWMol()
        
        # 1. Detección automática para Átomos (X)
        if X_single.dim() > 1 and X_single.size(-1) > 1:
            atom_types = torch.argmax(X_single, dim=-1)
        else:
            atom_types = X_single.squeeze()

        for idx_tensor in atom_types:
            idx = int(idx_tensor.item())
            if idx < len(self.atom_decoder):
                symbol = self.atom_decoder[idx]
            else:
                # FIX: antes esto se mapeaba a 'C' en silencio. Tu dataset real tiene
                # índices de átomo hasta 10, pero atom_decoder solo cubre 0-9.
                # Avisamos en vez de mentir sobre la estructura química.
                symbol = 'C'
                print(f"⚠️ Índice de átomo {idx} fuera de atom_decoder (len={len(self.atom_decoder)}); "
                      f"se está usando 'C' como placeholder. Revisa qué elemento falta en atom_decoder.")
            mol.AddAtom(Chem.Atom(symbol))

        # 2. Detección automática para Enlaces (E)
        if E_single.dim() > 2:
            bond_types = torch.argmax(E_single, dim=-1)
        else:
            bond_types = E_single

        n_nodes = atom_types.size(0)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                b_type = int(bond_types[i, j].item())
                if b_type == 1:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                elif b_type == 2:
                    mol.AddBond(i, j, Chem.BondType.DOUBLE)
                elif b_type == 3:
                    mol.AddBond(i, j, Chem.BondType.TRIPLE)
                elif b_type == 4:
                    mol.AddBond(i, j, Chem.BondType.AROMATIC)

        return mol.GetMol()