import sys
import os

# 1. Obtener la ruta del directorio raíz del proyecto (diffguidance)
# sube dos niveles desde src/guidance/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# 1. Obtener la ruta a la carpeta 'trying_one' (subiendo 3 niveles desde src/guidance/)
trying_one_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if trying_one_path not in sys.path:
    sys.path.insert(0, trying_one_path)

# 2. Obtener la ruta a la carpeta 'mk_predictions' (subiendo 4 niveles)
mk_predictions_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if mk_predictions_path not in sys.path:
    sys.path.insert(0, mk_predictions_path)

# 2. Agregar la raíz al inicio del path de Python
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# -------------------------------------------------------------
# AHÓRA SÍ PUEDES IMPORTAR TUS MÓDULOS DE 'src':
# -------------------------------------------------------------
from src.models.transformer_model import GraphTransformer

import numpy as np
import torch
import pytorch_lightning as pl
import time
import wandb
import os
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
import networkx as nx
from src.metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.metrics.train_metrics import TrainLossDiscrete
import src.utils as utils

# packages for conditional generation with guidance
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFHasAllMoleculeParams, MMFFOptimizeMolecule
from rdkit import Chem
from rdkit.Chem import AllChem
import math
try:
    import psi4
except ModuleNotFoundError:
    print("PSI4 not found")
from src.analysis.rdkit_functions import build_molecule, mol2smiles, build_molecule_with_partial_charges
import pickle
import pandas as pd

from trying_one.daugfinger_randomIII import TADF_MLP

import torch
import pytorch_lightning as pl

# 1. Importación corregida desde src/
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
import torch
from torch_geometric.loader import DataLoader

# Importaciones de los módulos de DiGress desde tu carpeta src/
from src.datasets.tadf_dataset import TADFDatasetInfos
from src.metrics.abstract_metrics import TrainAbstractMetrics as TrainMoleculeMetrics
from src.metrics.molecular_metrics import SamplingMolecularMetrics as SamplingMoleculeMetrics
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, guidance_model=None, load_model=False):
        super().__init__()

        # add for test
        if load_model:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.guidance = {'use_guidance': True, 'lambda_guidance': 0.5}

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.num_classes = dataset_infos.num_classes
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)
        # Marginal noise schedule
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                          y_classes=self.ydim_output)
        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = getattr(cfg.general, 'log_every_steps', 50)
        self.number_chain_steps = getattr(cfg.general, 'number_chain_steps', 25)
        self.best_val_nll = 1e8
        self.val_counter = 0

        # specific properties to generate molecules
        self.cond_val = MeanAbsoluteError()
        self.num_valid_molecules = 0
        self.num_total = 0

        if guidance_model is not None:
            self.guidance_mlp = guidance_model
            self.guidance_mlp.eval()

            for param in self.guidance_mlp.parameters():
                param.requires_grad = False
        else:
            self.guidance_mlp = None

        self.use_mlp_guidance = getattr(cfg.general, 'use_mlp_guidance', True)
        self.gamma_guidance = getattr(cfg.general, 'gamma_guidance', 2.0)
        self.guidance_mode = getattr(cfg.general, 'guidance_mode', 'minimize')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.train.lr, amsgrad=True,
                                 weight_decay=self.args.train.weight_decay)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_step(self, data, i):
        print(f'Select No.{i+1} test molecule')
        # Extract properties
        target_properties = data.y.clone()

        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)
        print("TARGET PROPERTIES", target_properties)

        start = time.time()

        ident = 0
        samples = self.sample_batch(batch_id=ident, batch_size=10, num_nodes=None,
                                    save_final=10,
                                    keep_chain=1,
                                    number_chain_steps=self.number_chain_steps,
                                    input_properties=target_properties)
        print(f'Sampling took {time.time() - start:.2f} seconds\n')

        self.save_cond_samples(samples, target_properties, file_path=os.path.join(os.getcwd(), f'cond_smiles{i}.pkl'))
        # save conditional generated samples
        mae = self.cond_sample_metric(samples, target_properties)
        return {'mae': mae}

    def test_epoch_end(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        final_mae = self.cond_val.compute()
        final_validity = self.num_valid_molecules / self.num_total
        print("Final MAE", final_mae)
        print("Final validity", final_validity * 100)

        wandb.run.summary['final_MAE'] = final_mae
        wandb.run.summary['final_validity'] = final_validity
        wandb.log({'final mae': final_mae,
                   'final validity': final_validity})

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data


    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, input_properties=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            if hasattr(self.node_dist, 'sample_n'):
                n_nodes = self.node_dist.sample_n(batch_size, self.device)
            else:
                # Si node_dist es un Tensor plano, muestreamos con multinomial
                prob = self.node_dist / self.node_dist.sum()
                n_nodes = torch.multinomial(prob, batch_size, replacement=True).to(self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        
        # Validar limites de pasos
        if number_chain_steps >= self.T:
            number_chain_steps = self.T - 1

        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            t_norm = t_array / self.T
            s_norm = s_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, input_properties)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            if keep_chain > 0 and number_chain_steps > 0:
                write_index = (s_int * number_chain_steps) // self.T
                if write_index < chain_X.size(0):
                    chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                    chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        print("Examples of generated graphs:")
        for i in range(min(5, X.shape[0])):
            print("E: ", E[i])
            print("X: ", X[i])

        # Prepare the chain for saving
        if keep_chain > 0 and number_chain_steps > 0 and chain_X.size(0) > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None and keep_chain > 0 and number_chain_steps > 0:
            print('Visualizing chains starts!')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            
            # Obtención segura del nombre de experimento
            exp_name = getattr(getattr(self, 'args', None), 'general', None)
            exp_name = exp_name.name if exp_name and hasattr(exp_name, 'name') else 'tadf_experiment'
            
            result_path = os.path.join(current_path, f'chains/{exp_name}/')
            os.makedirs(result_path, exist_ok=True)

            for i in range(num_molecules):
                try:
                    _ = self.visualization_tools.visualize_chain(
                        result_path,
                        chain_X[:, i, :].numpy(),
                        chain_E[:, i, :].numpy()
                    )
                except Exception as e:
                    print(f"\ Advertencia al generar animación GIF para la molécula {i+1}: {e}")
                
                print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            print('\nVisualizing chains Ends!')

            # Visualize the final molecules
            try:
                model_name = getattr(self, 'name', 'tadf_model')
                epoch = getattr(self, 'current_epoch', 0)
                final_path = os.path.join(current_path, f'graphs/{model_name}/epoch{epoch}_b{batch_id}/')
                os.makedirs(final_path, exist_ok=True)
                self.visualization_tools.visualize(final_path, molecule_list, save_final)
            except Exception as e:
                print(f"⚠️ Advertencia al guardar grafos finales: {e}")

        return molecule_list

    def cond_sample_metric(self, samples, input_properties):
        mols_dipoles = []
        mols_homo = []

        # Hardware side settings (CPU thread number and memory settings used for calculation)
        psi4.set_num_threads(nthread=4)
        psi4.set_memory("5GB")
        psi4.core.set_output_file('psi4_output.dat', False)

        for sample in samples:
            mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)

            try:
                Chem.SanitizeMol(mol)
            except:
                print('invalid chemistry')
                continue

            # Coarse 3D structure optimization by generating 3D structure from SMILES
            mol = Chem.AddHs(mol)
            params = ETKDGv3()
            params.randomSeed = 1
            try:
                EmbedMolecule(mol, params)
            except Chem.rdchem.AtomValenceException:
                print('invalid chemistry')
                continue

            # Structural optimization with MMFF (Merck Molecular Force Field)
            try:
                s = MMFFOptimizeMolecule(mol)
                print(s)
            except:
                print('Bad conformer ID')
                continue

            conf = mol.GetConformer()

            # Convert to a format that can be input to Psi4.
            # Set charge and spin multiplicity (below is charge 0, spin multiplicity 1)

            # Get the formal charge
            fc = 'FormalCharge'
            mol_FormalCharge = int(mol.GetProp(fc)) if mol.HasProp(fc) else Chem.GetFormalCharge(mol)

            sm = 'SpinMultiplicity'
            if mol.HasProp(sm):
                mol_spin_multiplicity = int(mol.GetProp(sm))
            else:
                # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
                NumRadicalElectrons = 0
                for Atom in mol.GetAtoms():
                    NumRadicalElectrons += Atom.GetNumRadicalElectrons()
                TotalElectronicSpin = NumRadicalElectrons / 2
                SpinMultiplicity = 2 * TotalElectronicSpin + 1
                mol_spin_multiplicity = int(SpinMultiplicity)

            mol_input = "%s %s" % (mol_FormalCharge, mol_spin_multiplicity)
            print(mol_input)
            #mol_input = "0 1"

            # Describe the coordinates of each atom in XYZ format
            for atom in mol.GetAtoms():
                mol_input += "\n " + atom.GetSymbol() + " " + str(conf.GetAtomPosition(atom.GetIdx()).x) \
                             + " " + str(conf.GetAtomPosition(atom.GetIdx()).y) \
                             + " " + str(conf.GetAtomPosition(atom.GetIdx()).z)

            try:
                molecule = psi4.geometry(mol_input)
            except:
                print('Can not calculate psi4 geometry')
                continue

            # Convert to a format that can be input to pyscf
            # Set calculation method (functional) and basis set
            level = "b3lyp/6-31G*"

            # Calculation method (functional), example of basis set
            # theory = ['hf', 'b3lyp']
            # basis_set = ['sto-3g', '3-21G', '6-31G(d)', '6-31+G(d,p)', '6-311++G(2d,p)']

            # Perform structural optimization calculations
            print('Psi4 calculation starts!!!')
            #energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)
            try:
                energy, wave_function = psi4.energy(level, molecule=molecule, return_wfn=True)
            except psi4.driver.SCFConvergenceError:
                print("Psi4 did not converge")
                continue

            print('Chemistry information check!!!')

            if self.args.general.guidance_target in ['mu', 'both']:
                dip_x, dip_y, dip_z = wave_function.variable('SCF DIPOLE')[0],\
                                      wave_function.variable('SCF DIPOLE')[1],\
                                      wave_function.variable('SCF DIPOLE')[2]
                dipole_moment = math.sqrt(dip_x**2 + dip_y**2 + dip_z**2) * 2.5417464519
                print("Dipole moment", dipole_moment)
                mols_dipoles.append(dipole_moment)

            if self.args.general.guidance_target in ['homo', 'both']:
                # Compute HOMO (Unit: au= Hartree）
                LUMO_idx = wave_function.nalpha()
                HOMO_idx = LUMO_idx - 1
                homo = wave_function.epsilon_a_subset("AO", "ALL").np[HOMO_idx]

                # convert unit from a.u. to ev
                homo = homo * 27.211324570273
                print("HOMO", homo)
                mols_homo.append(homo)

        num_valid_molecules = max(len(mols_dipoles), len(mols_homo))
        print("Number of valid samples", num_valid_molecules)
        self.num_valid_molecules += num_valid_molecules
        self.num_total += len(samples)

        mols_dipoles = torch.FloatTensor(mols_dipoles)
        mols_homo = torch.FloatTensor(mols_homo)

        if self.args.general.guidance_target == 'mu':
            mae = self.cond_val(mols_dipoles.unsqueeze(1),
                                input_properties.repeat(len(mols_dipoles), 1).cpu())

        elif self.args.general.guidance_target == 'homo':
            mae = self.cond_val(mols_homo.unsqueeze(1),
                                input_properties.repeat(len(mols_homo), 1).cpu())

        elif self.args.general.guidance_target == 'both':
            properties = torch.hstack((mols_dipoles.unsqueeze(1), mols_homo.unsqueeze(1)))
            mae = self.cond_val(properties,
                                input_properties.repeat(len(mols_dipoles), 1).cpu())

        print('Conditional generation metric:')
        print(f'Epoch {self.current_epoch}: MAE: {mae}')
        wandb.log({"val_epoch/conditional generation mae": mae,
                   'Valid molecules': num_valid_molecules})
        return mae

    def cond_fn(self, noisy_data, node_mask, target=None):
        # 1. Obtener t de forma segura
        t = noisy_data['t'] if isinstance(noisy_data, dict) else getattr(noisy_data, 't')

        # 2. Extracción robusta de tensores de nodos (X) y bordes (E)
        def _get_tensor(data, keys):
            if isinstance(data, dict):
                for k in keys:
                    if k in data:
                        return data[k]
            else:
                for k in keys:
                    if hasattr(data, k):
                        return getattr(data, k)
            raise KeyError(f"No se encontró ninguna de las claves {keys} en noisy_data.")

        X_raw = _get_tensor(noisy_data, ['X_t', 'X', 'x_t', 'x'])
        E_raw = _get_tensor(noisy_data, ['E_t', 'E', 'e_t', 'e'])

        # 3. Habilitar explícitamente el cálculo de gradientes
        with torch.enable_grad():
            x_in = X_raw.detach().clone().float().requires_grad_(True)
            e_in = E_raw.detach().clone().float().requires_grad_(True)

            # 4. Aplanar características e instanciar la proyección lineal a 2049 dimensiones
            mlp_input = torch.cat([x_in.flatten(start_dim=1), e_in.flatten(start_dim=1)], dim=-1)

            if not hasattr(self, 'graph_to_fp_proj') or self.graph_to_fp_proj.in_features != mlp_input.shape[-1]:
                self.graph_to_fp_proj = torch.nn.Linear(mlp_input.shape[-1], 2049).to(mlp_input.device)

            projected_input = self.graph_to_fp_proj(mlp_input)
            pred = self.guidance_model(projected_input)

            # 5. Determinar la propiedad objetivo (target) de forma segura
            if target is None:
                target = getattr(self, 'target_y', None)
            if target is None:
                target = torch.full((x_in.shape[0], 1), 0.05, device=x_in.device)

            target = target.type_as(x_in)

            # 6. Cálculo del error cuadrático medio (MSE)
            mse = torch.nn.functional.mse_loss(pred, target)

            # Impresión y reporte opcional de progreso
            t_int = int(t[0].item() * 500)
            if t_int % 10 == 0:
                print(f'Regressor MSE at step {t_int}: {mse.item():.6f}')
            
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({'Guidance MSE': mse.item()})
            except Exception:
                pass  # Si wandb no está configurado, ignora el log sin interrumpir

            # 7. Calcule el gradiente del MSE respecto a x_in y e_in
            grad_x = torch.autograd.grad(mse, x_in, retain_graph=True)[0]
            grad_e = torch.autograd.grad(mse, e_in)[0]

            # 8. Enmascaramiento topológico de nodos y bordes
            x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
            bs, n = x_mask.shape[0], x_mask.shape[1]

            e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
            
            diag_mask = torch.eye(n, device=x_mask.device)
            diag_mask = ~diag_mask.type_as(e_mask1).bool()
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

            mask_grad_x = grad_x * x_mask
            mask_grad_e = grad_e * e_mask1 * e_mask2 * diag_mask

            # Simetrizar la matriz de gradientes de los bordes (E = E^T)
            mask_grad_e = 0.5 * (mask_grad_e + torch.transpose(mask_grad_e, 1, 2))

            return mask_grad_x, mask_grad_e

    def compute_mlp_reward(self, X_0_pred, E_0_pred):
        "reward the prediction"
        fp_tensor = self.extract_fingerprints(X_0_pred, E_0_pred)

        if fp_tensor is None:
            return 1.0
        
        with torch.no_grad():
            pred_est = self.guidance_mlp(fp_tensor).item()

        if self.guidance_mode == "maximize":
            reward = pred_est
        elif self.guidance_mode == "minimize":
            reward = -pred_est
        else:
            reward = 0.0

        weight = torch.exp(torch.tensor(self.gamma_guidance*reward)).item()
        return weight 

    def extract_fingerprints(self, X_single, E_single, n_bits=2048, radius=2):
        "DiGress tensor to a fingerprint of 2049 dimensions"

        mol = self.dataset_info.to_molecules(X_single, E_single)
        if mol is None:
            return None

        try:
            Chem.SanitizeMol(mol)

            fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fp_2048 = np.array(fp_vect, dtype=np.float32)

            solvent_dumm = np.array([2.38/12.0], dtype=np.float32)

            combined_vector = np.concatenate([fp_2048, solvent_dumm])

            tensor_input = torch.tensor(combined_vector,dtype=torch.float32, device=self.device).unsqueeze(0)

            return tensor_input

        except Exception:
            return None


    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, input_properties):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        # # Guidance
        lamb = getattr(getattr(self, 'args', None), 'guidance', None)
        if lamb is not None and hasattr(lamb, 'lambda_guidance'):
            lamb = lamb.lambda_guidance
        else:
            lamb = getattr(self, 'guidance_scale', 2.5)

        grad_x, grad_e = self.cond_fn(noisy_data, node_mask, input_properties)

        p_eta_x = torch.softmax(- lamb * grad_x, dim=-1)
        p_eta_e = torch.softmax(- lamb * grad_e, dim=-1)

        prob_X_unnormalized = p_eta_x * prob_X
        prob_X_unnormalized[torch.sum(prob_X_unnormalized, dim=-1) == 0] = 1e-7
        prob_X = prob_X_unnormalized / torch.sum(prob_X_unnormalized, dim=-1, keepdim=True)

        prob_E_unnormalized = p_eta_e * prob_E
        prob_E_unnormalized[torch.sum(prob_E_unnormalized, dim=-1) == 0] = 1e-7
        prob_E = prob_E_unnormalized / torch.sum(prob_E_unnormalized, dim=-1, keepdim=True)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def save_cond_samples(self, samples, target, file_path):
        cond_results = {'smiles': [], 'input_targets': target}
        invalid = 0
        disconnected = 0

        print("\tConverting conditionally generated molecules to SMILES ...")
        for sample in samples:
            mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                cond_results['smiles'].append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        print("Number of invalid molecules", invalid)
        print("Number of disconnected molecules", disconnected)

        # save samples
        with open(file_path, 'wb') as f:
            pickle.dump(cond_results, f)

        return cond_results

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

guidance_mlp = TADF_MLP(input_dim=2049, hidden_dim1=256, hidden_dim2=64).to(device)
state_dict = torch.load("tadf_mlp_model.pt", map_location=device)
guidance_mlp.load_state_dict(state_dict)

guidance_mlp.eval()

model = DiscreteDenoisingDiffusion(cfg=cfg,
                                   dataset_infos=dataset_infos,
                                   train_metrics=train_metrics,
                                   sampling_metrics=sampling_metrics,
                                   visualization_tools=visualization_tools,
                                   extra_features=extra_features,
                                   domain_features=domain_features,
                                   guidance_model=guidance_mlp,
                                   load_model=False)
"""

def setup_tadf_components(cfg):
    """
    Instancia automáticamente los 7 objetos requeridos por DiscreteDenoisingDiffusion.
    """
    print("1/5 Calculando estadísticas de tu dataset TADF...")
    dataset_infos = TADFDatasetInfos("data/tadf_dataset.pt")
    
    print("2/5 Instanciando métricas de entrenamiento y muestreo...")
    train_metrics = TrainMoleculeMetrics()
    
    # Cargar SMILES originales para evaluar la novedad durante la generación
    dataset_raw = torch.load("data/tadf_dataset.pt", weights_only=False)
    train_smiles = [d.smiles for d in dataset_raw if hasattr(d, 'smiles')]
    sampling_metrics = SamplingMoleculeMetrics(dataset_infos, train_smiles=train_smiles)
    
    print("3/5 Configurando herramientas de visualización 2D...")
    remove_h = getattr(cfg.general, 'remove_h', True) if hasattr(cfg, 'general') else True
    visualization_tools = MolecularVisualization(remove_h=remove_h, dataset_infos=dataset_infos)
    
    print("4/5 Extrayendo características topológicas del grafo (extra_features)...")
    # Configuración por defecto si cfg no la define
    extra_features_type = getattr(cfg.model, 'extra_features', 'all') if hasattr(cfg, 'model') else 'all'
    extra_features = ExtraFeatures(extra_features_type, dataset_info=dataset_infos)
    
    print("5/5 Extrayendo características químicas avanzadas (domain_features)...")
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    
    print("¡Todos los objetos se han construido exitosamente!")
    
    return {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features
    }

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Configuración dummy o cargada desde Hydra
    class DummyConfig:
        class general:
            remove_h = True
            name = "tadf_experiment"
        class model:
            extra_features = 'all'
            diffusion_steps = 500 
            lambda_train = [5, 0]
            n_layers = 5
            diffusion_noise_schedule = 'cosine'
            model_type="discrete"
            extra_features = 'all'
        
            # 🔑 DIMENSIONES CORREGIDAS (Coincidentes con GuacaMol):
            hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 256}  # 'y' cambia de 128 a 256
            hidden_dims = {
                'dx': 256,
                'de': 64,
                'dy': 128,   # 'dy' cambia de 64 a 128
                'n_head': 8,
                'dim_ffX': 256,
                'dim_ffE': 128,
                'dim_ffy': 128
            }
            
    cfg = DummyConfig()
    
    # Generar los 7 objetos de infraestructura
    components = setup_tadf_components(cfg)

    # 2. Cargar tu MLP de guiado
    guidance_mlp = TADF_MLP(input_dim=2049, hidden_dim1=256, hidden_dim2=64).to(device)
    guidance_mlp.load_state_dict(torch.load("tadf_mlp_model.pt", map_location=device))
    guidance_mlp.eval()

    # 3. Instanciar DiGress PASÁNDOLE los componentes de tu dataset TADF
    model = DiscreteDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=components["dataset_infos"],
        train_metrics=components["train_metrics"],
        sampling_metrics=components["sampling_metrics"],
        visualization_tools=components["visualization_tools"],
        extra_features=components["extra_features"],
        domain_features=components["domain_features"],
        guidance_model=guidance_mlp,   # <-- Tu MLP inyectado
        load_model=False
    )

    # 🔑 Añadir la carpeta 'src' al path de Python para que reconozca el módulo 'datasets'
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # 4. Cargar los pesos del checkpoint de Guacamol (o de tu Fine-Tuning)
    guacamol_ckpt_path = "C:\\Users\\mafo_\\Desktop\\mk_predictions\\trying_one\\diffguidance\\src\\guidance\\checkpoint\\guacamol_last.ckpt" 
    checkpoint = torch.load(guacamol_ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    

    print("¡DiGress instanciado, componentes TADF integrados y MLP cargado exitosamente!")
    # 1. Extraer el state_dict (soporta checkpoints estándar y de PyTorch Lightning)
    if "state_dict" in checkpoint:
        raw_state_dict = checkpoint["state_dict"]
    else:
        raw_state_dict = checkpoint

    # 2. Filtrar descartando capas con dimensiones diferentes (debido a TADF vs GuacaMol)
    model_state = model.state_dict()
    filtered_state_dict = {}

    for k, v in raw_state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"⏩ Omitiendo capa '{k}' por ajuste de dataset: Checkpoint {v.shape} vs TADF {model_state[k].shape}")

        # 3. Cargar las capas filtradas en el modelo
        model.load_state_dict(filtered_state_dict, strict=False)
        print("✅ ¡Pesos del backbone preentrenado cargados exitosamente!")

    model.guidance_model = guidance_mlp
    model.eval()

    num_samples = 20
    # Especifica el Delta E_ST objetivo deseado en eV (ejemplo: 0.05 eV)
    target_E_ST_value = 0.05 
    target_y = torch.full((num_samples, 1), target_E_ST_value, device=device, dtype=torch.float32)

    # Fuerza de la guía (gamma): 
    # - Un valor mayor fuerza más a la MLP a cumplir el target, pero si es excesivo puede deformar la validez química.
    # - Valores recomendados para empezar: 1.5, 2.5, 5.0
    guidance_scale = 2.5 

    print(f"\n🚀 Generando {num_samples} moléculas TADF guiadas...")
    print(f" Target Delta E_ST: {target_E_ST_value} eV | Escala de guía (gamma): {guidance_scale}")

    model.model.mlp_in_y[0] = torch.nn.Linear(13, model.model.mlp_in_y[0].out_features).to(device)
    model.model.mlp_in_X[0] = torch.nn.Linear(18, model.model.mlp_in_X[0].out_features).to(device)
    y = target_y
    guidance_scale = guidance_scale
    with torch.no_grad():
        # Muestreo con guía usando la MLP cargada
        generated_molecules = model.sample_batch(
            batch_id=0,
            batch_size=num_samples,
            keep_chain=1,
            number_chain_steps=10,
            save_final=0
        )

    print("\n✅ Generación completada. Evaluando validez de SMILES...")

    # ==============================================================================
    # 4. REVISIÓN Y VALIDACIÓN DE RESULTADOS
    # ==============================================================================
    dataset_infos = TADFDatasetInfos()
    decoder_info = None
    if 'dataset_infos' in locals() or 'dataset_infos' in globals():
        decoder_info = dataset_infos
    elif 'datamodule' in locals() and hasattr(datamodule, 'dataset_infos'):
        decoder_info = datamodule.dataset_infos
    elif hasattr(model, 'dataset_infos'):
        decoder_info = model.dataset_infos

    if decoder_info is None:
        raise ValueError("❌ No se encontró 'dataset_infos'. Asegúrate de definirlo o cargarlo desde tu datamodule.")

    valid_smiles = []

    print("\n Decodificando grafos a estructuras SMILES...\n")

    for i, mol in enumerate(generated_molecules):
        # Decodificar el grafo a cadena SMILES
        rdkitmol = decoder_info.to_molecule(mol[0],mol[1])
        
        # Verificar si es una molécula químicamente válida con RDKit
        if rdkitmol is not None and Chem.MolToSmiles(rdkitmol) is not None:
            smiles = Chem.MolToSmiles(rdkitmol)
            valid_smiles.append(smiles)
            print(f"Molécula {i+1} [VÁLIDA]: {smiles}")
        else:
            print(f"Molécula {i+1} [INVÁLIDA o inconclusa]")

    print(f"\n Validez Química Total: {len(valid_smiles)} / {num_samples} ({len(valid_smiles)/num_samples * 100:.1f}%)")
