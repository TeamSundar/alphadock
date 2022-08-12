
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter_mean
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut, to_dense_batch
from torch_geometric.nn import MetaLayer, SplineConv, max_pool, GlobalAttention
from utils.distributions import *

def compute_cluster_batch_index(cluster, batch):
    max_prev_batch = 0
    for i in range(batch.max().item()+1):
        cluster[batch == i] += max_prev_batch
        max_prev_batch = cluster[batch == i].max().item() + 1
    return cluster

class NodeSampling(nn.Module):
    def __init__(self, nodes_per_graph):
        super(NodeSampling, self).__init__()
        
        self.num = nodes_per_graph
          
    def forward(self, x):
        if self.training:
            max_prev_batch = 0
            idx = []
            counts = torch.unique(x.batch, return_counts=True)[1]
            
            for i in range(x.batch.max().item()+1):
                idx.append(torch.randperm(counts[i])[:self.num] + max_prev_batch)
                max_prev_batch += counts[i]
            idx = torch.cat(idx)
            
            x.batch = x.batch[idx]
            x.pos = x.pos[idx]
            x.x = x.x[idx]
            
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.15):
        super(ResBlock, self).__init__()
        
        self.projectDown_node = nn.Linear(in_channels, in_channels//3)
        self.projectDown_edge = nn.Linear(in_channels, in_channels//3)
        self.bn1_node = nn.BatchNorm1d(in_channels//3, eps=10e-5, momentum=0.1)
        self.bn1_edge = nn.BatchNorm1d(in_channels//3, eps=10e-5, momentum=0.1)
        
        self.conv = MetaLayer(edge_model=EdgeModel(in_channels//3), node_model=NodeModel(in_channels//3), global_model=None)
                
        self.projectUp_node = nn.Linear(in_channels//3, in_channels)
        self.projectUp_edge = nn.Linear(in_channels//3, in_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2_node = nn.BatchNorm1d(in_channels, eps=10e-5, momentum=0.1)
        nn.init.zeros_(self.bn2_node.weight)
        self.bn2_edge = nn.BatchNorm1d(in_channels, eps=10e-5, momentum=0.1)
        nn.init.zeros_(self.bn2_edge.weight)
                
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        h_node = F.elu(self.bn1_node(self.projectDown_node(x)))
        h_edge = F.elu(self.bn1_edge(self.projectDown_edge(edge_attr)))
        h_node, h_edge, _ = self.conv(h_node, edge_index, h_edge, None, batch)
        
        h_node = self.dropout(self.bn2_node(self.projectUp_node(h_node)))
        data.x = F.elu(h_node + x)
        
        h_edge = self.dropout(self.bn2_edge(self.projectUp_edge(h_edge))) 
        data.edge_attr = F.elu(h_edge + edge_attr)
        
        return data

class EdgeModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(in_channels*3, in_channels), 
                                      nn.BatchNorm1d(in_channels, eps=10e-5, momentum=0.1),
                                      nn.ELU())

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        #print(out)
        return self.edge_mlp(out)
    
class NodeModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(nn.Linear(in_channels*2, in_channels),
                                        nn.BatchNorm1d(in_channels, eps=10e-5, momentum=0.1),
                                        nn.ELU())
        self.node_mlp_2 = nn.Sequential(nn.Linear(in_channels*2, in_channels),
                                        nn.BatchNorm1d(in_channels, eps=10e-5, momentum=0.1), 
                                        nn.ELU())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class TargetNet(nn.Module):
    def __init__(self, in_channels, edge_features=3, hidden_dim=128, residual_layers=20, dropout_rate=0.15):
        super(TargetNet, self).__init__()
        
        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        self.conv1 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv2 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv3 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        layers = [ResBlock(in_channels=hidden_dim, dropout_rate=dropout_rate) for i in range(residual_layers)] 
        self.resnet = nn.Sequential(*layers)
        
    def forward(self, data):
        data.edge_attr = None
        data = T.Cartesian(norm=False, max_value=None, cat=False)(data)

        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr)
        data.x, data.edge_attr, _ = self.conv1(data.x, data.edge_index, data.edge_attr, None, data.batch)
        #print(data.x, data.edge_attr)
        data.x, data.edge_attr, _ = self.conv2(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data.x, data.edge_attr, _ = self.conv3(data.x, data.edge_index, data.edge_attr, None, data.batch)
        
        data = self.resnet(data)
        
        return data
    
class LigandNet(nn.Module):
    def __init__(self, in_channels, edge_features=6, hidden_dim=128, residual_layers=20, dropout_rate=0.15):
        super(LigandNet, self).__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        self.conv1 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv2 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv3 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        layers = [ResBlock(in_channels=hidden_dim, dropout_rate=dropout_rate) for i in range(residual_layers)] 
        self.resnet = nn.Sequential(*layers)
                        
    def forward(self, data):
        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr)
        data.x, data.edge_attr, _ = self.conv1(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data.x, data.edge_attr, _ = self.conv2(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data.x, data.edge_attr, _ = self.conv3(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data = self.resnet(data)
        #print(data.x)
                
        return data

class AlphaDock(nn.Module):
    def __init__(self, ligand_model, target_model, hidden_dim, n_gaussians, dropout_rate=0.15, 
                 nodes_per_target=None, dist_threhold=1000):
        super(AlphaDock, self).__init__()
        
        self.ligand_model = ligand_model
        self.target_model = target_model
        if nodes_per_target:
            self.node_sampling = NodeSampling(nodes_per_graph=nodes_per_target)
        else :
            self.node_sampling = None
        self.MLP = nn.Sequential(nn.Linear(256, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim, eps=10e-5, momentum=0.1),  
                                 nn.ELU(), nn.Dropout(p=dropout_rate)) 
  
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
        self.atom_types = nn.Linear(128, 28)
        self.bond_types = nn.Linear(256, 6)
                
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dist_threhold = dist_threhold
    
    def forward(self, data_ligand, data_target, y=None):
        h_l = self.ligand_model(data_ligand)
        h_t = self.target_model(data_target)

        if self.node_sampling:
            h_t = self.node_sampling(h_t)
    
        h_l_x, l_mask = to_dense_batch(h_l.x, h_l.batch, fill_value=0)
        h_t_x, t_mask = to_dense_batch(h_t.x, h_t.batch, fill_value=0)
        h_l_pos, _ = to_dense_batch(h_l.pos, h_l.batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(h_t.pos, h_t.batch, fill_value=0)

        #print(h_l_x)
        
        assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
        
        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]

        h_t_x = h_t_x.unsqueeze(-3)
        h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]

        C = torch.cat((h_l_x, h_t_x), -1)
        #print(C)
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
        self.C = C = C[C_mask]
        C = self.MLP(C)

        # Get batch indexes for ligand-target combined features
        C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_t)[C_mask].to(self.device)
        
        #print(C)

        # Outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C))+1.1
        mu = F.elu(self.z_mu(C))+1
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos)[C_mask]
        atom_types = self.atom_types(h_l.x)
        bond_types = self.bond_types(torch.cat([h_l.x[h_l.edge_index[0]], h_l.x[h_l.edge_index[1]]], axis=1))
        
        return pi, sigma, mu, dist.unsqueeze(1).detach(), atom_types, bond_types, C_batch

    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()

        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        return dists**0.5

    
def mdn_loss_fn(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -torch.logsumexp(torch.log(pi) + loglik, dim=1)
    return loss

import copy
from utils.data import *
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem

def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)
           
def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])
    
def GetTransformationMatrix(transformations):
    x, y, z, disp_x, disp_y, disp_z = transformations
    transMat= np.array([[np.cos(z)*np.cos(y), (np.cos(z)*np.sin(y)*np.sin(x))-(np.sin(z)*np.cos(x)), (np.cos(z)*np.sin(y)*np.cos(x))+(np.sin(z)*np.sin(x)), disp_x],
                        [np.sin(z)*np.cos(y), (np.sin(z)*np.sin(y)*np.sin(x))+(np.cos(z)*np.cos(x)), (np.sin(z)*np.sin(y)*np.cos(x))-(np.cos(z)*np.sin(x)), disp_y],
                        [-np.sin(y),           np.cos(y)*np.sin(x),                                   np.cos(y)*np.cos(x),                                  disp_z],
                        [0,                    0,                                                     0,                                                    1]], dtype=np.double)
    return transMat

def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi)
    prob = logprob.exp().sum(1)
    return prob
    
def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.copy(mol)
    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[6+r]) for r in range(len(rotable_bonds))]
    # apply transformation matrix
    rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))
    return opt_mol

def get_torsions(mol_list):
    atom_counter=0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                        or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                        or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                        continue    
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append((idx4+atom_counter, idx3+atom_counter, idx2+atom_counter, idx1+atom_counter))
                        break
                    else:
                        torsionList.append((idx1+atom_counter, idx2+atom_counter, idx3+atom_counter, idx4+atom_counter))
                        break
                break
                    
        atom_counter += m.GetNumAtoms()
    return torsionList

def get_random_conformation(mol, rotable_bonds=None, seed=None):
    if isinstance(mol, Chem.Mol):
        # Check if ligand it has 3D coordinates, otherwise generate them
        try:
            mol.GetConformer()
        except:
                mol=Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
    else:
        raise Exception('mol should be an RDKIT molecule')
    if seed:
            np.random.seed(seed)
    if rotable_bonds is None:        
        rotable_bonds = get_torsions([mol])
    new_conf = apply_changes(mol, np.random.rand(len(rotable_bonds, )+6)*10, rotable_bonds)
    Chem.rdMolTransforms.CanonicalizeConformer(new_conf.GetConformer())
    return new_conf

def compute_euclidean_distances_matrix(X, Y):
    # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # (X-Y)^2 = X^2 + Y^2 -2XY
    X = X.double()
    Y = Y.double()
    dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
    return dists**0.5

def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.copy(mol)
    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[6+r]) for r in range(len(rotable_bonds))]
    # apply transformation matrix
    rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))
    return opt_mol

class optimze_conformation():
    def __init__(self, mol, target_coords, n_particles, pi, mu, sigma, save_molecules=False, dist_threshold=1000, seed=None):
        super(optimze_conformation, self).__init__()
        if seed:
            np.random.seed(seed)
            
        self.opt_mols = []
        self.n_particles = n_particles
        self.rotable_bonds = get_torsions([mol])
        self.save_molecules = save_molecules
        self.dist_threshold = dist_threshold
        self.mol =get_random_conformation(mol, rotable_bonds=self.rotable_bonds, seed=seed)
        
        self.targetCoords = torch.stack([target_coords for _ in range(n_particles)]).double()
        self.pi = torch.cat([pi for _ in range(n_particles)], axis=0)
        self.sigma = torch.cat([sigma for _ in range(n_particles)], axis=0)
        self.mu = torch.cat([mu for _ in range(n_particles)], axis=0)
        self.noHidx = [idx for idx in range(self.mol.GetNumAtoms()) if self.mol.GetAtomWithIdx(idx).GetAtomicNum()is not 1]
        
    def score_conformation(self, values):
        """
        Parameters
        ----------
        values : numpy.ndarray
            set of inputs of shape :code:`(n_particles, dimensions)`
        Returns
        -------
        numpy.ndarray
            computed cost of size :code:`(n_particles, )`
        """
        if len(values.shape) < 2: values = np.expand_dims(values, axis=0)
        mols = [copy.copy(self.mol) for _ in range(self.n_particles)]
        
        # Apply changes to molecules
        # apply rotations
        [SetDihedral(mols[m].GetConformer(), self.rotable_bonds[r], values[m, 6+r]) for r in range(len(self.rotable_bonds)) for m in range(self.n_particles)]
        
        # apply transformation matrix
        [rdMolTransforms.TransformConformer(mols[m].GetConformer(), GetTransformationMatrix(values[m, :6])) for m in range(self.n_particles)]
        
        # Calcualte distances between ligand conformation and target
        ligCoords_list = [torch.tensor(m.GetConformer().GetPositions()[self.noHidx]) for m in mols]
        ligCoords = torch.stack(ligCoords_list).double()
        dist = torch.cdist(ligCoords, self.targetCoords, 2).flatten().unsqueeze(1)
        
        # Calculate probability for each ligand-target pair
        prob = calculate_probablity(self.pi, self.sigma, self.mu, dist)
        prob[torch.where(dist > self.dist_threshold)[0]] = 0.
        
        # Reshape and sum probabilities
        prob = prob.reshape(self.n_particles, -1).sum(1)
        if self.save_molecules: self.opt_mols.append(mols[torch.argmax(prob)])
        
        # Delete useless tensors to free memory
        del ligCoords_list, ligCoords, dist, mols
        
        return -prob.detach().numpy()