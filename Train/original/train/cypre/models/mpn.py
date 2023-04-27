from argparse import Namespace
from typing import List, Union

import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from .features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from .nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.attention = args.attention
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = torch.zeros(self.hidden_size).cuda()

        # Input
        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)


        w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        self.W_U = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.batch1=nn.BatchNorm1d(self.hidden_size)

        # attention weights
        if self.attention:
            self.A = nn.Sequential(
                    nn.Linear(self.hidden_size + self.bond_fdim, self.hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.Linear(self.hidden_size, 1),
                    nn.Softmax(dim=1)
            )

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.args.cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

        # Input

        input = self.W_i(f_bonds)  # num_bonds x hidden_size

        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):

            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden

            message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        # Keep undirection of Bonds
        d_1=list(range(0,message.shape[0],2))
        d_2=[0]
        d_2.extend(list(range(1,message.shape[0],2)))
        message_und=(message[d_1]+message[d_2])/2
        # message_und=self.W_U(message_und)
        bonds_v = message_und

        a2x = a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
        atoms_v = atom_hiddens

        atoms_v = self.batch1(atoms_v[1:])
        bonds_v = self.batch1(bonds_v[1:])

        return atoms_v, bonds_v


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim() + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                BD_v_batch: List[List[str]],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, BD_v_batch)

        t_atoms,t_bonds=batch.get_true()
        t_atoms=torch.FloatTensor(np.array(t_atoms))
        t_bonds=torch.FloatTensor(np.array(t_bonds))

        atoms_v,bonds_v = self.encoder.forward(batch, features_batch)

        return atoms_v,bonds_v,t_atoms,t_bonds
