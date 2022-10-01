from argparse import Namespace
import torch
import torch.nn as nn

from .mpn import MPN


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def forward(self, mols, BD_v_batch):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """
        atoms_v,bonds_v,t_atoms,t_bonds=self.encoder(mols, BD_v_batch)
        r_atoms = self.ffn_bonds(atoms_v)
        #r_atoms = self.ffn_bonds(atoms_v)
        #t_atoms=t_atoms[1:]
        t_all = t_atoms
        r_all = r_atoms
        v_all = atoms_v
        return r_all,t_all,v_all
