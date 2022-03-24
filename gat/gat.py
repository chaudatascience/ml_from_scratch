## paper Graph Attention Networks - https://arxiv.org/abs/1710.10903
import enum

import torch
from torch import nn


class GAT(nn.Module):
    def __init__(self, node_dim: int, out_dim: int, num_heads: int, is_final_layer: bool, dropout: float):
        """
        :param node_dim: F, the number of features in each node
        :param out_dim: F', new node feature dimension
        :param num_heads: K, number of heads in multi-head attention
        :param is_final_layer: True if the final layer, False otherwise
        :param dropout: dropout prob for dropout layer
        """
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.W = nn.Linear(node_dim, out_dim * num_heads)  # Shared linear transformations

        self.is_final_layer = is_final_layer  # used for choosing equation (5) (if False) or equation (6) (if True)

        ## Attention `a` in equation 3. Instead of using a vector `a` with a shape of 2F',
        # we split `a` into 2 halves: `a_left` and `a_right`. This is only for implementation purpose
        # i.e., easier to stack vectors to use matrix multiplication.
        self.a_left = nn.Parameter(torch.Tensor(1, num_heads, out_dim))  # left part of the attention mechanism
        self.a_right = nn.Parameter(torch.Tensor(1, num_heads, out_dim))  # the right part

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        self.x_encoded = None  # cache for calculating h_prime in equation (4), (5), (6)
        self._init_params()  # initialize some parameters

    def _init_params(self):
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)
        nn.init.xavier_uniform_(self.W.weight)

    def _compute_energy(self, x):
        """
        Compute matrix e containing e_ij in equation (1)
        :param x: input, shape of (num_nodes N, node_dim F)
        :return: e: shape of (num_heads K, num_nodes N, num_nodes N)
        """
        x_encoded = self.W(x).reshape(-1, self.num_heads, self.out_dim)  # shape: (N, K, F')
        x_encoded = self.dropout(x_encoded)  # (N, K, F')
        self.x_encoded = x_encoded  # cache x_encoded for later use
        h_left = torch.einsum("nkf,ikf->kni", [x_encoded, self.a_left])  # (K, 1, N)
        h_right = torch.einsum("nkf,ikf->kin", [x_encoded, self.a_right])  # (K, N, 1)
        e = h_left + h_right  # shape (K,N,N)
        return e

    def _compute_attention(self, e, mask):
        """
         Compute matrix a containing a_ij in equation (3)
         :param e: energy matrix containing e_ij in equation (1), shape  of (num_heads K, num_nodes N, num_nodes N)
         :param mask: adjacency matrix to mask out the nodes that are not neighbors, shape  (1, num_nodes N, num_nodes N)
         :return: attention weights, shape  of (num_heads K, num_nodes N, num_nodes N)
         """
        e = self.leaky_relu(e)
        e = e.masked_fill(mask == torch.tensor(False), -float("inf"))
        a = self.softmax(e)
        return a

    def _compute_final_node_features(self, a):
        """
        Compute final output features `h_prime` in equation (4)
        :param a: attention matrix in equation (3), shape (num_heads K, num_nodes N, num_nodes N)
        :return: (num_nodes N, num_heads K, out_dim F')
        """
        h_prime = torch.einsum("knm,mkf->nkf", [a, self.x_encoded])  # shape (num_nodes N, num_heads K, out_dim F')
        h_prime = self.elu(h_prime)
        return h_prime

    def _concat_multi_head_features(self, node_features):
        """
        concatenate features from all attention heads, equation (5)
        :param node_features: updated node features from equation (4), shape  (num_nodes N, num_heads K, out_dim F')
        :return: just need to reshape to 2-D: (num_nodes N, num_heads K * out_dim F')
        """
        output = self.elu(node_features).reshape(-1, self.num_heads * self.out_dim)
        return output

    def _average_multi_head_features(self, node_features):
        """
        For the final layer, we employ averaging as described in equation (6)
        :param node_features: updated node features from equation (4), shape  (num_nodes N, num_heads K, out_dim F')
        :return: (num_nodes N, out_dim F'), for the last layer, we use F'=C (#classes)
        """
        output = torch.mean(node_features, dim=1) # No need softmax() here as it will be used in cross entropy loss
        return output

    def forward(self, data):
        """
        The flow of GAT network, following the description in the paper
        :param data: tuple (input, mask), shape of (num_nodes N, node_dim F) and (num_nodes N, num_nodes N) respectively
        :return: output, mask
        """
        ## unpack data
        # x: node features, shape (num_nodes N, node_dim F)
        # mask: bool 2D-tensor,  mask = adjacency_matrix + identity_matrix; shape (num_nodes, num_nodes)
        x, mask = data

        ## apply dropout to the input
        x = self.dropout(x)

        ## compute energy matrix `e` containing e_ij in the equation (1), page 3
        e = self._compute_energy(x)  # shape of e: (num_heads K, num_nodes N, num_nodes N)

        ## compute attention matrix `a` containing a_ij in the equation (3)
        a = self._compute_attention(e, mask)   # shape (num_heads K, num_nodes N, num_nodes N)

        ## compute final output features `output_features` in equation (4)
        final_features = self._compute_final_node_features(a)  # (num_nodes N, num_heads K, out_dim F')

        # collect features from multi heads
        if self.is_final_layer:  # last layer
            output = self._average_multi_head_features(final_features)  # eq (6), (num_nodes N, out_dim F')
        else:  # intermediate layer
            output = self._concat_multi_head_features(final_features)  #eq (5), (num_nodes N, num_heads K * out_dim F')

        return output, mask  ## in addition to output, we return mask for the next layer
