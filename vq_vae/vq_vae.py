import torch
from torch import nn

from decoder import Decoder
from encoder import Encoder
from vq_vae_layer import VQVAELayer


class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 num_embeddings, embedding_dim, beta, decay=0, eps = 1e-5,
                 use_moving_avg: bool=True):
        super().__init__()

        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)
        self.vq_vae_layer = VQVAELayer(num_embeddings, embedding_dim, beta, decay, eps, use_moving_avg)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)

    def forward(self, x: torch.Tensor):
        """
        steps: x -> encode -> vector quantization -> decode -> output
        :param x: shape of (B, C, H, W)
        :return:
        """

        ## encode
        z_e = self._encoder(x)

        ## pre vq ## TODO ?
        z_e = self._pre_vq_conv(z_e)

        ## vector quantization
        vq_loss, z_q = self.vq_vae_layer(z_e)  # codebook loss and commitment loss

        ## decode
        x_recon = self._decoder(z_q)

        ## Note: loss consists of reconstruction loss, codebook loss, and commitment loss
        return vq_loss, x_recon
