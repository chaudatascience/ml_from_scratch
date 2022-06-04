import torch
from torch import nn
from torch.nn import functional as F


class VQVAELayer(nn.Module):
    def __init__(self, n_embs, emb_dim, beta, decay, eps, use_moving_avg):
        super().__init__()
        self.n_embs = n_embs
        self.emb_dim = emb_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.use_moving_avg = use_moving_avg
        self.emb = nn.Embedding(n_embs, emb_dim)

        ## exponential moving average to update emb (instead of codebook loss) - Appendix
        ## todo: experiment on this
        if use_moving_avg:
            self.register_buffer("ema_sum", torch.Tensor(n_embs, emb_dim))
            self.register_buffer("cluster_size", torch.zeros(n_embs))

        self._initialize()

    def _initialize(self):
        self.emb.weight.data.normal_()  # TODO: try with xavier
        if self.use_moving_avg:
            self.ema_sum.data.normal_()

    @staticmethod
    def _moving_avg(x, new_x, decay):
        return x * decay + new_x * (1 - decay)

    @staticmethod
    def _laplace_smooth(x, eps, num_x, total):
        """ https://www.youtube.com/watch?v=gCI-ZC7irbY """
        return (x + eps) / (num_x + total * eps) * num_x

    def forward(self, x: torch.Tensor):
        """

        :param x: (B,C=emb_dim,H,W)
        :return:
        """
        ## distance between x and codebook `self.emb`:
        z_e = x.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)

        dist = torch.cdist(z_e.view(-1, self.emb_dim), self.emb.weight.data)  # (B*H*W,n_embs)
        nearest_indices = dist.argmin(dim=-1)  # (B*H*W)
        nearest_one_hot = F.one_hot(nearest_indices, num_classes=self.n_embs)  # (B*H*W, n_embs)

        ## vq_output: get e (i.e., z_q) by snapping inputs z_e to the nearest embedding self.emb
        z_q = nearest_one_hot.float() @ self.emb.weight  # (B*H*W, emb_dim)
        z_q = z_q.reshape(*z_e.shape[:-1], -1).permute(0, 3, 1, 2).contiguous()  # (B, emb_dim, H, W)

        #### losses
        ## The third loss term, force encoder to generate outputs near the codebook
        commit_loss = F.mse_loss(x, z_q.detach())

        if self.use_moving_avg:
            if self.training:
                self.cluster_size = self._moving_avg(self.cluster_size, nearest_one_hot.sum(dim=0), self.decay)

                ## laplace to smooth cluster_size:
                self.cluster_size = self._laplace_smooth(self.cluster_size, self.eps, self.cluster_size.sum(),
                                                         self.n_embs)

                sum_cluster_in_batch = nearest_one_hot.T.float() @ z_e.reshape(-1, self.emb_dim)  # (n_embs, emb_dim)
                self.ema_sum = self._moving_avg(self.ema_sum, sum_cluster_in_batch, self.decay).detach()  # (n_embs, emb_dim)

                ## update codebook
                self.emb.weight.data = self.ema_sum / self.cluster_size.unsqueeze(-1)  # (n_embs, emb_dim)
                assert self.emb.weight.data.shape == (self.n_embs, self.emb_dim)
            vq_loss = self.beta * commit_loss

        else:  # the second loss term in the paper
            codebook_loss = F.mse_loss(x.detach(), z_q)  # second loss: Move embedding closer to encoder's output
            vq_loss = codebook_loss + self.beta * commit_loss

        ## gradient trick: copy gradient from z_q to z_e
        z_q = x + (z_q - x).detach()

        return vq_loss, z_q
