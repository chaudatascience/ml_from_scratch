import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from utils import show
from vq_vae import VQVAE

def train():
    start = time.time()
    batch_size = 128
    num_training_updates = 25000

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = 64
    num_embeddings = 512

    commitment_cost = 0.25
    use_moving_avg = True

    decay = 0.99

    learning_rate = 1e-3
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                       ]))

    data_variance = np.var(training_data.data / 255.0)

    training_loader = DataLoader(training_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True)

    validation_loader = DataLoader(validation_data,
                                   batch_size=32,
                                   shuffle=True,
                                   pin_memory=True)

    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay, use_moving_avg=use_moving_avg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.train()
    train_res_recon_error = []

    for i in range(num_training_updates):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print()

    model.eval()

    (valid_originals, _) = next(iter(validation_loader))
    valid_originals = valid_originals.to(device)
    print(valid_originals.shape)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize = model.vq_vae_layer(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    (train_originals, _) = next(iter(training_loader))
    train_originals = train_originals.to(device)
    _, train_reconstructions = model.vq_vae_layer(train_originals)

    from pathlib import Path
    folder = f"{str(time.time())}"
    Path(f"logs/{folder}").mkdir(parents=True, exist_ok=True)

    show(make_grid(valid_reconstructions.cpu().data) + 0.5, f"logs/{folder}", "valid_reconstructions")
    show(make_grid(valid_originals.cpu() + 0.5), f"logs/{folder}", "valid_original")

    # train_res_recon_error = savgol_filter(train_res_recon_error, 201, 7)
    f = plt.figure(figsize=(16, 8))
    ax = f.add_subplot(1, 2, 1)
    ax.plot(train_res_recon_error)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')
    plt.savefig(f'logs/{folder}/train_res_recon_error.png')

    end= time.time()
    print(f"running time: {round((end-start)/60,2)} mins")


