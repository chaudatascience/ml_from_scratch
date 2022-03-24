A PyTorch implementation of [Graph Attention Networks (GAT)](https://arxiv.org/abs/1710.10903) [ICLR 2018]

```
python main.py
```

```
#params in GAT NET: 92,373
odict_keys(['gat_net.0.a_left', 'gat_net.0.a_right', 'gat_net.0.W.weight', 'gat_net.0.W.bias', 'gat_net.1.a_left', 'gat_net.1.a_right', 'gat_net.1.W.weight', 'gat_net.1.W.bias'])
gat_net.0.a_left torch.Size([1, 8, 8])
gat_net.0.a_right torch.Size([1, 8, 8])
gat_net.0.W.weight torch.Size([64, 1433])
gat_net.0.W.bias torch.Size([64])
gat_net.1.a_left torch.Size([1, 1, 7])
gat_net.1.a_right torch.Size([1, 1, 7])
gat_net.1.W.weight torch.Size([7, 64])
gat_net.1.W.bias torch.Size([7])
epoch: 1 | time elapsed =  3.87[s] | train_loss: 2.00 | train_acc: 0.13 | val_acc: 0.25
epoch: 2 | time elapsed =  5.98[s] | train_loss: 1.94 | train_acc: 0.18 | val_acc: 0.57
epoch: 3 | time elapsed =  7.96[s] | train_loss: 1.83 | train_acc: 0.34 | val_acc: 0.66
epoch: 4 | time elapsed =  9.94[s] | train_loss: 1.77 | train_acc: 0.39 | val_acc: 0.72
epoch: 5 | time elapsed =  12.09[s] | train_loss: 1.67 | train_acc: 0.49 | val_acc: 0.74
epoch: 6 | time elapsed =  14.16[s] | train_loss: 1.56 | train_acc: 0.54 | val_acc: 0.75
epoch: 7 | time elapsed =  16.12[s] | train_loss: 1.49 | train_acc: 0.60 | val_acc: 0.76
epoch: 8 | time elapsed =  18.08[s] | train_loss: 1.44 | train_acc: 0.59 | val_acc: 0.75
epoch: 9 | time elapsed =  20.10[s] | train_loss: 1.39 | train_acc: 0.59 | val_acc: 0.76
epoch: 10 | time elapsed =  22.21[s] | train_loss: 1.31 | train_acc: 0.59 | val_acc: 0.76
epoch: 11 | time elapsed =  24.26[s] | train_loss: 1.30 | train_acc: 0.61 | val_acc: 0.76
epoch: 12 | time elapsed =  26.26[s] | train_loss: 1.25 | train_acc: 0.63 | val_acc: 0.77
epoch: 13 | time elapsed =  28.46[s] | train_loss: 1.14 | train_acc: 0.68 | val_acc: 0.77
epoch: 14 | time elapsed =  32.34[s] | train_loss: 1.09 | train_acc: 0.68 | val_acc: 0.77
epoch: 15 | time elapsed =  35.84[s] | train_loss: 1.16 | train_acc: 0.62 | val_acc: 0.77
epoch: 16 | time elapsed =  38.35[s] | train_loss: 1.06 | train_acc: 0.69 | val_acc: 0.77
epoch: 17 | time elapsed =  40.71[s] | train_loss: 1.08 | train_acc: 0.69 | val_acc: 0.76
epoch: 18 | time elapsed =  43.03[s] | train_loss: 1.00 | train_acc: 0.73 | val_acc: 0.77
epoch: 19 | time elapsed =  45.17[s] | train_loss: 0.97 | train_acc: 0.69 | val_acc: 0.77
epoch: 20 | time elapsed =  47.28[s] | train_loss: 0.93 | train_acc: 0.76 | val_acc: 0.77
epoch: 21 | time elapsed =  49.44[s] | train_loss: 0.92 | train_acc: 0.75 | val_acc: 0.77
epoch: 22 | time elapsed =  51.55[s] | train_loss: 0.85 | train_acc: 0.79 | val_acc: 0.78
epoch: 23 | time elapsed =  53.54[s] | train_loss: 0.95 | train_acc: 0.74 | val_acc: 0.78
epoch: 24 | time elapsed =  55.54[s] | train_loss: 0.86 | train_acc: 0.74 | val_acc: 0.78
epoch: 25 | time elapsed =  57.55[s] | train_loss: 0.87 | train_acc: 0.71 | val_acc: 0.78
epoch: 26 | time elapsed =  59.64[s] | train_loss: 0.92 | train_acc: 0.75 | val_acc: 0.79
epoch: 27 | time elapsed =  61.66[s] | train_loss: 0.83 | train_acc: 0.76 | val_acc: 0.79
epoch: 28 | time elapsed =  63.73[s] | train_loss: 0.89 | train_acc: 0.71 | val_acc: 0.78
epoch: 29 | time elapsed =  65.78[s] | train_loss: 0.85 | train_acc: 0.74 | val_acc: 0.79
epoch: 30 | time elapsed =  67.72[s] | train_loss: 0.77 | train_acc: 0.81 | val_acc: 0.79
epoch: 31 | time elapsed =  69.79[s] | train_loss: 0.76 | train_acc: 0.73 | val_acc: 0.78
epoch: 32 | time elapsed =  71.81[s] | train_loss: 0.79 | train_acc: 0.76 | val_acc: 0.78
epoch: 33 | time elapsed =  73.82[s] | train_loss: 0.74 | train_acc: 0.75 | val_acc: 0.78
epoch: 34 | time elapsed =  75.77[s] | train_loss: 0.73 | train_acc: 0.79 | val_acc: 0.78
epoch: 35 | time elapsed =  77.81[s] | train_loss: 0.83 | train_acc: 0.73 | val_acc: 0.79
epoch: 36 | time elapsed =  79.76[s] | train_loss: 0.70 | train_acc: 0.78 | val_acc: 0.79
epoch: 37 | time elapsed =  81.69[s] | train_loss: 0.77 | train_acc: 0.76 | val_acc: 0.79
epoch: 38 | time elapsed =  84.38[s] | train_loss: 0.77 | train_acc: 0.74 | val_acc: 0.79
epoch: 39 | time elapsed =  86.65[s] | train_loss: 0.74 | train_acc: 0.78 | val_acc: 0.79
epoch: 40 | time elapsed =  88.74[s] | train_loss: 0.71 | train_acc: 0.77 | val_acc: 0.79
epoch: 41 | time elapsed =  90.71[s] | train_loss: 0.74 | train_acc: 0.78 | val_acc: 0.80
epoch: 42 | time elapsed =  92.65[s] | train_loss: 0.69 | train_acc: 0.76 | val_acc: 0.79
epoch: 43 | time elapsed =  94.63[s] | train_loss: 0.68 | train_acc: 0.77 | val_acc: 0.79
epoch: 44 | time elapsed =  96.57[s] | train_loss: 0.65 | train_acc: 0.79 | val_acc: 0.79
epoch: 45 | time elapsed =  98.52[s] | train_loss: 0.77 | train_acc: 0.73 | val_acc: 0.79
epoch: 46 | time elapsed =  100.43[s] | train_loss: 0.64 | train_acc: 0.82 | val_acc: 0.79
epoch: 47 | time elapsed =  102.40[s] | train_loss: 0.66 | train_acc: 0.81 | val_acc: 0.79
epoch: 48 | time elapsed =  104.41[s] | train_loss: 0.62 | train_acc: 0.79 | val_acc: 0.79
epoch: 49 | time elapsed =  107.61[s] | train_loss: 0.58 | train_acc: 0.83 | val_acc: 0.79
epoch: 50 | time elapsed =  110.87[s] | train_loss: 0.65 | train_acc: 0.78 | val_acc: 0.79
epoch: 51 | time elapsed =  113.96[s] | train_loss: 0.57 | train_acc: 0.81 | val_acc: 0.79
epoch: 52 | time elapsed =  116.04[s] | train_loss: 0.54 | train_acc: 0.86 | val_acc: 0.79
epoch: 53 | time elapsed =  118.18[s] | train_loss: 0.63 | train_acc: 0.79 | val_acc: 0.79
epoch: 54 | time elapsed =  120.26[s] | train_loss: 0.70 | train_acc: 0.76 | val_acc: 0.79
epoch: 55 | time elapsed =  122.23[s] | train_loss: 0.55 | train_acc: 0.82 | val_acc: 0.78
epoch: 56 | time elapsed =  124.23[s] | train_loss: 0.56 | train_acc: 0.86 | val_acc: 0.79
epoch: 57 | time elapsed =  126.22[s] | train_loss: 0.67 | train_acc: 0.80 | val_acc: 0.79
epoch: 58 | time elapsed =  128.24[s] | train_loss: 0.54 | train_acc: 0.85 | val_acc: 0.79
epoch: 59 | time elapsed =  130.24[s] | train_loss: 0.58 | train_acc: 0.86 | val_acc: 0.79
epoch: 60 | time elapsed =  132.26[s] | train_loss: 0.60 | train_acc: 0.81 | val_acc: 0.79
epoch: 61 | time elapsed =  134.32[s] | train_loss: 0.70 | train_acc: 0.78 | val_acc: 0.79
epoch: 62 | time elapsed =  136.62[s] | train_loss: 0.62 | train_acc: 0.79 | val_acc: 0.79
epoch: 63 | time elapsed =  140.52[s] | train_loss: 0.56 | train_acc: 0.85 | val_acc: 0.78
epoch: 64 | time elapsed =  142.59[s] | train_loss: 0.58 | train_acc: 0.84 | val_acc: 0.78
epoch: 65 | time elapsed =  145.91[s] | train_loss: 0.69 | train_acc: 0.79 | val_acc: 0.77
epoch: 66 | time elapsed =  147.91[s] | train_loss: 0.44 | train_acc: 0.87 | val_acc: 0.77
epoch: 67 | time elapsed =  149.94[s] | train_loss: 0.62 | train_acc: 0.79 | val_acc: 0.77
epoch: 68 | time elapsed =  152.03[s] | train_loss: 0.53 | train_acc: 0.81 | val_acc: 0.77
epoch: 69 | time elapsed =  154.16[s] | train_loss: 0.57 | train_acc: 0.79 | val_acc: 0.78
epoch: 70 | time elapsed =  156.21[s] | train_loss: 0.44 | train_acc: 0.86 | val_acc: 0.78
epoch: 71 | time elapsed =  158.24[s] | train_loss: 0.63 | train_acc: 0.77 | val_acc: 0.78
epoch: 72 | time elapsed =  160.26[s] | train_loss: 0.45 | train_acc: 0.86 | val_acc: 0.78
epoch: 73 | time elapsed =  162.27[s] | train_loss: 0.49 | train_acc: 0.85 | val_acc: 0.78
epoch: 74 | time elapsed =  164.19[s] | train_loss: 0.52 | train_acc: 0.81 | val_acc: 0.79
epoch: 75 | time elapsed =  166.15[s] | train_loss: 0.63 | train_acc: 0.79 | val_acc: 0.80
epoch: 76 | time elapsed =  168.11[s] | train_loss: 0.52 | train_acc: 0.81 | val_acc: 0.80
epoch: 77 | time elapsed =  170.11[s] | train_loss: 0.51 | train_acc: 0.86 | val_acc: 0.80
epoch: 78 | time elapsed =  172.11[s] | train_loss: 0.53 | train_acc: 0.84 | val_acc: 0.80
epoch: 79 | time elapsed =  174.24[s] | train_loss: 0.58 | train_acc: 0.81 | val_acc: 0.80
epoch: 80 | time elapsed =  176.35[s] | train_loss: 0.48 | train_acc: 0.84 | val_acc: 0.80
epoch: 81 | time elapsed =  178.38[s] | train_loss: 0.53 | train_acc: 0.81 | val_acc: 0.80
epoch: 82 | time elapsed =  180.44[s] | train_loss: 0.49 | train_acc: 0.84 | val_acc: 0.80
epoch: 83 | time elapsed =  182.70[s] | train_loss: 0.44 | train_acc: 0.85 | val_acc: 0.80
...
```
