import numpy as np
from matplotlib import pyplot as plt


def show(img, folder, name):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(f'{folder}/{name}.png')