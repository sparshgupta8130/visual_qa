import os
from time import gmtime, strftime
from torch.autograd import Variable
import torch
import torch.nn as nn
import random
#import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def create_directory(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_time_prefix():
    ts = strftime("%Y-%m-%d__%Hh%Mm%Ss_", gmtime())
    return ts


def get_indices(X, vocab, mlen):
    maxlen = 0
    for i in range(len(X)):
        t = X[i].split()
        maxlen = max(maxlen, len(t))
    maxlen = min(maxlen, mlen)

    idxs = []
    for i in range(len(X)):
        v = []
        t = X[i].split()
        for j in range(maxlen):
            if j < len(t):
                if t[j] in vocab:
                    v.append(vocab[t[j]])
                else:
                    v.append(len(vocab))
            else:
                v.append(len(vocab))
        idxs.append(v)

    inds = Variable(torch.LongTensor(idxs))
    return inds

"""
def plot_att(image, alpha, fname):
    plt.imshow(image)
    al = skimage.transform.pyramid_expand(alpha, upscale=32)
    plt.imshow(al, alpha=0.85)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
    plt.savefig(fname)


def visualize_att(model, config, dataloader):
    vis_path = './visualizations/'
    create_directory(vis_path)
    model_name = config['name']
    ts = get_time_prefix()
    vis_fname = vis_path + ts + model_name

    device = torch.device("cuda:0" if config['GPU'] is True else "cpu")
    model.to(device)
    if config['GPU'] is True:
        model = nn.DataParallel(model, device_ids=config['GPU_Ids'])

    model.eval()

    ctr = 0
    for ims, questions, answers in dataloader:
        images = Variable(ims).to(device)
        q_idxs = Variable(questions).to(device)
        out = model(images, q_idxs)
        alpha = model.attention.alpha
        alpha = alpha.view(ims.shape[0], 2, -1)
        alpha1 = alpha[:, 0, :]
        alpha2 = alpha[:, 1, :]
        if config['GPU'] is True:
            alpha1 = alpha1.cpu().data.numpy()
            alpha2 = alpha2.cpu().data.numpy()
        else:
            alpha1 = alpha1.data.numpy()
            alpha2 = alpha2.data.numpy()

        for i in range(ims.shape[0]):
            al1 = alpha1[i, :]
            al2 = alpha2[i, :]
            plot_att(ims[i].numpy(), al1, vis_fname + str(ctr) + '_glmp1.png')
            plot_att(ims[i].numpy(), al2, vis_fname + str(ctr) + '_glmp2.png')
            ctr += 1
"""

