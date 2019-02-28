import os
from time import gmtime, strftime
from torch.autograd import Variable
import torch
import random


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
