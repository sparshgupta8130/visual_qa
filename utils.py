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


def get_indices():
    pass
    # TODO: Convert input string to vocabulary indices
