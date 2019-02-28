import torch
import torch.nn as nn
import numpy as np


class JointEmbedModel(nn.Module):
    def __init__(self, config, vlen, embeds=None):
        super(JointEmbedModel, self).__init__()
        self.vlen = vlen
        self.embedding_dim = config['embedding_dim']
        self.embed = nn.Embedding(self.vlen + 1, self.embedding_dim, padding_idx=self.vlen)
        if embeds is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embeds))

    def forward(self, images, q_idxs):
        return torch.ones(images.shape[0])
        # TODO: Complete forward pass of the model
