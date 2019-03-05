import torch
import torch.nn as nn
import numpy as np
from torchvision import models


class JointEmbedModel(nn.Module):
    def __init__(self, config, vlen, embeds=None):
        super(JointEmbedModel, self).__init__()
        self.vlen = vlen
        self.embedding_dim = config['embedding_dim']
        self.embed = nn.Embedding(self.vlen + 1, self.embedding_dim, padding_idx=self.vlen)
        if embeds is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embeds))

        self.lstm_hidden_size = 512
        self.lstm_num_layers = 2
        self.lstmq = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        
        self.lstm_transform = nn.Sequential(
            nn.Linear(2*self.lstm_num_layers*self.lstm_hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh())
        # self.lstm_transform.apply(init_weights)

        self.vggnet = models.vgg16(pretrained=True)
        self.vggnet.classifier = self.vggnet.classifier[:-1]
        self.vggnet.eval()
        for param in self.vggnet.parameters():
            param.requires_grad = False
        
        self.vggnet_transform = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh())
        # self.vggnet_transform.apply(init_weights)

        self.mlp = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.BatchNorm1d(1000),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, config['n_classes']))
        # self.mlp.apply(init_weights)


    def forward(self, images, q_idxs):
        ques_embedded = self.embed(q_idxs)

        lstm_out, (hn, cn) = self.lstmq(ques_embedded)
        hn = hn.permute(1, 0, 2)
        cn = cn.permute(1, 0, 2)
        hn = hn.contiguous().view(images.shape[0], self.lstm_hidden_size*self.lstm_num_layers)
        cn = cn.contiguous().view(images.shape[0], self.lstm_hidden_size*self.lstm_num_layers)

        ques_embedding = torch.cat((hn,cn), 1)
        ques_embedding_transformed = self.lstm_transform(ques_embedding)

        imgs_embedding = self.vggnet(images)
        imgs_embedding = nn.functional.normalize(imgs_embedding)
        imgs_embedding_transformed = self.vggnet_transform(imgs_embedding)

        joint_embedding = ques_embedding_transformed*imgs_embedding_transformed

        output = self.mlp(joint_embedding)

        return output

def init_weights(net):
    if type(net) == nn.Linear():
        torch_init.xavier_normal_(net.weight)