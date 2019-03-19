import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.nn.functional as F


class JointEmbedModel(nn.Module):
	def __init__(self, config, vlen, embeds=None):
		super(JointEmbedModel, self).__init__()
		self.vlen = vlen
		self.embedding_dim = config['embedding_dim']
		self.embed = nn.Embedding(self.vlen + 1, self.embedding_dim, padding_idx=self.vlen)
		if embeds is not None:
			self.embed.weight.data.copy_(torch.from_numpy(embeds))
		self.embed.eval()
		for param in self.embed.parameters():
			param.requires_grad = False

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
		# self.lstmq.flatten_parameters()
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

	
class JointEmbedResNet(nn.Module):
	def __init__(self, config, vlen, embeds=None):
		super(JointEmbedResNet, self).__init__()
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

		self.resnet = models.resnet152(pretrained=True)
		self.resnet.fc = nn.Sequential()
		self.resnet.eval()
		for param in self.resnet.parameters():
			param.requires_grad = False
		
		self.resnet_transform = nn.Sequential(
							nn.Linear(2048, 1024),
							nn.BatchNorm1d(1024),
							nn.Tanh())

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
		# self.lstmq.flatten_parameters()
		ques_embedded = self.embed(q_idxs)

		lstm_out, (hn, cn) = self.lstmq(ques_embedded)
		hn = hn.permute(1, 0, 2)
		cn = cn.permute(1, 0, 2)
		hn = hn.contiguous().view(images.shape[0], self.lstm_hidden_size*self.lstm_num_layers)
		cn = cn.contiguous().view(images.shape[0], self.lstm_hidden_size*self.lstm_num_layers)

		ques_embedding = torch.cat((hn,cn), 1)
		ques_embedding_transformed = self.lstm_transform(ques_embedding)

		imgs_embedding = self.resnet(images)
		imgs_embedding = imgs_embedding.view(imgs_embedding.shape[0], -1)
		imgs_embedding = nn.functional.normalize(imgs_embedding)
		imgs_embedding_transformed = self.resnet_transform(imgs_embedding)

		joint_embedding = ques_embedding_transformed*imgs_embedding_transformed

		output = self.mlp(joint_embedding)

		return output


class Attention(nn.Module):
	def __init__(self, v_dim, q_dim, mid_dim, glimpses, drop=0.0):
		super(Attention, self).__init__()
		self.glimpses = glimpses
		self.mid_dim = mid_dim
		self.conv1 = nn.Conv2d(v_dim, mid_dim, 1, bias=False)
		self.fc1 = nn.Linear(q_dim, mid_dim)
		self.conv2 = nn.Conv2d(mid_dim, glimpses, 1)
		self.drop = nn.Dropout(drop)
		self.relu = nn.ReLU()

	def forward(self, v, q):
		n = v.shape[0]
		c = v.shape[1]
		x = self.conv1(self.drop(v))
		y = self.fc1(self.drop(q))
		y = y.view(n, self.mid_dim, 1, 1).expand_as(x)
		x = self.relu(x + y)
		a = self.conv2(self.drop(x))

		v = v.view(n, c, -1)
		a = a.view(n, self.glimpses, -1)
		s = v.shape[2]
		a = a.view(n * self.glimpses, -1)
		a = F.softmax(a, dim=1)
		v = v.unsqueeze(1).expand(n, self.glimpses, c, s)
		a = a.view(n, self.glimpses, -1).unsqueeze(2).expand(n, self.glimpses, c, s)
		x = v * a
		out = x.sum(dim=3)

		return out.view(n, -1)


class AttentionModel(nn.Module):
	def __init__(self, config, vlen, embeds=None):
		super(AttentionModel, self).__init__()
		self.vlen = vlen
		self.embedding_dim = config['embedding_dim']
		self.drop = config['dropout']
		self.glimpses = config['glimpses']
		self.embed = nn.Embedding(self.vlen + 1, self.embedding_dim, padding_idx=self.vlen)
		if embeds is not None:
			self.embed.weight.data.copy_(torch.from_numpy(embeds))

		self.lstm_dim = 1024
		self.lstm_layers = 1
		self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim, num_layers=self.lstm_layers)
		self.v_dim = 2048
		self.mid_dim = 512
		self.drop_layer = nn.Dropout(self.drop)
		self.tanh = nn.Tanh()

		self.resnet152 = models.resnet152(pretrained=True)
		mods = list(self.resnet152.children())[:-2]
		self.resnet152 = nn.Sequential(*mods)
		for param in self.resnet152.parameters():
			param.requires_grad = False

		self.attention = Attention(self.v_dim, self.lstm_dim, self.mid_dim, self.glimpses, self.drop)

		self.classifier = nn.Sequential(
			nn.Dropout(self.drop),
			nn.Linear(self.glimpses * self.v_dim + self.lstm_dim, 1024),
			nn.ReLU(),
			nn.Dropout(self.drop),
			nn.Linear(1024, config['n_classes']))

	def forward(self, images, q_idxs):
		ques = self.embed(q_idxs)
		ques = self.tanh(self.drop_layer(ques))
		_, (_, c) = self.lstm(ques.permute(1, 0, 2))
		q = c.squeeze(0)
		
		v = self.resnet152(images)
		v = F.normalize(v, p=2, dim=1)
		v = self.attention(v, q)
		x = torch.cat((v, q), dim=1)
		out = self.classifier(x)

		return out

class SelfAttention(nn.Module):
	def __init__(self, q_dim=1024, mid_dim=512, drop=0.0):
		super(SelfAttention, self).__init__()
		self.attention_network = nn.Sequential(
				nn.Dropout(drop),
				nn.Linear(2*q_dim,mid_dim),
				nn.Tanh(),
				nn.Dropout(drop),
				nn.Linear(mid_dim,1))
		self.softmax = nn.Softmax(dim=1)

	def forward(self, memory, annotations):
		batch_size, seq_len, hid_size = annotations.size()
		expanded_memory = memory.unsqueeze(1).expand(batch_size,seq_len,hid_size)
		concat = torch.cat((expanded_memory,annotations),2)
		attention_output = self.attention_network(concat)
		normalized_attention = self.softmax(attention_output)             # batch_size x seq_len x 1
		context = torch.sum(normalized_attention*annotations, 1)          # batch_size x hid_size

		return context

class InteractiveAttention(nn.Module):
	def __init__(self, v_dim=2048, q_dim=1024, mid_dim=512, drop=0.0):
		super(InteractiveAttention, self).__init__()
		self.glimpses = 1
		self.mid_dim = mid_dim
		self.conv1 = nn.Conv2d(v_dim, mid_dim, 1)
		self.fc1 = nn.Linear(q_dim, mid_dim)
		self.conv2 = nn.Conv2d(mid_dim, self.glimpses, 1)
		self.drop = nn.Dropout(drop)
		self.relu = nn.ReLU()

	def forward(self, v, q):
		n = v.shape[0]
		c = v.shape[1]
		x = self.conv1(self.drop(v))
		y = self.fc1(self.drop(q))
		y = y.view(n, self.mid_dim, 1, 1).expand_as(x)
		x = self.relu(x + y)
		a = self.conv2(self.drop(x))

		v = v.view(n, c, -1)
		a = a.view(n, self.glimpses, -1)
		s = v.shape[2]
		a = a.view(n * self.glimpses, -1)
		a = F.softmax(a, dim=1)
		v = v.unsqueeze(1).expand(n, self.glimpses, c, s)
		a = a.view(n, self.glimpses, -1).unsqueeze(2).expand(n, self.glimpses, c, s)
		x = v * a
		out = x.sum(dim=3)

		return out.view(n, -1)

class MultihopAttentionModel(nn.Module):
	def __init__(self, config, vlen, embeds=None):
		super(MultihopAttentionModel, self).__init__()
		self.vlen = vlen
		self.embedding_dim = config['embedding_dim']
		self.drop = config['dropout']
		self.hops = config['hops']
		self.embed = nn.Embedding(self.vlen + 1, self.embedding_dim, padding_idx=self.vlen)
		if embeds is not None:
			self.embed.weight.data.copy_(torch.from_numpy(embeds))

		self.v_dim = 2048
		self.mid_dim = 512
		self.drop_layer = nn.Dropout(self.drop)

		self.lstm_dim = 1024
		self.lstm_layers = 2
		self.lstmq = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.lstm_dim, num_layers=self.lstm_layers, batch_first=True, dropout=self.drop)

		self.resnet = models.resnet152(pretrained=True)
		mods = list(self.resnet.children())[:-2]
		self.resnet = nn.Sequential(*mods)
		self.resnet.eval()
		for param in self.resnet.parameters():
			param.requires_grad = False

		self.self_attention = SelfAttention(self.lstm_dim, self.mid_dim, self.drop)
		self.interactive_attention = InteractiveAttention(self.v_dim, self.lstm_dim, self.mid_dim, self.drop)

		self.mlp = nn.Sequential(
			nn.Linear(self.v_dim + self.lstm_dim, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Dropout(self.drop),
			nn.Linear(1024, config['n_classes']))

	def forward(self, images, q_idxs):
		ques = self.embed(q_idxs)
		q_annot, (hn, cn) = self.lstmq(ques)                 # batch_size x seq_len x hid_size

		v_feat = self.resnet(images)
		v_feat = F.normalize(v_feat, p=2, dim=1)

		q = q_annot.mean(dim=1)
		q_memory = q
		v = self.interactive_attention(v_feat, q)
		v_memory = v

		for hop_no in range(self.hops):
			q = self.self_attention(q_memory, q_annot)
			q_memory += q
			v = self.interactive_attention(v_feat, q)
			v_memory += v

		x = torch.cat((v_memory, q_memory), dim=1)
		out = self.mlp(x)

		return out
	
def init_weights(net):
	if type(net) == nn.Linear():
		torch_init.xavier_normal_(net.weight)