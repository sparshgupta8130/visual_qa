import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
from PIL import Image
import numpy as np
import pickle


class VQADataset(Dataset):

	def __init__(self, transform=transforms.ToTensor(), image_dir='./datasets/VQAimages/train2014', data_csv='./datasets/train_3000_data.csv', color='RGB'):
		
		self.transform = transform
		self.color = color
		self.image_dir = image_dir
		data_all = pd.read_csv(data_csv, sep=",", header=None)

		if len(data_all.columns) == 3:
			self.test = False
			data_all.columns = ["img_id", "ques", "ans_id"]
		else:
			self.test = True
			data_all.columns = ["img_id", "ques"]
		self.data_imgids = data_all["img_id"].astype(int)
		self.data_questions = data_all["ques"]
		if not self.test:
			self.data_answers = data_all["ans_id"].astype(int)
		
	def __len__(self):
		return len(self.data_questions)

	def __getitem__(self, ind):

		data_split = self.image_dir.split('/')[-1]
		image_filename = 'COCO_' + data_split + '_000000' + str(self.data_imgids.ix[ind]).zfill(6) + '.jpg'
		image_path = os.path.join(self.image_dir, image_filename)
		image = Image.open(image_path).convert(mode=str(self.color))

		if self.transform is not None:
			image = self.transform(image)

		if type(image) is not torch.Tensor:
			image = TF.to_tensor(image)

		question = self.data_questions.ix[ind]

		if not self.test:
			answer = torch.tensor(self.data_answers.ix[ind])
			return image, question, answer
		else:
			return image, question


def create_dataloader(config, transform=transforms.ToTensor(), image_dir='./datasets/VQAimages/train2014', data_csv='./datasets/train_3000_data.csv'):
	vqa_dataset = VQADataset(transform, image_dir, data_csv)

	batch_size = config['batch_size']
	num_workers = config['num_workers']
	pin_memory = config['pin_memory']

	dataloader = DataLoader(dataset=vqa_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
	
	return dataloader


def get_vocab(fname):
	ctr = 0
	vocab = {}
	for l in open(fname, 'r'):
		temp = l.strip().split(',')
		ques = temp[1].split()
		for t in ques:
			if t not in vocab:
				vocab[t] = ctr
				ctr += 1

	return vocab


def read_embeds(fname):
	data = []
	i = 0
	word2idx = {}
	for l in open(fname, 'r'):
		temp = l.strip().split()
		data.append(np.array([float(x) for x in temp[1:]]))
		word2idx[temp[0]] = i
		i += 1

	embeddings = np.array(data)
	return embeddings, word2idx


def get_embeds(data_dir, dataset, orig_vocab, em_fname):
	vocab_path = data_dir + dataset + '_vocab.pickle'
	embed_path = data_dir + dataset + '_embed.pickle'

	if os.path.exists(vocab_path):
		with open(vocab_path, 'rb') as f:
			vocab = pickle.load(f)
		with open(embed_path, 'rb') as f:
			embed = pickle.load(f)
	else:
		embeddings, word2idx = read_embeds(em_fname)
		embeds = [0] * len(orig_vocab)
		for v in orig_vocab:
			if v in word2idx:
				embeds[orig_vocab[v]] = embeddings[word2idx[v]]
			else:
				embeds[orig_vocab[v]] = np.random.normal(scale=0.25, size=embeddings.shape[1])
		embeds.append(np.zeros(embeddings.shape[1]))
		embed = np.array(embeds)
		vocab = orig_vocab

		with open(vocab_path, 'wb') as f:
			pickle.dump(vocab, f)
		with open(embed_path, 'wb') as f:
			pickle.dump(embed, f)

	return vocab, embed
