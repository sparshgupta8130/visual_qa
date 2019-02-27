import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import os
from PIL import Image


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

def create_dataloader(batch_size=8, transform=transforms.ToTensor(), image_dir='./datasets/VQAimages/train2014', data_csv='./datasets/train_3000_data.csv', extras={}):
	vqa_dataset = VQADataset(transform, image_dir, data_csv)

	num_workers = 0
	pin_memory = False
	if extras:       #CUDA
		num_workers = extras["num_workers"]
		pin_memory = extras["pin_memory"]

	dataloader = DataLoader(dataset=vqa_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
	
	return dataloader

def read_embeds():
    pass
    # TODO: Read embeddings from text file


def get_embeds():
    pass
    # TODO: Fetch relevant embeddings and return to main

val_loader = create_dataloader(batch_size=2, transform=transforms.Resize((224,224)), image_dir='./datasets/VQAimages/val2014', data_csv='./datasets/val_3000_data.csv')
for images, questions, answers in val_loader:
	print images.size(), questions, answers