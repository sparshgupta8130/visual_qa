import sys
import os
import torch
from torchvision import transforms
import warnings
from cfg import *
from models import JointEmbedModel, AttentionModel
from eval import test
from load_data import get_embeds, create_dataloader, get_vocab
from trainer import train


def main():
	data_dir = config['data_dir']
	dataset = config['dataset']
	train_image_dir = data_dir + config['train_img_dir']
	val_image_dir = data_dir + config['val_img_dir']
	test_image_dir = data_dir + config['test_img_dir']
	train_csv = data_dir + 'train_' + dataset + '_data.csv'
	val_csv = data_dir + 'val_' + dataset + '_data.csv'
	test_csv = data_dir + 'test-dev_data.csv'
	transform_steps = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	vocab = get_vocab(train_csv)
	vocab, embeds = get_embeds(data_dir, dataset, vocab, config['em_fname'])
	vlen = len(vocab)
	train_dataloader = create_dataloader(config, transform_steps, train_image_dir, train_csv, vocab)
	val_dataloader = create_dataloader(config, transform_steps, val_image_dir, val_csv, vocab)
	test_dataloader = create_dataloader(config, transform_steps, test_image_dir, test_csv, vocab)

	if config['train'] is True:
		if config['model'] == 'JointEmbedModel':
			use_config = jointembedmodel_cfg
			use_config['name'] = 'JointEmbedModel_' + dataset + '_'
			use_config['maxlen'] = config['maxlen']
			use_config['n_classes'] = int(dataset)
			model = JointEmbedModel(use_config, vlen, embeds)
		elif config['model'] == 'AttentionModel':
			use_config = attentionmodel_cfg
			use_config['name'] = 'AttentionModel_' + dataset + '_'
			use_config['maxlen'] = config['maxlen']
			use_config['n_classes'] = int(dataset)
			model = AttentionModel(use_config, vlen, embeds)

		print('\n', model)
		print("Number of Parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad), '\n')

		try:
			train(model, use_config, train_dataloader, val_dataloader, vocab)
		except KeyboardInterrupt:
			print('Exiting early from training.')

	else:
		test_model = config['test_model'].split('/')[1]
		if test_model == '':
			raise ValueError('Please pass a model path to test')

		if 'JointEmbedModel' in test_model:
			use_config = jointembedmodel_cfg
			use_config['name'] = 'JointEmbedModel_' + dataset + '_'
			use_config['maxlen'] = config['maxlen']
			use_config['n_classes'] = int(dataset)
		elif 'AttentionModel' in test_model:
			use_config = attentionmodel_cfg
			use_config['name'] = 'AttentionModel_' + dataset + '_'
			use_config['maxlen'] = config['maxlen']
			use_config['n_classes'] = int(dataset)

		model = torch.load(config['test_model'])
		model.to(torch.device("cpu"))
		print('\n', model)
		print("Number of Parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad), '\n')

		test(model, use_config, test_dataloader, vocab)


if __name__ == "__main__":
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		main()