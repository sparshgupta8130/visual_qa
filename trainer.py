import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from utils import create_directory, get_time_prefix, get_indices
import gc
import os
import sys


def print_loss(i, t_loss, v_loss, st_time, en_time):
	print("Epoch: ", i + 1, "\t\tTraining Loss: ", t_loss, "\t\tValidation Loss: ", v_loss, "\t(", en_time - st_time,
		  " s)")


def write_loss(i, fname, t_loss, v_loss):
	f = open(fname, 'a+')
	f.write("Epoch: " + str(i + 1) + "\t\tTraining Loss: " + str(t_loss) + "\t\tValidation Loss: " + str(v_loss) + '\n')
	f.close()


def plot_loss(loss):
	pass
	# TODO: Plot the training and validation loss


def early_stop(min_v_loss, v_loss):
	if v_loss > min_v_loss:
		return True
	else:
		return False


def forward_pass(model, config, images, questions, vocab, answers=None, criterion=None, train=True):
	device = torch.device("cuda:0" if config['GPU'] is True else "cpu")

	images = Variable(images).to(device)
	# q_idxs = get_indices(questions, vocab, config['maxlen']).to(device)
	q_idxs = Variable(questions).to(device)

	out = model(images, q_idxs)

	if train is False:
		del images, q_idxs
		gc.collect()
		if config['GPU'] is True:
			torch.cuda.empty_cache()
		return out

	answers = Variable(answers).to(device)
	loss = criterion(out, answers)

	del images, q_idxs, answers
	gc.collect()
	if config['GPU'] is True:
		torch.cuda.empty_cache()
	return loss.sum()


def train(model, config, train_dataloader, val_dataloader, vocab):

	obs_path = './observations/'
	mod_path = './saved_models/'
	create_directory(obs_path)
	create_directory(mod_path)
	model_name = config['name']
	ts = get_time_prefix()
	obs_fname = obs_path + ts + model_name + '.txt'
	mod_fname = mod_path + ts + model_name + '.pt'

	lr = config['lr']
	L2 = config['L2']
	optimizer = config['opt'](filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=L2)
	early_stop_epoch = config['early_stop_epoch']
	criterion = nn.CrossEntropyLoss()
	n_epochs = config['epochs']
	train_loss = []
	val_loss = []
	min_val_loss = 1e6
	incr = 0

	print("Training Begins...")
	device = torch.device("cuda:0" if config['GPU'] is True else "cpu")
	model.to(device)
	if config['GPU']:
		model = nn.DataParallel(model, device_ids=config['GPU_Ids'])

	for e in range(n_epochs):
		# lr *= config['lr_decay']
		# for param_group in optimizer.param_groups:
		#     param_group['lr'] = lr

		st_time = time.time()

		######## Training Data ########
		model.train()
		t_loss = 0
		for minibatch_no, (images, questions, answers) in enumerate(train_dataloader, 1):
			if (minibatch_no%5 == 0):
				lr *= config['lr_decay']
				for param_group in optimizer.param_groups:
					param_group['lr'] = lr

			optimizer.zero_grad()

			loss = forward_pass(model, config, images, questions, vocab, answers, criterion)
			loss.backward(retain_graph=True)
			optimizer.step()

			if config['GPU'] is True:
				t_loss += loss.cpu().data.numpy()
			else:
				t_loss += loss.data.numpy()

			print (loss.item())
			del loss
			gc.collect()
			torch.cuda.empty_cache()

		train_loss.append(t_loss)

		######## Validation Data ########
		model.eval()
		v_loss = 0
		for images, questions, answers in val_dataloader:
			loss = forward_pass(model, config, images, questions, vocab, answers, criterion)
			if config['GPU'] is True:
				v_loss += loss.cpu().data.numpy()
			else:
				v_loss += loss.data.numpy()

			del loss
			gc.collect()
			torch.cuda.empty_cache()

		val_loss.append(v_loss)
		en_time = time.time()
		print_loss(e, t_loss, v_loss, st_time, en_time)
		write_loss(e, obs_fname, t_loss, v_loss)

		if early_stop(min_val_loss, v_loss):
			incr += 1
		else:
			min_val_loss = v_loss
			incr = 0
			if config['GPU'] is True:
				torch.save(model.module, mod_fname)
			else:
				torch.save(model, mod_fname)
		if incr >= early_stop_epoch and config['early_stop'] is True:
			print("Early Stopping Here...")
			break

	print("\nModel Trained!\n")