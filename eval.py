from utils import create_directory, get_time_prefix
from trainer import forward_pass
import numpy as np
import torch
import torch.nn as nn


def write_score(fname, val):
    f = open(fname, 'w')
    f.write("Accuracy : " + str(val))
    f.close()


def get_accuracy(logits, targets):
    corr = 0
    tot = len(logits)
    for i in range(len(logits)):
        if logits[i] == targets[i]:
            corr += 1

    score = corr / tot
    return score


def test(model, config, dataloader, vocab, test=True):
    eval_path = './evaluations/'
    create_directory(eval_path)
    model_name = config['name']
    ts = get_time_prefix()
    eval_fname = eval_path + ts + model_name + '.txt'

    device = torch.device("cuda:0" if config['GPU'] is True else "cpu")
    model.to(device)
    if config['GPU'] is True:
        model = nn.DataParallel(model, device_ids=config['GPU_Ids'])

    model.eval()

    logits = []
    targets = []
    if test is True:
        for images, questions in dataloader:
            out = forward_pass(model, config, images, questions, vocab, train=False)
    else:
        for images, questions, answers in dataloader:
            out = forward_pass(model, config, images, questions, vocab, train=False)
            preds = torch.argmax(out, dim=1).tolist()
            logits = logits + preds
            targets = targets + answers.tolist()
        score = get_accuracy(logits, targets)
        print("Accuracy : ", score)
        write_score(eval_fname, score)
