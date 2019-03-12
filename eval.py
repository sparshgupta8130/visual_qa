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

def save_results(logits, targets, imgs, ques, corr_fname, incorr_fname):
    fcorr = open(corr_fname, 'w')
    fincorr = open(incorr_fname, 'w')
    for i in range(len(logits)):
        if logits[i] == targets[i]:
            fcorr.write(str(imgs[i]) + ',' + ques[i] + ',' + str(targets[i]) + '\n')
        else:
            fincorr.write(str(imgs[i]) + ',' + ques[i] + ',' + str(targets[i]) + '\n')
    fcorr.close()
    fincorr.close()

def test(model, config, dataloader, vocab, test=True, gen=False):
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
    for param in model.parameters():
            param.requires_grad = False

    logits = []
    targets = []
    imgs = []
    ques = []
    if gen is True:
        corr_fname = eval_path + ts + model_name + '_corr.csv'
        incorr_fname = eval_path + ts + model_name + '_incorr.csv'
        for images, questions, answers, image_ids, questions_orig in dataloader:
            out = forward_pass(model, config, images, questions, vocab, train=False)
            preds = torch.argmax(out, dim=1).tolist()
            logits = logits + preds
            targets = targets + answers.tolist()
            imgs = imgs + image_ids.tolist()
            ques = ques + questions_orig
        save_results(logits, targets, imgs, ques, corr_fname, incorr_fname)
        score = get_accuracy(logits, targets)
        print("Accuracy : ", score)
        write_score(eval_fname, score)
    elif test is True:
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