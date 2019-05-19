#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Project

Usage:
    train.py [options]
    train.py qtest

Options:
    -h --help                               show this screen.
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --validate-every=<int>                  validate every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --n-valid=<int>                         number of samples to validate on [default: 1000]
    --dropout=<float>                       dropout [default: 0.3]
    --n-words=<int>                         number of words in language model [default: 10000]

"""

import sys
import math
import torch
import time

import pandas as pd
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch import optim
from pathlib import Path
from docopt import docopt

from model import SentModel
from utils import prepare_df
from language_structure import load_model, Lang

base = Path('../aclImdb')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_iter(lang, data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)

    if shuffle:
        data = data.sample(frac=1.)
    
    for i in range(batch_num):
        lb, ub = i * batch_size, min((i + 1) * batch_size, len(data))
        batch_df = data[lb:ub]
        
        # open, clean, sort the batch_df
        results = prepare_df(lang, batch_df, base)
        results = sorted(results, key=lambda e: len(e[0].split(' ')), reverse=True)
        sents, targets = [e[0].split(' ') for e in results], [e[1] for e in results]
        
        yield sents, torch.tensor(targets, dtype=torch.float32, device=device)


def qtest():
    train({'--batch-size': '8',
            '--clip-grad': '5.0',
            '--dropout': '0.3',
            '--embed-size': '20',
            '--help': False,
            '--hidden-size': '20',
            '--log-every': '2',
            '--lr': '0.001',
            '--max-epoch': '2',
            '--n-valid': '8',
            '--n-words': '1000',
            '--save-to': 'model.bin',
            '--seed': '0',
            '--valid-niter': '1',
            '--validate-every': '1'})


def train(args):
    n_words =           int(args['--n-words'])
    valid_niter =       int(args['--validate-every'])
    model_save_path =   args['--save-to']
    train_batch_size =  int(args['--batch-size'])
    epochs =            int(args['--max-epoch'])
    clip_grad =         float(args['--clip-grad'])
    n_valid =           int(args['--n-valid'])

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    lang = load_model()
    lang = lang.top_n_words_model(n_words)

    hidden_size = int(args['--hidden-size'])
    embed_size = int(args['--embed-size'])
    model = SentModel(embed_size, hidden_size, lang, device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
    loss_fcn = nn.BCELoss()

    hist_valid_scores = []

    for e in range(epochs):
        epoch_loss = train_iter = 0
        begin_time = time.time()
        
        # train
        for sents, targets in batch_iter(lang, train_df, train_batch_size, shuffle=True):
            print('training...{} - {}'.format(train_iter, valid_niter))
            train_iter += 1 
            optimizer.zero_grad()
        
            preds = model(sents)
            loss = loss_fcn(preds, targets)
            epoch_loss += loss.item()
        
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step() 

            # perform validation
            if train_iter % valid_niter == 0:
                print('validating...')
                threshold = torch.tensor([0.5])
                n_examples = n_correct = 0
                test_df = test_df.sample(frac=1.)
                for sents, targets in batch_iter(lang, test_df[:n_valid], train_batch_size):
                    preds = (model(sents) >= threshold).float()
                    n_correct += (1 - torch.abs(preds - targets)).sum()
                    n_examples += len(targets)

                valid_metric = n_correct / n_examples
                print("validation accuracy: %.2f" % (valid_metric))
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better: 
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)
                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
        
        print('epoch %d, avg. loss %.2f, validation_acc: %.2f time elapsed %.2f sec' % (e, 
        epoch_loss / train_iter, 
        valid_metric, 
        time.time() - begin_time), file=sys.stderr)


def main():
    args = docopt(__doc__)
    
    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['qtest']:
        qtest()
    else:
        train(args)

if __name__ == '__main__':
    print('using device {}...'.format(device))
    main()