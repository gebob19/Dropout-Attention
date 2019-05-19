#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Project

Usage:
    train.py [options]

Options:
    -h --help                               show this screen.
    --qtest                                 quick test mode
    --load                                  load model flag
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
    --load-from=<file>                      model load path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 500]
    --n-valid=<int>                         number of samples to validate on [default: 10000]
    --dropout=<float>                       dropout [default: 0.3]
    --n-words=<int>                         number of words in language model [default: 2000]

"""
import sys
import math
import torch
import time
import pprint

import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

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


def accuracy(preds, targets, threshold=torch.tensor([0.5], device=device)):
    preds = (preds >= threshold).float()
    n_correct = torch.eq(preds, targets).sum()
    n_examples = len(targets)
    return n_correct, n_examples


def load(path):
    model_checkpoint = torch.load(path)
    vocab = model_checkpoint['vocab']

    model = SentModel(model_checkpoint['args']['embed_size'],
                     model_checkpoint['args']['hidden_size'],
                     vocab, device)
    optimizer = torch.optim.Adam(model.parameters())
    
    model.load_state_dict(model_checkpoint['state_dict'])
    optimizer.load_state_dict(torch.load(path+'.optim'))

    return model, optimizer, vocab


def qtest():
    train({ '--batch-size': '8',
            '--clip-grad': '5.0',
            '--dropout': '0.3',
            '--embed-size': '20',
            '--help': False,
            '--hidden-size': '20',
            '--load': False,
            '--load-from': 'model.bin',
            '--log-every': '5',
            '--lr': '0.001',
            '--max-epoch': '2',
            '--n-valid': '8',
            '--n-words': '1000',
            '--qtest': True,
            '--save-to': 'model.bin',
            '--seed': '0',
            '--valid-niter': '10',
            '--validate-every': '5'})

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

    if args['--load']:
        model, optimizer, lang = load('model_saves/' + args['--load-from'])
        print('model loaded...')
    else: 
        lang = load_model()
        lang = lang.top_n_words_model(n_words)

        hidden_size = int(args['--hidden-size'])
        embed_size = int(args['--embed-size'])
        model = SentModel(embed_size, hidden_size, lang, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    loss_fcn = nn.BCELoss()

    # metric tracking 
    loss_m = []
    accuracy_m = []
    val_loss_m = []
    val_accuracy_m = []
    absolute_start_time = time.time()
    absolute_train_time = 0

    for e in range(epochs):
        epoch_loss = train_iter = val_acc = val_loss = 0
        begin_time = time.time()
        
        # train
        for sents, targets in batch_iter(lang, train_df, train_batch_size, shuffle=True):
            # print('training...{} - {}'.format(train_iter, valid_niter))
            start_train_time = time.time()
            train_iter += 1 
            optimizer.zero_grad()
        
            preds = model(sents)
            loss = loss_fcn(preds, targets)
            epoch_loss += loss.item()
        
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step() 

            absolute_train_time += time.time() - start_train_time

            # perform validation
            if train_iter > int(args['--valid-niter']) and train_iter % valid_niter == 0:
                # print('validating...')
                threshold = torch.tensor([0.5])
                n_examples = n_correct = val_loss = 0

                test_df = test_df.sample(frac=1.)
                for val_sents, val_targets in batch_iter(lang, test_df[:n_valid], train_batch_size):
                    val_preds = model(val_sents)
                    batch_n_correct, batch_n_examples = accuracy(val_preds, val_targets)

                    val_loss += loss_fcn(val_preds, val_targets).item()
                    n_correct += batch_n_correct
                    n_examples += batch_n_examples

                val_acc = n_correct.float() / n_examples
                val_loss = val_loss / n_examples

                is_better = len(val_accuracy_m) == 0 or val_acc > max(val_accuracy_m)
                val_loss_m.append(round(val_loss / n_examples, 4))
                val_accuracy_m.append(val_acc.item())

                print(('epoch %d, train itr %d, avg. loss %.2f, '
                        'val_acc: %.2f, val_loss: %.2f, '
                        'time elapsed %.2f sec') % (e, train_iter,
                        epoch_loss / train_iter, val_acc, val_loss,
                        time.time() - begin_time), file=sys.stderr)
                begin_time = time.time()

                if is_better: 
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)
                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), 'model_saves/' + model_save_path + '.optim')

            if train_iter % int(args['--log-every']) == 0:
                # track metrics
                loss_m.append(round(loss.item() / len(targets), 4))
                n_correct, n_examples = accuracy(preds, targets)
                accuracy_m.append((n_correct.float() / n_examples).item())
                torch.cuda.empty_cache()
                
                print(('epoch %d, train itr %d, avg. loss %.2f, '
                        'train accuracy: %.2f, '
                        'time elapsed %.2f sec') % (e, train_iter,
                        epoch_loss / train_iter, accuracy_m[-1],
                        time.time() - begin_time), file=sys.stderr)


            if args['--qtest'] and train_iter > 5: break

    metrics = {'train_loss':loss_m,
                'train_acc': accuracy_m,
                'val_loss': val_loss_m,
                'val_acc': val_accuracy_m,
                'total_time': round(time.time() - absolute_start_time, 4),
                'train_time': round(absolute_train_time, 4),
                'args': args}
    torch.save(metrics, 'metric_saves/' + model_save_path + '.metrics')
    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(metrics)

def main():
    args = docopt(__doc__)
    
    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['--qtest']:
        qtest()
    else:
        train(args)

if __name__ == '__main__':
    print('using device: {}'.format(device))
    main()