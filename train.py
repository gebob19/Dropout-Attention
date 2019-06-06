#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Project

Usage:
    train.py [options]

Options:
    -h --help                               show this screen.
    --qtest                                 quick test mode
    --load                                  load model flag
    --save                                  save model flag 
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 128]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --validate-every=<int>                  validate every [default: 40]
    --max-epoch=<int>                       max epoch [default: 30]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<file>                        model save path [default: default-model]
    --load-from=<file>                      model load path [default: default-model]
    --valid-niter=<int>                     perform validation after how many iterations [default: 500]
    --n-valid=<int>                         number of samples to validate on [default: 10000]
    --dropout=<float>                       dropout [default: 0.3]
    --n-words=<int>                         number of words in language model [default: 10000]
    --max-sent-len=<int>                    max sentence length to encode  [default: 10000]
    --n-heads=<int>                         n of parralel attention layers in MHA [default: 1]
    --n-layers=<int>                        n of transfomer layers stacked [default: 3]
    --dset-size=<int>                       size of the dataset (for quick testing) [default: 0]
"""

import sys
import math
import torch
import time
import pprint
import os
import random

import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import optim
from pathlib import Path
from docopt import docopt
 
from model import *
from utils import prepare_df, clip_sents
from language_structure import load_model, Lang

base = Path('../aclImdb')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_iter(lang, data, batch_size, max_sentence_len, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)

    if shuffle:
        data = data.sample(frac=1.)

    n = 10
    tmpdf = data.copy()
    count = 0
    
    while len(tmpdf) > batch_size:
        count += 1
        # grab first row
        (_, _, _, length) = tmpdf.values[0]
        lb, ub = length - n, length + n

        # find similar lengthed files
        file_lengths = tmpdf.file_length.values
        fl_idxs = tmpdf.file_length.index
        idxs = [i for i, fl in zip(fl_idxs, file_lengths) if (fl >= lb and fl <= ub)]

        if len(idxs) < batch_size / 2 and count > 50: break
        
        # shuffle & get batch
        random.shuffle(idxs)
        idxs = idxs[:batch_size] if len(idxs) > batch_size else idxs
        batchdf = tmpdf.loc[idxs]
        
        # remove selected batch rows from main df
        tmpdf = tmpdf[~tmpdf.index.isin(batchdf.index)]
        
        # open, clean, index txt files 
        results = prepare_df(lang, batchdf, base)
        results = sorted(results, key=lambda e: len(e[0].split(' ')), reverse=True)
        sents, targets = [e[0].split(' ') for e in results], [e[1] for e in results]

        sents = clip_sents(sents, max_sentence_len)
        
        yield sents, torch.tensor(targets, dtype=torch.float32, device=device).squeeze()
    
    # for i in range(batch_num):
    #     # consistent batch sizes
    #     if min((i + 1) * batch_size, len(data)) == len(data): break

    #     lb, ub = i * batch_size, min((i + 1) * batch_size, len(data))
    #     batch_df = data[lb:ub]
        
    #     # open, clean, sort the batch_df
    #     results = prepare_df(lang, batch_df, base)
    #     results = sorted(results, key=lambda e: len(e[0].split(' ')), reverse=True)
    #     sents, targets = [e[0].split(' ') for e in results], [e[1] for e in results]

    #     sents = clip_sents(sents, max_sentence_len)
        
    #     yield sents, torch.tensor(targets, dtype=torch.float32, device=device)


def accuracy(preds, targets, threshold=torch.tensor([0.5], device=device)):
    preds = (preds >= threshold).float()
    n_correct = torch.eq(preds, targets).sum().item()
    return n_correct

def load(path, cpu=False):
    model_dir = 'model_saves/' + path

    if cpu:
        model_checkpoint = torch.load(model_dir + '/model.bin', map_location=lambda storage, loc: storage)
        optim_checkpoint =  torch.load(model_dir + '/optimizer.pt', map_location=lambda storage, loc: storage)
    else:
        model_checkpoint = torch.load(model_dir + '/model.bin')
        optim_checkpoint =  torch.load(model_dir + '/optimizer.pt')

    metrics = torch.load(model_dir + '/metrics.pt')
    lang = model_checkpoint['vocab']

    n_heads =           int(metrics['args']['--n-heads'])
    n_layers =          int(metrics['args']['--n-layers'])
    embed_size =        int(metrics['args']['--embed-size'])
    hidden_size =       int(metrics['args']['--hidden-size'])
    max_sentence_len =  int(metrics['args']['--max-sent-len'])
    train_batch_size =  int(metrics['args']['--batch-size'])
    dropout =  float(metrics['args']['--dropout'])

    model = TaskSpecificAttention(lang, device, embed_size, hidden_size, max_sentence_len, lang.n_words, n_heads, n_layers, dropout, 1)

    # model = TransformerClassifier(language=lang, 
    #                                 device=device,
    #                                 embed_dim=embed_size, 
    #                                 hidden_dim=hidden_size,
    #                                 num_embed=lang.n_words,
    #                                 num_pos=max_sentence_len, 
    #                                 num_heads=n_heads,
    #                                 num_layers=n_layers,
    #                                 dropout=dropout,
    #                                 n_classes=1)
    # model = RNN_Self_Attention_Classifier(language=lang, device=device,
    #                                   batch_size=train_batch_size,
    #                                   embed_dim=embed_size, 
    #                                   hidden_dim=hidden_size,
    #                                   num_embed=lang.n_words,
    #                                   n_classes=1)
    optimizer = torch.optim.Adam(model.parameters())
    
    model.load_state_dict(model_checkpoint['state_dict'])
    # optimizer.load_state_dict(optim_checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(metrics['args']['--lr']))

    return model, optimizer, lang, metrics

def save(model_save_path, metrics, model, optimizer):

    # save metrics
    model_dir = 'model_saves/' + model_save_path
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    else:
        print('Overwritting...')
    
    torch.save(metrics, model_dir + '/metrics.pt')
    # save model + optimizer
    model.save(model_dir + '/model.bin')
    torch.save(optimizer.state_dict(), model_dir + '/optimizer.pt')

    print('Model saved.')


def qtest(args):
    args['--batch-size'] = '2'
    args['--embed-size'] = '100'
    args['--hidden-size'] = '50'
    args['--n-heads'] = '1'
    args['--n-layers'] =  '1'

    args['--n-words'] = '10000'
    
    args['--log-every'] = '2'
    args['--validate-every'] = '1'
    args['--n-valid'] = '8'
    args['--valid-niter'] = '3'
    
    args['--save-to'] = 'quick_test_model'

    train(args)


def train(args):
    torch.autograd.set_detect_anomaly(True)
    n_words =           int(args['--n-words'])
    valid_niter =       int(args['--validate-every'])
    model_save_path =   args['--save-to']
    train_batch_size =  int(args['--batch-size'])
    epochs =            int(args['--max-epoch'])
    clip_grad =         float(args['--clip-grad'])
    n_valid =           int(args['--n-valid'])
    max_sentence_len =  int(args['--max-sent-len'])
    n_heads =           int(args['--n-heads'])
    n_layers =          int(args['--n-layers'])

    assert train_batch_size <= n_valid, "Batch Size must be > Number of Validations"

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    size = int(args['--dset-size'])
    if size:
        train_df = train_df.sample(frac=1.)
        test_df = test_df.sample(frac=1.)
        train_df = train_df[:size]
        test_df = test_df[:size]
    # train on longer lengths 
    # train_df = train_df[train_df.file_length > 200]
    # test_df = test_df[test_df.file_length > 200]
    # train_df = train_df[train_df.file_length < max_sentence_len]
    # test_df = test_df[test_df.file_length < max_sentence_len]
    print("Length of (Train, Test) : ({}, {})".format(len(train_df), len(test_df)))

    if args['--load']:
        model, optimizer, lang, metrics = load(args['--load-from'])
        model = model.to(device)
        print('model loaded...')
    else: 
        lang = load_model()
        lang = lang.top_n_words_model(n_words)

        hidden_size = int(args['--hidden-size'])
        embed_size = int(args['--embed-size'])

        model = TaskSpecificAttention(lang, device, embed_size, hidden_size, max_sentence_len, lang.n_words, n_heads, n_layers, float(args['--dropout']), 1)

        # model = TransformerClassifier(language=lang, device=device,
        #                               embed_dim=embed_size, 
        #                               hidden_dim=hidden_size,
        #                               num_embed=lang.n_words,
        #                               num_pos=max_sentence_len, 
        #                               num_heads=n_heads,
        #                               num_layers=n_layers,
        #                               dropout=float(args['--dropout']),
        #                               n_classes=1)
        # model = RNN_Self_Attention_Classifier(language=lang, device=device,
        #                               batch_size=train_batch_size,
        #                               embed_dim=embed_size, 
        #                               hidden_dim=hidden_size,
        #                               num_embed=lang.n_words,
        #                               n_classes=1)
        # init weights 
        for p in model.parameters():
            assert p.requires_grad == True
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # print('model param check')

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    print("Model Stats:")
    print("# of parameters: {}".format(sum([p.numel() for p in model.parameters()])))
    print("# of trainable parameters: {}".format(sum([p.numel() for p in model.parameters() if p.requires_grad])))

    loss_fcn = nn.BCELoss()

    # metric tracking 
    loss_m = []
    accuracy_m = []
    train_itrs = []
    epoch_track = []
    val_loss_m = [0]
    val_accuracy_m = [0]
    absolute_start_time = time.time()
    absolute_train_time = 0

    def get_metrics():
        return {'train_loss':loss_m,
                'train_acc': accuracy_m,
                'train_iterations': train_itrs,
                'epochs': epoch_track,
                'val_loss': val_loss_m,
                'val_acc': val_accuracy_m,
                'total_time': round(time.time() - absolute_start_time, 4),
                'train_time': round(absolute_train_time, 4),
                'args': args}

    try:
        model.train()
        for e in range(epochs):
            epoch_loss = train_iter = val_acc = val_loss = 0
            total_correct = total_examples = 0
            begin_time = time.time()
            
            # train
            for sents, targets in batch_iter(lang, train_df, train_batch_size, max_sentence_len, shuffle=True):
                torch.cuda.empty_cache()
                start_train_time = time.time()
                train_iter += 1 
                optimizer.zero_grad()
                
                preds = model(sents)
                
                loss = loss_fcn(preds, targets)
                epoch_loss += loss.item()
                
                # accuracy check 
                n_correct = accuracy(preds, targets)
                total_correct += n_correct
                total_examples += train_batch_size
            
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step() 

                absolute_train_time += time.time() - start_train_time

                # perform validation
                if (e > 1 or train_iter > int(args['--valid-niter'])) and train_iter % valid_niter == 0:
                    model.eval()
                    n_examples = n_correct = b_val_loss = 0

                    with torch.no_grad():
                        for i, (val_sents, val_targets) in enumerate(batch_iter(lang, test_df, train_batch_size, max_sentence_len)):
                            val_preds = model(val_sents)
                            batch_n_correct = accuracy(val_preds, val_targets)
                            vloss = loss_fcn(val_preds, val_targets)

                            b_val_loss += vloss.item()
                            n_correct += batch_n_correct
                            n_examples += train_batch_size

                            if i > 1: break

                    if n_examples:
                        val_acc = n_correct / n_examples
                        val_loss = b_val_loss / n_examples

                        is_better = len(val_accuracy_m) == 0 or val_acc > max(val_accuracy_m)
                        val_loss_m.append(round(val_loss / n_examples, 5))
                        val_accuracy_m.append(val_acc)

                        if is_better: 
                            if args['--save']:
                                print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                                save(model_save_path, get_metrics(), model, optimizer)
                            else:
                                print("Saving not enabeled...")
                    else:
                        print("Warning: division by zero in validation avoided", file=sys.stderr)

                    model.train()

                if train_iter % int(args['--log-every']) == 0:
                    # track metrics
                    loss_m.append(round(epoch_loss / train_iter, 4))
                    accuracy_m.append(total_correct / total_examples)
                    train_itrs.append(train_iter)
                    epoch_track.append(e)
                    total_correct = total_examples = 0

                    print(('epoch %d, train itr %d, avg. loss %.2f, '
                            'train accuracy: %.2f, val loss %.2f, val acc %.2f '
                            'time elapsed %.2f sec') % (e, train_iter,
                            loss_m[-1], accuracy_m[-1],
                            val_loss_m[-1], val_accuracy_m[-1],
                            time.time() - begin_time), file=sys.stderr)

                # if args['--qtest'] and train_iter > 50: break

    finally:
        if args['--save']:
            metrics = get_metrics()
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(metrics)

            end = 'cancel' if e != (epochs-1) else 'complete'
            prefix = 'e={}_itr={}_{}_'.format(e, train_iter, end)
            save(prefix + model_save_path, metrics, model, optimizer)

def main(): 
    args = docopt(__doc__)
    
    # seed the random number generators
    # seed = int(args['--seed'])
    # torch.manual_seed(seed)
    # np.random.seed(seed * 13 // 7)

    if args['--qtest']:
        qtest(args)
    else: 
        train(args)

if __name__ == '__main__':
    print('using device: {}'.format(device))
    main()