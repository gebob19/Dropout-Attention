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
    --test                                  run model on test set 
    --attention-dropout                     use attention dropout flag
    --IMDB                                  train on the IMDB dataset
    --QQP                                   train on the QQP dataset
    --QNLI                                  train on the QNLI dataset
    --RTE                                   train on the RTE dataset
    --COLA                                  train on the COLA dataset
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 128]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --validate-every=<int>                  validate every [default: 40]
    --max-epoch=<int>                       max epoch [default: 30]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<file>                        model save path [default: default-model]
    --load-from=<file>                      model load path [default: default-model]
    --n-valid=<int>                         number of samples to validate on [default: 10000]
    --dropout=<float>                       dropout [default: 0.3]
    --n-words=<int>                         number of words in language model [default: 10000]
    --max-sent-len=<int>                    max sentence length to encode  [default: 10000]
    --n-heads=<int>                         n of parralel attention layers in MHA [default: 1]
    --n-layers=<int>                        n of transfomer layers stacked [default: 3]
    --dset-size=<int>                       size of the dataset (for quick testing) [default: 0]
    --decrease-dropout=<int>                how many training iterations will pass of non-improvement before decreasing the dropout rate [default: 10]
    --start-decrease=<int>                  when to start to begin to decrease dropout % [default: 100]
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
from bert_pytorch.model.bert import BERTClassificationWrapper
from bert import tokenization
from utils import prepare_df, clip_sents
from language_structure import load_model, Lang
from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(preds, y, threshold=torch.tensor([0.5], device=device)):
    y_preds = torch.softmax(preds, -1)
    y_preds = (y_preds >= threshold).long()
    y = y.cpu().numpy()
    n_correct = (y_preds[np.arange(len(y)), y] == 1).sum()
    return n_correct.cpu().numpy()

def load(path, cpu=False, load_model=True):
    model_dir = 'model_saves/' + path
    # lang = model_checkpoint['vocab']
    if cpu:
        metrics = torch.load(model_dir + '/metrics.pt', map_location='cpu')
    else:
        metrics = torch.load(model_dir + '/metrics.pt')

    if load_model:
        if cpu:
            
            model_checkpoint = torch.load(model_dir + '/model.bin', map_location='cpu')
            optim_checkpoint =  torch.load(model_dir + '/optimizer.pt', map_location='cpu')
        else:
            model_checkpoint = torch.load(model_dir + '/model.bin')
            optim_checkpoint =  torch.load(model_dir + '/optimizer.pt')

        n_heads =           int(metrics['args']['--n-heads'])
        n_layers =          int(metrics['args']['--n-layers'])
        hidden_size =       int(metrics['args']['--hidden-size'])
        max_sentence_len =  int(metrics['args']['--max-sent-len'])
        train_batch_size =  int(metrics['args']['--batch-size'])
        dropout =  float(metrics['args']['--dropout'])

        vocab_file = './uncased_L-12_H-768_A-12/vocab.txt'
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

        model = BERTClassificationWrapper(device,
                            len(tokenizer.vocab),
                            number_classes=2,
                            hidden=hidden_size,
                            n_layers=n_layers,
                            attn_heads=n_heads,
                            dropout=float(metrics['args']['--dropout']),
                            attention_dropout=metrics['args']['--attention-dropout'])

        optimizer = torch.optim.Adam(model.parameters())
        
        model.load_state_dict(model_checkpoint['state_dict'])
        # optimizer.load_state_dict(optim_checkpoint)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(metrics['args']['--lr']))
    else:
        model, optimizer = None, None

    return model, optimizer, None, metrics

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
    args['--hidden-size'] = '10'
    args['--n-heads'] = '1'
    args['--n-layers'] =  '1'

    args['--n-words'] = '10000'
    args['--max-sent-len'] = '100'
    
    args['--log-every'] = '2'
    args['--validate-every'] = '2'
    args['--n-valid'] = '8'
    args['--max-epoch'] = '1'
    
    args['--save-to'] = 'quick_test_model'
    args['--dset-size'] = '20'

    arg_flags = ['--QQP', '--IMDB', '--QNLI', '--RTE', '--COLA']
    for a in arg_flags:
        # reset
        for arg in arg_flags: args[arg] = False
        args[a] = True
        train(args)
        print("Test passed for {}".format(a))

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
    dropout =           float(args['--dropout'])

    assert train_batch_size <= n_valid, "Batch Size must be > Number of Validations"    
    
    # LOAD / INIT MODEL
    if args['--load']:
        model, optimizer, lang, metrics = load(args['--load-from'])
        model = model.to(device)
        print('model loaded...')
    else: 
        vocab_file = './uncased_L-12_H-768_A-12/vocab.txt'
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

        hidden_size = int(args['--hidden-size'])

        model = BERTClassificationWrapper(device,
                                len(tokenizer.vocab),
                                number_classes=2,
                                hidden=hidden_size,
                                n_layers=n_layers,
                                attn_heads=n_heads,
                                dropout=dropout,
                                attention_dropout=args['--attention-dropout'])

        # init weights 
        for p in model.parameters():
            assert p.requires_grad == True
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    dataloader = load_dataloader(args, tokenizer)

    n_params = sum([p.numel() for p in model.parameters()])
    n_train_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Model Stats:")
    print("# of parameters: {}".format(n_params))
    print("# of trainable parameters: {}".format(n_train_params))

    # metric tracking 
    loss_m = []
    time_tracker = []
    accuracy_m = []
    train_itrs = []
    val_iters = []
    epoch_track = []
    val_loss_m = [0]
    val_accuracy_m = [0]
    absolute_start_time = time.time()
    absolute_train_time = 0
    total_iters = 0
    decrease_dropout_waiter = int(args['--start-decrease'])
    w_start = []
    if args['--attention-dropout']:
        for t in model.bert.transformer_blocks:
            w1 = t.input_sublayer.dropout_attention.layer_embedding
            w2 = t.output_sublayer.dropout_attention.layer_embedding
            w_start.append([w1, w2])

    def get_metrics():
        # extract layer attention vectors
        w_end = []
        if args['--attention-dropout']:
            for t in model.bert.transformer_blocks:
                w1 = t.input_sublayer.dropout_attention.layer_embedding
                w2 = t.output_sublayer.dropout_attention.layer_embedding
                w_end.append([w1, w2])

        return {'train_loss': loss_m,
                'train_acc': accuracy_m,
                'train_iterations': train_itrs,
                'epochs': epoch_track,
                'val_loss': val_loss_m,
                'val_acc': val_accuracy_m,
                'val_iterations': val_iters,
                'total_time': round(time.time() - absolute_start_time, 4),
                'train_time': round(absolute_train_time, 4),
                'seconds_spent_training': time_tracker,
                'n_params': n_params,
                'n_train_params': n_train_params,
                'weigth_start': w_start,
                'weight_end': w_end,
                'args': args}


    loss_fcn = nn.CrossEntropyLoss()
    try:
        model.train()
        print('Training...')
        for e in range(epochs):
            epoch_loss = train_iter = val_loss = 0
            total_correct = total_examples = 0
            begin_time = time.time()
            
            for x, y, lengths, idxs in dataloader.batch_iter(train_batch_size, train=True, process_full_df=True, shuffle=True):
                test=False
                torch.cuda.empty_cache()
                start_train_time = time.time()
                train_iter += 1 
                total_iters += 1
                decrease_dropout_waiter -= 1
                optimizer.zero_grad()
                
                # fix that sents is a tensor X which has been tokenified
                y_hat = model(x, lengths)
                loss = loss_fcn(y_hat, y)
                epoch_loss += loss.item()
                
                # accuracy check 
                n_correct = accuracy(y_hat, y)
                total_correct += n_correct
                total_examples += train_batch_size
            
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step() 

                absolute_train_time += time.time() - start_train_time

                # perform validation
                if train_iter % valid_niter == 0 and train_iter > 0:
                    model.eval()
                    n_correct = n_examples = val_loss = 0

                    with torch.no_grad():
                        test=True
                        for x, y, lengths, idxs in dataloader.batch_iter(train_batch_size, train=False, shuffle=True, process_full_df=True):
                            y_hat = model(x, lengths)
                            bcorrect = accuracy(y_hat, y)
                            bloss = loss_fcn(y_hat, y)

                            val_loss += bloss.item()
                            n_correct += bcorrect
                            n_examples += train_batch_size

                            if n_examples > int(args['--n-valid']): break
                    
                    assert n_examples > 0, "Validation Warning: No Examples Recorded"
                    
                    val_acc = n_correct / n_examples
                    val_loss = val_loss / n_examples
    
                    is_better = len(val_accuracy_m) == 0 or val_acc > max(val_accuracy_m)
                    val_accuracy_m.append(round(val_acc, 5))
                    val_loss_m.append(round(val_loss, 5))
                    val_iters.append(total_iters)
                    
                    if is_better: 
                        if args['--save']:
                            print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                            save(model_save_path, get_metrics(), model, optimizer)
                        else:
                            print("Saving not enabeled...")

                    model.train()
                
                # track + log results 
                if train_iter % int(args['--log-every']) == 0:
                    time_tracker.append(round(absolute_start_time - time.time(), 4))
                    loss_m.append(round(epoch_loss / train_iter, 4))
                    accuracy_m.append(total_correct / total_examples)
                    train_itrs.append(total_iters)
                    epoch_track.append(e)
                    epoch_loss = total_correct = total_examples = 0

                    print(('epoch %d, train itr %d, avg. loss %.2f, '
                            'train accuracy: %.2f, avg. val loss %.2f, val acc %.2f '
                            'time elapsed %.2f sec') % (e, train_iter,
                            loss_m[-1], accuracy_m[-1],
                            val_loss_m[-1], val_accuracy_m[-1],
                            time.time() - begin_time), file=sys.stderr)

                # decrease attention dropout every n-steps of no-increase training score
                # if decrease_dropout_waiter < 0:
                #     best_train = max(accuracy_m)
                #     best_last_n = max(accuracy_m[-int(args['--decrease-dropout']):])
                #     if best_train != best_last_n:
                #         dropout = dropout - 0.1
                #         if dropout > 0.:
                #             model.update_dropout(dropout)
                #             decrease_dropout_waiter = int(args['--start-decrease'])
                #             print('Decreased dropout to {}...'.format(dropout))

    finally:
        print(idxs, test)
        if args['--save']:
            metrics = get_metrics()
            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(metrics)

            prefix = 'cancel_e={}_itr={}'.format(e, train_iter) if e != (epochs-1) else 'complete_'
            save(prefix + model_save_path, metrics, model, optimizer)
            print('Model Saved: {}'.format(prefix + model_save_path))

def load_dataloader(args, tokenizer):
    # Load dataloader 
    loader = None 
    if args['--IMDB']:
        loader = IMDBLoader
    elif args['--QQP']:
        loader = QQPLoader
    elif args['--QNLI']:
        loader = QNLILoader
    elif args['--RTE']:
        loader = RTELoader
    elif args['--COLA']:
        loader = COLALoader
    else:
        raise RuntimeError("No Dataloader Specified.")
    loader = loader(max_len=int(args['--max-sent-len']),
                    size=int(args['--dset-size']),
                    device=device, 
                    tokenizer=tokenizer)
    return loader

def test_model(model, dataloader, batch_size, args):
    from tqdm import tqdm
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        model.eval()
        for x, y, lengths, idxs in tqdm(dataloader.batch_iter(batch_size, train=False, process_full_df=True, show_progress=False)):
            y_pred = model(x, lengths)
            n_correct = accuracy(y_pred, y)

            total_correct += n_correct
            total_examples += batch_size
    print("-------------{}-------------".format(args['--save-to']))
    print('Accuracy: %.4f' % (total_correct / total_examples))

def main(): 
    args = docopt(__doc__)
    
    # seed the random number generators
    # seed = int(args['--seed'])
    # torch.manual_seed(seed)
    # np.random.seed(seed * 13 // 7)

    if args['--qtest']:
        qtest(args)
    elif args['--test']:
        assert args['--load'], 'Must load a model for testing...'
        # model
        model, _, _, metrics = load(args['--load-from'])
        loaded_args = metrics['args']
        model = model.to(device)
        # dataloader
        vocab_file = './uncased_L-12_H-768_A-12/vocab.txt'
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        dataloader = load_dataloader(loaded_args, tokenizer)
        # evaluate
        test_model(model, dataloader, int(loaded_args['--batch-size']), loaded_args)
    else: 
        train(args)

if __name__ == '__main__':
    print('using device: {}'.format(device))
    main()