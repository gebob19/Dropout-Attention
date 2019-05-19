#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Project

Usage:
    train.py [options]

Options:
    -h --help                               show this screen.
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
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


def train(args):
    n_words = int(args['--n-words'])

    df = pd.read_csv('train.csv')
    
    lang = load_model()
    lang = lang.top_n_words_model(n_words)

    hidden_size = int(args['--hidden-size'])
    embed_size = int(args['--embed-size'])
    model = SentModel(embed_size, hidden_size, lang, device)
    model = model.to(device)

    lr = float(args['--lr'])
    clip_grad = float(args['--clip-grad'])
    optimizer = torch.optim.Adam(model.parameters())
    loss_fcn = nn.BCELoss()

    train_batch_size = int(args['--batch-size'])
    epochs = int(args['--max-epoch'])

    for e in range(epochs):
        epoch_loss = 0
        train_iter = 0
        begin_time = time.time()
        
        for sents, targets in batch_iter(lang, df, train_batch_size, shuffle=True):
            train_iter += 1 
            optimizer.zero_grad()
        
            preds = model(sents)
            loss = loss_fcn(preds, targets)
            epoch_loss += loss.item()
        
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        print('epoch %d, avg. loss %.2f, time elapsed %.2f sec' % (e, 
                                                                    epoch_loss / train_iter,
                                                                    time.time() - begin_time), file=sys.stderr)


def main():
    args = docopt(__doc__)
    
    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    train(args)

if __name__ == '__main__':
    print('using device {}...'.format(device))
    main()