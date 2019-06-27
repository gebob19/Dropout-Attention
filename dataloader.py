import pprint
import math
import random
import torch

import pandas as pd

from pathlib import Path

from language_structure import *
from utils import clip_pad_to_max

class BatchLoader:
    def __init__(self, base, tokenizer, max_len, device, size):
        self.base = base
        self.size = size
        self.traindf = pd.read_csv(base/'train.csv')
        self.testdf = pd.read_csv(base/'test.csv')
        self.max_len = max_len
        self.device = device
        self.tokenizer = tokenizer
        
        if self.size:
            self.traindf = self.traindf.sample(frac=1.)
            self.testdf = self.testdf.sample(frac=1.)
            self.traindf = self.traindf[:self.size]
            self.testdf = self.testdf[:self.size]
        print("Length of (Train, Test) : ({}, {})".format(len(self.traindf), len(self.testdf)))
    
    def prepare(self, df):
        raise NotImplementedError("BatchLoader Prepare Not Implemented")
    
    def tokenize(self, sents):
        raise NotImplementedError("BatchLoader Tokenize Not Implemented")
        
    def batch_iter(self, batch_size, train=True, shuffle=False, process_full_df=False, show_progress=False):
        data = self.traindf if train else self.testdf
        tmpdf = data.copy()
        if shuffle:
            tmpdf = tmpdf.sample(frac=1.)
            
        count = 0
        # file length lower and upper bound to batch together
        n = 10
        
        while len(tmpdf) > 0:
            count += 1
            
            # grab first row
            length = tmpdf['file_length'].values[0]
            lb, ub = length - n, length + n

            # find similar lengthed files
            file_lengths = tmpdf.file_length.values
            fl_idxs = tmpdf.file_length.index
            idxs = [i for i, fl in zip(fl_idxs, file_lengths) if (fl >= lb and fl <= ub)]

            # break early if we dont want to process the full dataframe 
            if not process_full_df:
                if len(idxs) < batch_size / 2 and count > 6: break
            elif show_progress and count % 10 == 0:
                print('Examples Left: {}'.format(len(tmpdf)))

            # shuffle & get batch
            random.shuffle(idxs)
            idxs = idxs[:batch_size] if len(idxs) > batch_size else idxs
            batchdf = tmpdf.loc[idxs]
            # remove selected batch rows from main df
            tmpdf = tmpdf[~tmpdf.index.isin(batchdf.index)]
            # open, clean, index txt files 
            sents, targets = self.prepare(batchdf)
            
            # tokenize and tensorize
            x, lengths = self.tokenize(sents)
            y = torch.tensor(targets, device=self.device).long()
        
            yield x, y, lengths, idxs
        
# TWO SENTENCE TASKS

class TwoSentenceLoader(BatchLoader):
    def __init__(self, base, tokenizer, max_len, device, size=None):
        super().__init__(base, tokenizer, max_len, device, size)
        self.number_classes = 2
        
    def tokenize(self, bert_tokens):
        # alread been tokenized by prepare
        return BERT_tokenize_tokens(self.tokenizer, bert_tokens, self.max_len, self.device)
        
    def prepare(self, df):
        sentences = []
        targets = []
        for q, s, y in zip(df[self.s1_name].values,\
                           df[self.s2_name].values,\
                           df[self.target_name].values):
            # clean
            try:
                q = normalizeString(q, stopwords=True, contractions=True)
            except:
                print(q)
            s = normalizeString(s, stopwords=True, contractions=True)
            # tokenize
            q_tokens = self.tokenizer.tokenize(q)
            s_tokens = self.tokenizer.tokenize(s)
            # combine
            s = ['[CLS]'] + q_tokens + ['[SEP]'] + s_tokens
            sentences.append(s)
            targets.append(int(y))
        return sentences, targets

class QQPLoader(TwoSentenceLoader):
    def __init__(self, tokenizer, max_len, device, base=Path('../data/QQP'), size=None):
        super().__init__(base, tokenizer, max_len, device, size)    
        self.s1_name = 'question1'
        self.s2_name = 'question2'
        self.target_name = 'is_duplicate'

class QNLILoader(TwoSentenceLoader):
    def __init__(self, tokenizer, max_len, device, base=Path('../data/QNLI'), size=None):
        super().__init__(base, tokenizer, max_len, device, size)        
        self.s1_name = 'question'
        self.s2_name = 'sentence'
        self.target_name = 'targets'

class RTELoader(TwoSentenceLoader):
    def __init__(self, tokenizer, max_len, device, base=Path('../data/RTE'), size=None):
        super().__init__(base, tokenizer, max_len, device, size)    
        self.s1_name = 'sentence1'
        self.s2_name = 'sentence2'
        self.target_name = 'targets'

# SINGLE SENTENCE TASKS

class SingleSentenceLoader(BatchLoader):
    def __init__(self, base, tokenizer, max_len, device, size=None):
        super().__init__(base, tokenizer, max_len, device, size)
        self.number_classes = 2
        
    def tokenize(self, sentences):
        bert_tokens = [['[CLS]'] + self.tokenizer.tokenize(s) for s in sentences]
        return BERT_tokenize_tokens(self.tokenizer, bert_tokens, self.max_len, self.device)

class COLALoader(SingleSentenceLoader):
    def __init__(self, tokenizer, max_len, device, base=Path('../data/cola_public/raw'), size=None):
        super().__init__(base, tokenizer, max_len, device, size)
        
    def prepare(self, df):
        results = list(zip(df['sentence'].values, df['label'].values))
        sents, targets = [e[0] for e in results], [e[1] for e in results]
        return sents, targets
    
class IMDBLoader(SingleSentenceLoader):
    def __init__(self, tokenizer, max_len, device, base=Path('../data/aclImdb'), size=None):
        super().__init__(base, tokenizer, max_len, device, size)
        
    def open_and_clean(self, path):
        file = open(str(self.base/path), encoding='utf-8').read()
        # contractions = True for BERT consistency 
        clean_file = normalizeString(file, stopwords=False, contractions=True)
        return clean_file
        
    def prepare(self, df):
        results = [(self.open_and_clean(p), t) for (p, t) in zip(df['path'].values, df['target'].values)] 
        results = sorted(results, key=lambda e: len(e[0].split(' ')), reverse=True)
        sents, targets = [e[0] for e in results], [e[1] for e in results]
        return sents, targets

# Helper Functions

def BERT_tokenize_tokens(tokenizer, bert_tokens, max_len, device):
    token_ids = [tokenizer.convert_tokens_to_ids(ts) for ts in bert_tokens]
    token_lengths = list(map(len, token_ids))
    token_ids = clip_pad_to_max(token_ids, max_len, 0)
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    # return token_tensor, token_lengths
    return torch.t(token_tensor), token_lengths