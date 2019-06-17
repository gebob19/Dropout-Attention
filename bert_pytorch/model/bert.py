import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from .transformer import TransformerBlock
from .embedding import BERTEmbedding

from utils import bert_input_tensor, to_input_tensor


class BertSaveModel(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path):
        params = {
            'tokenizer': self.tokenizer,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

class BERTClassificationWrapper(BertSaveModel):
    def __init__(self, device, tokenizer, number_classes, max_seq_len, hidden, n_layers, attn_heads, dropout, attention_dropout):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.number_classes = number_classes
        self.device = device
        
        self.bert = BERT(len(tokenizer.vocab),
                         device, 
                         max_seq_len,
                         hidden, 
                         n_layers, 
                         attn_heads, 
                         dropout, 
                         attention_dropout)
        self.linear = nn.Linear(hidden, number_classes)
        
    def forward(self, sentences):
        # tokenize + id sentences using bert tokenizer
        x, lengths = bert_input_tensor(self.tokenizer, sentences, self.max_seq_len, self.device)

        # model pass through
        x = self.bert(x, segment_info=None, lengths=lengths)
        
        # embedding of [CLS]
        x = self.linear(x[0, :, :])
    
        if self.number_classes == 1:
            y = torch.sigmoid(x)
        else:
            y = torch.softmax(x, -1)
        return y.squeeze()

    def update_dropout(self, new_dropout):
        self.bert.update_dropout(new_dropout)

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, device, max_seq_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, attention_dropout=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # embedding for BERT, sum of positional, segment, token embeddings
        embed_dropout = 0. if attention_dropout else dropout
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=embed_dropout)

        # multi-layers transformer blocks, deep network
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, 4 * hidden, dropout, attention_dropout, device) for _ in range(n_layers)])

    def forward(self, x, segment_info, lengths):
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, lengths) #mask)

        return x

    def update_dropout(self, new_dropout):
        for transformer in self.transformer_blocks:
            transformer.update_dropout(new_dropout)

def glove_embeddings(trainable):
    with open('./glove/imdb_weights.pkl', 'rb') as f:
        weights_matrix = np.load(f, allow_pickle=True)
    mtrx = torch.tensor(weights_matrix)
    
    embedding = nn.Embedding(mtrx.size(0), 300)
    embedding.load_state_dict({'weight': mtrx})
    
    if not trainable:
        embedding.requires_grad = False
    return embedding