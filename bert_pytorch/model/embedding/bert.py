import torch
import torch.nn as nn
import numpy as np 
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        # self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.token = glove_embeddings(trainable=True)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence)# + self.position(sequence) #+ self.segment(segment_label)
        return x
        # return self.dropout(x)


def glove_embeddings(trainable):
    with open('./glove/imdb_weights.pkl', 'rb') as f:
        weights_matrix = np.load(f, allow_pickle=True)
    mtrx = torch.tensor(weights_matrix)
    
    embedding = nn.Embedding(mtrx.size(0), 300)
    embedding.load_state_dict({'weight': mtrx})
    
    if not trainable:
        embedding.requires_grad = False
    return embedding