import torch
import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding

from utils import bert_input_tensor

class SaveModel(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path):
        params = {
            'vocab': self.language,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

class BERTClassificationWrapper(SaveModel):
    def __init__(self, device, tokenizer, number_classes, max_seq_len, hidden, n_layers, attn_heads, dropout, attention_dropout):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.number_classes = number_classes
        self.device = device
        self.bert = BERT(len(tokenizer.vocab), 
                         device, 
                         hidden, 
                         n_layers, 
                         attn_heads, 
                         dropout, 
                         attention_dropout)
        self.linear = nn.Linear(hidden, number_classes)
        
    def forward(self, sentences):
        # tokenize + id sentences using bert tokenizer
        x, _ = bert_input_tensor(self.tokenizer, sentences, self.max_seq_len, self.device)
        
        # model pass through
        x = self.bert(x, segment_info=None)
        
        # embedding of [CLS] 
        x = self.linear(x[0, :, :])
        if self.number_classes == 1:
            y = torch.sigmoid(x)
        else:
            y = torch.softmax(x, -1)
        return y.squeeze()

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, device, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, attention_dropout=False):
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

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout, attention_dropout, device) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
