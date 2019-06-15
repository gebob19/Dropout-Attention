import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, attention_dropout, device):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        self.device = device
        
        self.input_sublayer = SublayerConnection(device, size=hidden, dropout=dropout, attention_dropout=attention_dropout)
        self.output_sublayer = SublayerConnection(device, size=hidden, dropout=dropout, attention_dropout=attention_dropout)

        self.ln1 = nn.LayerNorm(hidden, eps=1e-12)
        self.ln2 = nn.LayerNorm(hidden, eps=1e-12)

        # DONT APPLY DROPOUT ELSE WHERE
        if attention_dropout:
            dropout = 0.

        # self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        # self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden, attn_heads, dropout=0.)
        self.feed_forward = nn.Sequential(nn.Linear(hidden, feed_forward_hidden),
                                        nn.ReLU(), 
                                        nn.Linear(feed_forward_hidden, hidden))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h, mask):
        x, _ = self.attention(h, h, h)
        x = self.dropout(x)
        h = self.ln1(x + h)

        x = self.feed_forward(h)
        x = self.dropout(x)
        h = self.ln2(x + h)

        # x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, need_weights=False)[0])
        # x = self.output_sublayer(x, self.feed_forward)
        # x = self.dropout(x) 
        return h

