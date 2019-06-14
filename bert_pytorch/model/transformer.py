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

        # DONT APPLY DROPOUT ELSE WHERE
        if attention_dropout:
            dropout = 0.

        # self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention = nn.MultiheadAttention(hidden, attn_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, need_weights=False)[0])
        x = self.output_sublayer(x, self.feed_forward)
        x = self.dropout(x) 
        return x

