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
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.device = device

        if attention_dropout:
            self.layer_embeddings = nn.Embedding(1, hidden)
            self.task_attention = TaskAttention(device, dropout)

        # self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention = nn.MultiheadAttention(hidden, attn_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, need_weights=False)[0])
        x = self.output_sublayer(x, self.feed_forward)

        if self.attention_dropout:
            task_batch = torch.tensor([0] * x.size(1), device=self.device)
            task_embed = self.layer_embeddings(task_batch).unsqueeze(-1)
            x = x * self.task_attention(x, task_embed)
        else:
            x = self.dropout_layer(x) 

        return x


class TaskAttention(nn.Module):
    def __init__(self, device, dropout):
        super().__init__()
        self.device = device
        self.dropout = dropout
        
    def forward(self, q, k):
        q = q.transpose(0, 1)
        
        # restricted attention dropout
        w = torch.bmm(q, k).squeeze(-1)

        # n is the # words to ignore 
        n = int(w.size(-1) * self.dropout)
        n = n if n != 0 else 1

        # inverse probability hack for multinomial sampling
        mx, _ = torch.max(w, -1)
        mx = mx.unsqueeze(-1)
        p_inv = F.softmax(mx - w, -1)
        attnmask = torch.multinomial(p_inv, n)
        
        # create restricted attention mask
        byte_mask = torch.ones_like(w)
        for bm, mask in zip(torch.split(byte_mask, 1), attnmask):
            bm.squeeze()[mask] = 0. 
        w = byte_mask.to(self.device).unsqueeze(-1)

        w = w.transpose(0, 1)
        return w
