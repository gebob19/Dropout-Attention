import torch
import torch.nn as nn
import torch.nn.functional as F


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, device, size, dropout, dropout_type):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-12)
        self.dropout_type = dropout_type
        if dropout_type == 'Attention':
            self.dropout_attention = DropoutAttention(size, dropout, device)
            # print('Using Attention Dropout')
        elif self.dropout_type == 'EntireEmbeddingRandom':
            self.embeddingDropout = EmbeddingDropout(size, dropout, device)
            # print('Using Entire EMbedding Drops')
        elif self.dropout_type == 'SingleUnitRandom':
            self.dropout = nn.Dropout(dropout)
            # print('Using Normal Dropout')
        else:
            print('Warning: Not using Dropout ({})'.format(self.dropout_type))

    def forward(self, x, sublayer, lengths):
        "Apply residual connection to any sublayer with the same size."
        h = sublayer(x)
        # apply dropout of choice
        if self.dropout_type == 'Attention':
            h = self.dropout_attention(h, lengths)
        elif self.dropout_type == 'SingleUnitRandom':
            h = self.dropout(h)
        elif self.dropout_type == 'EntireEmbeddingRandom':
            h = self.embeddingDropout(h)
        return self.norm(x + h)

    def update_dropout(self, new_dropout):
        if self.attention_dropout:
            self.dropout_attention.dropout = new_dropout

class EmbeddingDropout(nn.Module):
    def __init__(self, hidden, dropout, device):
        super().__init__()
        self.device = device
        self.dropout = dropout
    
    def forward(self, x):
        # (bs, seq_len, hidden)
        if self.training:
            x = x.transpose(0, 1)
            inds = torch.bernoulli(torch.zeros((x.size(0), x.size(1))) + self.dropout)
            x[torch.nonzero(inds).split(1, dim=1)] = 0 
            x = x.transpose(0, 1)
            return (x * (1. / (1. - self.dropout))) 
        else:
            return x

class DropoutAttention(nn.Module):
    def __init__(self, hidden, dropout, device):
        super().__init__()
        self.device = device
        self.dropout = dropout
        # set as parameter to update
        self.layer_embedding = nn.Parameter(torch.randn(hidden, device=device, requires_grad=True))
        
    def forward(self, x, lengths):
        # set query key value
        bs = x.size(1)
        batch_task = self.layer_embedding.repeat(bs).view(bs, -1).unsqueeze(-1)
        q, k, v = x, batch_task, x

        if self.training:
            # (bs, seq_len, hidden)
            q = q.transpose(0, 1) 
            
            # restricted attention dropout (bs, seq_len)
            w = torch.bmm(q, k).squeeze(-1)

            # n is the # words to ignore 
            n = max(int(w.size(-1) * self.dropout), 1)

            # inverse probability hack for multinomial sampling
            mx, _ = torch.max(w, -1)
            mx = mx.unsqueeze(-1)
            p_inv = F.softmax(mx - w, -1)
            attnmask = torch.multinomial(p_inv, n)
            
            # create restricted attention mask
            byte_mask = torch.ones_like(w)
            for bm, mask in zip(torch.split(byte_mask, 1), attnmask):
                bm.squeeze()[mask] = 0. 
            
            # # ignore paddings
            # for i, length in enumerate(lengths):
            #     bm[i, length:, :] = 0.

            w = byte_mask.to(self.device).unsqueeze(-1)
            w = w.transpose(0, 1)

            # inverse dropout training
            return (v * (1. / (1. - self.dropout))) * w
        else:
            return v