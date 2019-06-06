import torch

import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from utils import to_input_tensor

class SaveModel(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, path):
        params = {
            'vocab': self.language,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

def glove_embeddings(trainable):
    with open('./glove/imdb_weights.pkl', 'rb') as f:
        weights_matrix = np.load(f, allow_pickle=True)
    mtrx = torch.tensor(weights_matrix)
    
    embedding = nn.Embedding(mtrx.size(0), 300)
    embedding.load_state_dict({'weight': mtrx})
    
    if not trainable:
        embedding.requires_grad = False
    return embedding

class TaskSpecificAttention(SaveModel):
    def __init__(self, language, device, embed_dim, hidden_dim, num_pos, num_embed, num_heads, num_layers, dropout, n_classes):
        super().__init__()
        self.device = device
        self.language = language
        self.final_dim = 100
        
        # self.w_embedding = nn.Embedding(self.language.n_words, embed_dim)
        self.w_embedding = glove_embeddings(trainable=True)
        embed_dim = 300

        self.pos_embeddings = nn.Embedding(num_pos, embed_dim)

        self.t_embedding = nn.Embedding(num_layers, embed_dim)
        self.ff_embedding = nn.Embedding(num_layers, embed_dim)

        # self.ff_embedding.requires_grad = False
        # self.t_embedding.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.weight1 = nn.Parameter(torch.tensor([[1.]], requires_grad=True))
        self.weight2 = nn.Parameter(torch.tensor([[1.]], requires_grad=True))
        
        self.mhas, self.linear_1, self.linear_2 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.ff = nn.ModuleList()
        self.ln_1, self.ln_2, self.ln_3 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.tasks = [] 
        self.attention = TaskAttention(device)
        # self.maxpool = nn.MaxPool1d(8)
        # self.ln3 = nn.BatchNorm1d(hidden_dim, eps=1e-12)
        
        for i in range(num_layers):
            self.mhas.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.linear_1.append(nn.Linear(embed_dim, hidden_dim))
            self.linear_2.append(nn.Linear(hidden_dim, embed_dim))
            self.ff.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                    nn.ReLU(), 
                                                    nn.Linear(hidden_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embed_dim)))
            self.tasks.append(i)
            
            self.ln_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_2.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_3.append(nn.LayerNorm(embed_dim, eps=1e-12))
        
        self.classify = nn.Linear(embed_dim, n_classes)

        # self.h1 = nn.Linear(self.final_dim, hidden_dim)
        # self.h2 = nn.Linear(hidden_dim, hidden_dim)
        # self.h3 = nn.Linear(embed_dim, 1)
        # self.classify = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, sents):
        batch_size = len(sents)
        x, _ = to_input_tensor(self.language, sents, self.device)

        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        w_embed = self.w_embedding(x)
        h = w_embed + self.pos_embeddings(positions).expand_as(w_embed)
        h = h + self.pos_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        for task, mha, linear_1, linear_2, feed_forward, lnorm_1, lnorm_2, lnorm_3 in zip(self.tasks, self.mhas, self.linear_1, self.linear_2, self.ff, self.ln_1, self.ln_2, self.ln_3):
            tasks = torch.tensor([task] * batch_size, device=self.device)
            ff_tasks = torch.tensor([task] * batch_size, device=self.device)

            te = self.t_embedding(tasks).unsqueeze(-1)
            ffe = self.ff_embedding(ff_tasks).unsqueeze(-1)
             
            # seq, bs, embed
            x, _ = mha(h, h, h)
            # x = self.weight1 * x * self.attention(x, te)
            # x = self.weight1 * x * self.attention(w_embed, te)
            # x = self.weight1 * x 
            # x = self.weight1 * self.attention(x, te)
            # x = self.attention(w_embed, te) + self.attention(x, te)
            # x = x + self.weight1 * self.attention(w_embed, te) * w_embed
            # h = x + h * self.attention(h, te)
            x = x + w_embed * self.attention(w_embed, te)
            h = x + h * self.attention(h, ffe)
            # h = x + h
            h = lnorm_1(h)
            
            # seq, bs, embed
            # x = feed_forward(h)
            # x = self.dropout(x)

            # x = self.weight2 * x * self.attention(x, ffe)
            # x = self.weight2 * x * self.attention(w_embed, ffe)
            # x = self.weight2 * x * self.attention(h, ffe)
            # x = self.weight2 * self.attention(x, ffe)
            # x = self.attention(w_embed, ffe) + self.attention(x, ffe) * x
            # x = x + self.weight2 * self.attention(w_embed, ffe) * w_embed
            # h = x + h * self.attention(h, ffe)
            # h = x + w_embed * self.attention(w_embed, ffe)
            # h = x + h 
            # h = lnorm_2(h)

        # bs, seq, embed_dim
        h = h.transpose(0, 1)

        # BERT classification head 
        # bs, embed_dim
        x = h[:, 0, :]

        # m, _ = torch.max(h, -2)
        y = torch.sigmoid(self.classify(x)).squeeze()
        return y

class TaskAttention(SaveModel):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, x, te):
        ## task attention
        x = x.transpose(0, 1)
        w = torch.bmm(x, te)
        # w = torch.softmax(w.squeeze(-1), -1).unsqueeze(-1)
        
        # restrict attention
        # restict to half of the sentence (can tune later)
        w = w.squeeze(-1)
        # n = w.size(-1) // 2
        # n is the # words to ignore 
        n = int(w.size(-1) * 0.5)

        # inverse probability hack for multinomial sampling
        mx, _ = torch.max(w, -1)
        mx = mx.unsqueeze(-1)
        p_inv = F.softmax(mx - w, -1)
        attnmask = torch.multinomial(p_inv, n)
        # create restricted attention mask
        inf = torch.tensor(float("inf")).to(self.device)
        byte_mask = torch.zeros_like(w)
        for bm, mask in zip(torch.split(byte_mask, 1), attnmask):
            bm.squeeze()[mask] = 1
        attn_bytes = byte_mask.byte().to(self.device)
        # apply restricted attention mask
        w.data.masked_fill_(attn_bytes, -inf)
        # re-scale with softmax
        w = F.softmax(w, -1).unsqueeze(-1)

        w = w.transpose(0, 1)
        return w


class RNN_Self_Attention_Classifier(SaveModel):
    def __init__(self, language, device, batch_size, embed_dim, hidden_dim, num_embed, n_classes):
        super().__init__()
        self.device = device
        self.language = language 
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(num_embed, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, bias=True)
        self.classify = nn.Linear(2 * hidden_dim, n_classes)
        self.attention = RNNAttention()
        
    def forward(self, sents):
        # Embed the sequence
        x, lengths = to_input_tensor(self.language, sents, self.device)
        x_embed = self.embedding(x)
        # RNN encoding
        x = nn.utils.rnn.pack_padded_sequence(x_embed, lengths) 
        x, _ = self.gru(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        
        x = x.transpose(0, 1)
        # get attention over RNN outputs 
        I = torch.eye(max(lengths))
        attn_mask = torch.stack([I] * self.batch_size)
        for i, l in zip(list(range(self.batch_size)), lengths):
            attn_mask[i, :, l:] = 1
            attn_mask[i, l:, :] = 1
        
        attn = self.attention(x, attn_mask, self.device)
        attn_vec = attn.unsqueeze(-1) * x.unsqueeze(1)
        attn_vec = attn_vec.sum(-2)
        attn_out = torch.cat([attn_vec, x], dim=-1)
        
        # max pool over sequence 
        attn_out = attn_out.transpose(-1, -2)
        max_vec, _ = torch.max(attn_out, -1)
        max_vec = max_vec.unsqueeze(-2)
        
        # binary classification activ.
        y = torch.sigmoid(self.classify(max_vec)).squeeze()
        return y        
    
class RNNAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, attn_mask, device):
        # apply attention over RNN outputs (batch, seq, hidden)
        attn = torch.bmm(x, x.transpose(1, 2))
        inf = torch.tensor(float("inf")).to(device)
        attn_bytes = attn_mask.byte().to(device)
        
        attn.data.masked_fill_(attn_bytes, -inf)
        attn = torch.softmax(attn, dim=2)
        # account for padding 
        attn.data.masked_fill_(attn_bytes, 0)
        return attn   


# Thanks to, 
# https://twitter.com/Thom_Wolf/status/1129658539142766592
# AND using the Transformer as Classifier Thanks To, 
# https://www.aclweb.org/anthology/W18-5429
class TransformerClassifier(SaveModel):
    def __init__(self, language, device, embed_dim, hidden_dim, num_embed, num_pos, num_heads, num_layers, dropout, n_classes):
        super().__init__()
        self.device = device
        self.language = language
        
        self.encoder = TransformerEmbedder(embed_dim, num_embed, num_pos, dropout)
        self.dropout = nn.Dropout(dropout)
        
        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.ln_1, self.ln_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                    nn.ReLU(), 
                                                    nn.Linear(hidden_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embed_dim)))
            self.ln_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_2.append(nn.LayerNorm(embed_dim, eps=1e-12))
        
        self.classify = nn.Linear(embed_dim, n_classes)
        
                                      
    def forward(self, x):
        h, x_len = self.encoder(x, self.language, self.device)
        
        # Create masks for attention to only look left 
        attn_mask = torch.full((x_len, x_len), -float('Inf'), device=self.device, dtype=h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        
        # Through the layers we go
        for layer_norm1, attention, layer_norm2, feed_forward in zip(self.ln_1, self.attentions,
                                                                     self.ln_2, self.feed_forwards):
            h = layer_norm1(h)
            x, _ = attention(h, h, h)#, attn_mask=attn_mask)
            x = self.dropout(x)
            # h = x + h 

            h = layer_norm2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            # h = x + h
            
        # bs, sent_len, embed_dim
        h = h.transpose(0, 1)

        ## NEW 
        x = self.classify(h).squeeze(-1)
        y = torch.sigmoid(torch.mean(x, -1))

        # x, _ = torch.max(h, 1)
        # y = torch.sigmoid(self.classify(x)).squeeze(-1)
        return y


# seperate module to make attention analysis easy
class TransformerEmbedder(nn.Module):
    def __init__(self, embed_dim, num_embed, num_pos, dropout):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embed, embed_dim)
        # Lazy positional embeddings
        self.pos_embeddings = nn.Embedding(num_pos, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lang, device):
        x, _ = to_input_tensor(lang, x, device)
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.token_embeddings(x)
        h = h + self.pos_embeddings(positions).expand_as(h)
        h = self.dropout(h)
        
        return h, len(x)
