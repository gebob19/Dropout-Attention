import torch
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

class TaskSpecificAttention(SaveModel):
    def __init__(self, language, device, embed_dim, hidden_dim, num_embed, num_heads, num_layers, dropout, n_classes):
        super().__init__()
        self.device = device
        self.language = language
        self.final_dim = 100
        
        self.w_embedding = nn.Embedding(self.language.n_words, embed_dim)
        self.t_embedding = nn.Embedding(num_layers, embed_dim)
        self.t_embedding.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        
        self.mhas, self.linear_1, self.linear_2 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.ff = nn.ModuleList()
        self.ln_1, self.ln_2, self.ln_3 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.tasks = [] 
        self.attention = TaskAttention()
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
        x, lengths = to_input_tensor(self.language, sents, self.device)
        x = x.transpose(0, 1)
        # bs, seq, embed
        h = self.w_embedding(x)

        for task, mha, linear_1, linear_2, feed_forward, lnorm_1, lnorm_2, lnorm_3 in zip(self.tasks, self.mhas, self.linear_1, self.linear_2, self.ff, self.ln_1, self.ln_2, self.ln_3):
            tasks = torch.tensor([task] * batch_size, device=self.device)
            te = self.t_embedding(tasks).unsqueeze(-1)
            
            top = h
            # bs, seq, embed
            x, _ = mha(h, h, h)
            h = x + h
            h = lnorm_1(h)
            # print("After Multihead Attention")
            # (-0.007 mean, 1.5 var)
            # print(torch.mean(x), torch.var(x))
            
            # bs, seq, hidden    
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
            h = lnorm_2(h)
            # print("After FF")
            # very close to zero now (0.12 mean, 0.13 var)
            # print(torch.mean(x), torch.var(x))
        
            # task attention
            w = self.attention(h, te)
            x = 10 * w * h
            h = x + h
            h = lnorm_3(h)

            h = h + top
            # print("After Attention")
            # (-0.007 mean, 0.0218 var)
            # print(torch.mean(x), torch.var(x))
            # h = self.dropout(h)

            # # x = lnorm_2(x)
            # bs, seq, embed
            # x = F.relu(linear_2(x))
            # print("After Linear 2 + RELU")
            # (0.056 mean, 0.0097 var)
            # print(torch.mean(x), torch.var(x))

        # bs, embed, seq
        x = x.transpose(1, 2)

        # # pool + padding
        # p = self.maxpool(x)
        # diff = self.final_dim - p.size(-1)
        # pad = torch.zeros((p.size(0), p.size(1), diff), device=self.device)
        # pad.require_grad = False
        # x = torch.cat([p, pad], -1)

        # classification layers
        # x = F.relu(self.h1(x))
        # x = F.relu(self.h2(x))
        # x = self.h3(x.transpose(-1, -2)).squeeze()
        # y = torch.sigmoid(self.classify(x)).squeeze()

        m, _ = torch.max(x, -1)
        y = torch.sigmoid(self.classify(m)).squeeze()
        
        return y

class TaskAttention(SaveModel):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, te):
        # task attention
        w = torch.bmm(x, te)
        w = torch.softmax(w.squeeze(-1), -1).unsqueeze(-1)
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
