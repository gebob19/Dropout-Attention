import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import to_input_tensor

# Thanks to, 
# https://twitter.com/Thom_Wolf/status/1129658539142766592
# AND using the Transformer as Classifier Thanks To, 
# https://www.aclweb.org/anthology/W18-5429
class TransformerClassifier(nn.Module):
    def __init__(self, lang, device, embed_dim, hidden_dim, num_embed, num_pos, num_heads, num_layers, dropout, n_classes):
        super().__init__()
        self.device = device
        self.lang = lang
        
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
        h, x_len = self.encoder(x, self.lang, self.device)
        
        # Create masks for attention to only look left 
        attn_mask = torch.full((x_len, x_len), -float('Inf'), device=self.device, dtype=h.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        
        # Through the layers we go
        for layer_norm1, attention, layer_norm2, feed_forward in zip(self.ln_1, self.attentions,
                                                                     self.ln_2, self.feed_forwards):
            h = layer_norm1(h)
            x, w = attention(h, h, h, attn_mask=attn_mask)
            x = self.dropout(x)
            h = x + h 

            h = layer_norm2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
            
        # bs, sent_len, embed_dim
        h = h.transpose(0, 1)
        x, _ = torch.max(h, 1)
        y = torch.sigmoid(self.classify(x)).squeeze(-1)
        return y

    def save(self, path):
        params = {
            'vocab': self.lang,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

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
