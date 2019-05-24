import torch
import torch.nn as nn

from utils import to_input_tensor

class SentModel(nn.Module):
    def __init__(self, embed_size, hidden_size, lang, device):
        super(SentModel, self).__init__()
        self.device = device
        self.lang = lang
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(len(lang.word2id), embed_size, padding_idx=lang.word2id['<pad>'])
        self.gru = nn.GRU(embed_size, hidden_size, bias=True)
        self.l1 = nn.Linear(hidden_size, 1)
        
    def forward(self, sents):
        s_tensor, lengths = to_input_tensor(self.lang, sents, self.device)
        emb = self.embed(s_tensor)
        
        # pack + rnn sequence + unpack
        x = nn.utils.rnn.pack_padded_sequence(emb, lengths)
        output, hidden = self.gru(x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        
        # batch_size, seq_len, hidden_size
        output_batch = output.transpose(0, 1)
        
        # batch_size, hidden_size
        out_avg = output_batch.sum(dim=1)
        
        # batch_size, 1
        linear_out = self.l1(out_avg)
        out = torch.sigmoid(linear_out).squeeze(-1)
        
        return out
    
    def save(self, path):
        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size),
            'vocab': self.lang,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
