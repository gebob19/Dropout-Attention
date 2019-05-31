import torch

from language_structure import normalizeString

def to_input_tensor(lang, sents, device):
    sents_id = [indexesFromSentence(lang, s) for s in sents]
    lengths = [len(s) for s in sents_id]
    sents_pad = pad_sents(sents_id, lang.word2id['<pad>'])
    sents_var = torch.tensor(sents_pad, dtype=torch.long, device=device)
    return torch.t(sents_var), lengths

def indexesFromSentence(lang, sentence):
    return [lang.get_id(word) for word in sentence]

# open + clean all examles in a dataframe
def prepare_df(lang, df, base):
    results = [(open_and_clean(lang, p, base), t) for (p, t) in zip(df['path'].values, df['target'].values)] 
    return results

def open_and_clean(lang, path, base):
    file = open(str(base/path), encoding='utf-8').read()
    clean_file = normalizeString(file, stopwords=False, contractions=False)
    return clean_file

# pad sentences with pad token to equal length
def pad_sents(sents, pad_token):
    sents_padded = []
    max_seq = max(map(len, sents))
    def pad(s):
        diff = max_seq - len(s)
        s = s + [pad_token for _ in range(diff)]
        return s
    sents_padded = [pad(s) for s in sents]
    return sents_padded

def clip_sents(sents):
    min_seq = min(map(len, sents))
    def clip(s):
        if len(s) > min_seq:
            s = s[:min_seq]
        return s
    sents_clipped = [clip(s) for s in sents]
    return sents_clipped