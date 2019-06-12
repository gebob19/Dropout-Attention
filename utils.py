import torch

import matplotlib.pyplot as plt

from language_structure import normalizeString

def to_input_tensor(lang, sents, max_seq_len, device):
    sents = [s.split(' ') for s in sents]
    sents_id = [indexesFromSentence(lang, s) for s in sents]
    lengths = [len(s) for s in sents_id]
    
    sents_pad = clip_pad_to_max(sents_id, max_seq_len, lang.word2id['<pad>'])
    sents_var = torch.tensor(sents_pad, dtype=torch.long, device=device)
    return torch.t(sents_var), lengths

def indexesFromSentence(lang, sentence):
    return [lang.get_id('<s>')] + [lang.get_id(word) for word in sentence]

# open + clean all examles in a dataframe
def prepare_df(lang, df, base):
    results = [(open_and_clean(lang, p, base), t) for (p, t) in zip(df['path'].values, df['target'].values)] 
    return results

def open_and_clean(lang, path, base):
    file = open(str(base/path), encoding='utf-8').read()
    # contractions = True for BERT consistency 
    clean_file = normalizeString(file, stopwords=False, contractions=True)
    return clean_file

# pad sentences with pad token or clip to equal length 
def clip_pad_to_max(sents, max_sentence_len, pad_token):
    max_seq_len = min(max_sentence_len, max(map(len, sents)))
    resized_sents = []
    for s in sents:
        length = len(s)
        if length > max_seq_len:
            s = s[:max_sentence_len]
        else:
            s = pad(s, max_sentence_len, pad_token)
        resized_sents.append(s)
    return resized_sents

def pad_sents(sents, pad_token):
    sents_padded = []
    max_seq = max(map(len, sents))
    sents_padded = [pad(s, max_seq, pad_token) for s in sents]
    return sents_padded

def pad(s, max_seq, pad_token):
    diff = max_seq - len(s)
    s = s + [pad_token for _ in range(diff)]
    return s

def clip_sents(sents, max_sentence_len):
    min_seq = min(map(len, sents))
    clip_size = min(max_sentence_len, min_seq)
    def clip(s):
        if len(s) > clip_size:
            s = s[:clip_size]
        return s
    sents_clipped = [clip(s) for s in sents]
    return sents_clipped


## Vis. Helpers 

    formatted_sentence.append(str(len(idxs)+1))
    
    return ' '.join(formatted_sentence)

def plot_metrics(metrics, first_n=100):
    print("Total Time: {} \nTrain Time: {}".format(metrics['total_time'], metrics['train_time']))
    print("Max (Validation, Train): (%.2f, %.2f)" % (max(metrics['val_acc']), max(metrics['train_acc'])))
    
    metrics['train_loss'] = metrics['train_loss'][:first_n]
    metrics['val_loss'] = metrics['val_loss'][:first_n]
    metrics['train_acc'] = metrics['train_acc'][:first_n]
    metrics['val_acc'] = metrics['val_acc'][:first_n]
        
    metrics['itr'] = list(range(len(metrics['train_loss'])))
    metrics['val_itr'] = list(range(len(metrics['val_loss'])))
    
    figsize = (20, 15)
    plt.figure(1, figsize=figsize)                
    plt.subplot(211)
    plt.plot(metrics['itr'], metrics['train_loss'],  label='train')
    plt.plot(metrics['val_itr'], metrics['val_loss'], label='validation')
    plt.legend()
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.show()
    
    # plot accuracy
    plt.figure(2, figsize=figsize)                
    plt.subplot(212)
    plt.plot(metrics['itr'], metrics['train_acc'], label='train')
    plt.plot(metrics['val_itr'], metrics['val_acc'], label='validation')
    plt.legend()
    plt.xlabel('Training Iteration')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy @ 50% Threshold')
    
    plt.show()



def compare_metrics(metrics, first_n=100):
    # metric pre-processing
    for m in metrics:
        for key in ['val_acc', 'train_acc', 'val_loss', 'train_loss']:
            m[key] = m[key][:first_n]
        m['itr'] = list(range(len(m['train_loss'])))
        m['val_itr'] = list(range(len(m['val_loss'])))
    names = [m['args']['--save-to'] for m in metrics]    

    for m in metrics:
        print('--------{}----------'.format(m['args']['--save-to']))
        print("Max (Validation, Train): (%.2f, %.2f)" % (max(m['val_acc']), max(m['train_acc'])))
    
    figsize = (15, 15)
    
    plt.figure(1, figsize=figsize)                
    plt.subplot(211)
    for m, name in zip(metrics, names):
        plt.plot(m['itr'], m['train_loss'],  label='{}-train'.format(name))
        plt.plot(m['val_itr'], m['val_loss'], label='{}-validation'.format(name))
    plt.legend()
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.show()
    
    # plot accuracy
    plt.figure(2, figsize=figsize)                
    plt.subplot(212)
    for m, name in zip(metrics, names):
        plt.plot(m['itr'], m['train_acc'],  label='{}'.format(name))
    plt.legend()
    plt.xlabel('Training Iteration')
    plt.ylabel('Accuracy')
    plt.title('Model Training Accuracy @ 50% Threshold')
    
    # plot accuracy
    plt.figure(3, figsize=figsize)                
    plt.subplot(211)
    for m, name in zip(metrics, names):
        plt.plot(m['val_itr'], m['val_acc'], label='{}'.format(name))
    plt.legend()
    plt.xlabel('Training Iteration')
    plt.ylabel('Accuracy')
    plt.title('Model Validation Accuracy @ 50% Threshold')