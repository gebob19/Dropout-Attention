import unicodedata
import re
import operator
import pickle

from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

## Main Model
# Thanks to, 
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# & stanford nlp a4
class Lang:
    def __init__(self):
        self.word2id = dict()
        self.word2count = dict()
        self.id2word = dict()
        self.word2id['<pad>'] = 0   # Pad Token
        self.word2id['<s>'] = 1     # Start Token
        self.word2id['</s>'] = 2    # End Token
        self.word2id['<unk>'] = 3   # Unknown Token
        self.fixed_vocab = {'<pad>', '<s>' , '</s>', '<unk>'}
        
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.n_words = len(self.id2word)

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.n_words
            self.word2count[word] = 1
            self.id2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def top_n_words_model(self, n):
        top_lang = Lang()
        ordered_words = sorted(self.word2count.items(), key=operator.itemgetter(1), reverse=True)
        for w, f in ordered_words[:n]: 
            top_lang.addWord(w)
            top_lang.word2count[w] = f 
        return top_lang
    
    def get_id(self, word):
        return self.word2id[word] if word in self.word2id else self.word2id['<unk>']
            

def normalizeString(s, stopwords=True, contractions=False):
    # Remove html tags 
    s = strip_html_tags(s.lower().strip())
    # Lowercase, trim, and remove non-letter characters
    s = unicodeToAscii(s)
    # add spaces too ! ? .
    s = re.sub(r"([.!?])", r" \1 ", s)
    # expand contractions 
    if not contractions:
        s = expand_contractions(s)
    # remove all other characters
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s).strip()
    # remove stop words 
    if not stopwords: 
        s = remove_stopwords(s)
    return s

## Corpus Cleaning Helpers

def strip_html_tags(s):
    soup = BeautifulSoup(s, "html.parser")
    return soup.get_text()

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Expand contractions (it's = it is), thanks to 
# https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# again, thanks to 
# https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72
def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stop_words]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


## Model Helpers

def normalize_and_track(lang, base, path):
    file = open(str(base/path), encoding='utf-8').read()
    # normalize
    clean_file = normalizeString(file, stopwords=True, contractions=False)
    # track words into model
    for w in clean_file.split(' '):
        lang.addWord(w)   
    return True

def populate_language(lang, df, base):
    start_time = time.time()
    results = [normalize_and_track(lang, base, p) for p in df['path'].values]
    duration = time.time() - start_time
    print("Normalized and Tracked in {} seconds".format(duration))
    # Ensure success on all path values
    assert all(results)

def dump_model(lang, name='imdb_language_class'):
    lang_pkl = pickle.dumps(lang, protocol=pickle.HIGHEST_PROTOCOL)
    open('{}.pkl'.format(name), 'wb').write(lang_pkl)
    
def load_model(name='imdb_language_class'):
    with open('imdb_language_class.pkl', 'rb') as fp:
        lang = pickle.load(fp)
    return lang

