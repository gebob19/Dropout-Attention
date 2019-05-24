import pandas as pd
from pathlib import Path
import os

from language_structure import Lang, populate_language, dump_model

base = Path('../aclImdb')

def extract_helper(path, target):
    exs, ratings = [], []
    
    for ex in (base/path).iterdir():
        exs.append(path + ex.name)
        ratings.append(int(ex.name.split('_')[-1].split('.')[0]))
    labels = [target] * len(exs)
    return exs, labels, ratings

if __name__ == '__main__':
    # quick and dirty update - update the length of the docs
    if True:
        train_df = pd.read_csv('train.csv')
        lengths = []
        for path in train_df['path'].values:
            file = open(str(base/path), encoding='utf-8').read()
            lengths.append(len(file.split(' ')))
        train_df['file_length'] = lengths
        train_df.to_csv('train.csv', index=False)

    else:
        print('Extracting Data...')
        # parse + structure the *train* data
        neg_exs, neg_labels, neg_ratings = extract_helper('train/neg/', 0)
        pos_exs, pos_labels, pos_ratings = extract_helper('train/pos/', 1)
        
        df = pd.DataFrame(data={'path': neg_exs + pos_exs,
                                'target': neg_labels + pos_labels,
                                'review_rating': neg_ratings + pos_ratings})
        df = df.sample(frac=1.)
        df.to_csv('train.csv', index=False)

        # parse + structure the *test* data
        neg_exs, neg_labels, neg_ratings = extract_helper('test/neg/', 0)
        pos_exs, pos_labels, pos_ratings = extract_helper('test/pos/', 1)

        test_df = pd.DataFrame(data={'path': neg_exs + pos_exs,
                                    'target': neg_labels + pos_labels,
                                    'review_rating': neg_ratings + pos_ratings})
        test_df = test_df.sample(frac=1.)
        test_df.to_csv('test.csv', index=False)

        print('Reading Corpus...')
        # read the corpus into a language data model
        lang = Lang()
        populate_language(lang, df, base)
        dump_model(lang)

        # make dir for saves
        os.mkdir('model_saves')
        os.mkdir('metric_saves')
        
    print('FIN')
        

