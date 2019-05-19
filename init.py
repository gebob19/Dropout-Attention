import pandas as pd
from pathlib import Path

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

    # read the corpus into a language data model
    lang = Lang()
    populate_language(lang, df, base)
    dump_model(lang)
    

