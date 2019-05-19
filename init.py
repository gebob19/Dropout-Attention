import pandas as pd
from pathlib import Path

base = Path('../aclImdb')

def extract_helper(path, target):
    exs, ratings = [], []
    
    for ex in (base/path).iterdir():
        exs.append(path + ex.name)
        ratings.append(int(ex.name.split('_')[-1].split('.')[0]))
    labels = [target] * len(exs)
    return exs, labels, ratings

if __name__ == '__main__':
    neg_exs, neg_labels, neg_ratings = extract_helper('train/neg/', 0)
    pos_exs, pos_labels, pos_ratings = extract_helper('train/pos/', 1)
    
    df = pd.DataFrame(data={'path': neg_exs + pos_exs,
                            'target': neg_labels + pos_labels,
                            'review_rating': neg_ratings + pos_ratings})

    df.to_csv('train.csv', index=False)