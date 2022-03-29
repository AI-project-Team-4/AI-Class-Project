import pandas as pd
from sklearn.model_selection import train_test_split

def get_sample(cutoff, test_size):
    train_df = pd.read_csv("data/train.tsv", sep='\t')[:cutoff]

    return train_test_split(train_df.drop(columns=['price']), train_df['price'], test_size=test_size, random_state=42)