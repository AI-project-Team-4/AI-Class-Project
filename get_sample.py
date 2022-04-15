# Authors: William Wiemann, Tyler Carr, Benjamin Ranew
# Project title: Mercari Price Prediction Project
# File description: This file was created to get a streamlined sample of training and testing data. Cutoff and test size are specified, and data is pulled from the data file. The data is split into x and y sections for training and testing and then returned. 

import pandas as pd
from sklearn.model_selection import train_test_split

def get_sample(cutoff, test_size):
    # Only the train file is used because it has a column of the true prices, which can be used to compute scores on how accurate our predictions are. There is sufficient data in the train file alone.
    train_df = pd.read_csv("data/train.tsv", sep='\t')[:cutoff]

    return train_test_split(train_df.drop(columns=['price']), train_df['price'], test_size=test_size, random_state=42)