import pandas as pd

def merge_df(df1, df2):
    df2.reset_index(inplace=True,drop=True)
    return pd.concat([df1, df2], axis=1)