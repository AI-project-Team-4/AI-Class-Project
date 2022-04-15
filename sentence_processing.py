from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def process_sentence(df, cat, feat=5000):
    # Stemming each instance in the dataset. Creates an list of stemmed words
    new_cat = cat + '_stemmed'
    stemmer = SnowballStemmer("english")
    df[new_cat] = df[cat].apply(lambda x: [stemmer.stem(y) for y in x])
    # Turns that list of stemmed words into a str for Tfidf vectorizer
    df[new_cat] = df[new_cat].apply(' '.join)
    df = df.drop([cat], axis=1)

    # Performs Tfidf Vectorizing on the column
    tfidf = TfidfVectorizer(strip_accents='ascii', max_features=feat, max_df=0.95, min_df=1)
    tfidf.fit(df[new_cat])
    X = tfidf.transform(df[new_cat])
    # Drop the old column (since df will be merged)
    df = df.drop([new_cat], axis=1)

    X_df = pd.DataFrame(X.toarray(), columns=sorted(tfidf.vocabulary_))
    return pd.merge(df, X_df, left_index=True, right_index=True)