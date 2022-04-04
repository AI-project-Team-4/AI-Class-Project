from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from get_sample import get_sample
from get_tfidf_df import apply_normalize
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.ensemble import VotingRegressor

X_train, X_test, y_train, y_test = get_sample(cutoff=10000, test_size=0.33)

def category_model(i):
    # Turn category names into numbers for ML model
    category_cols = ['item_condition_id', 'category_name', 'brand_name']

    category_transformer =  ColumnTransformer([
        ('preprocessing', OneHotEncoder(handle_unknown='ignore'), category_cols),
    ])

    category_model = Pipeline([
        ('preprocessing', category_transformer),
        ('model', KNeighborsRegressor(n_neighbors=i))
    ])

    return category_model

def tfidf_model(i):
    # https://stackoverflow.com/a/65298286/3675086
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')

    tfidf_transformer =  ColumnTransformer([
        ('tfidf', tfidf_vectorizer, 'item_description')
    ], sparse_threshold=0)

    tfidf_model = Pipeline([
        ('normalize', FunctionTransformer(apply_normalize)),
        ('tfidf', tfidf_transformer),
        ('model', KNeighborsRegressor(n_neighbors=i))
    ])

    return tfidf_model

def main(i):
    cm = category_model(i)

    tm = tfidf_model(i)

    combined_model = VotingRegressor(estimators=[
        ('category_model', cm),
        ('tfidf_model', tm)
    ])

    combined_model.fit(X_train, y_train)

    predictions = combined_model.predict(X_test)

    arr = [abs(pred - y_test.iloc[i]) < 5 for i, pred in enumerate(predictions)]

    print(f"{sum(arr)} / {len(arr)}")