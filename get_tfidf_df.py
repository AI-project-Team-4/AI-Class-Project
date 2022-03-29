import pandas as pd
from tqdm import tqdm
import spacy
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import ssl

tqdm.pandas()

ssl._create_default_https_context = ssl._create_unverified_context

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('stopwords')
stops = stopwords.words("english")

def normalize(comment, lowercase, remove_stopwords):
    if comment == "No description yet":
        return ''
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops) and not word.is_punct:
                lemmatized.append(lemma)
    normalized = " ".join(lemmatized)
    return normalized

def get_tfidf_df(train_set, testing_set):
    tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')

    train_set['item_description_new'] = train_set['item_description'].progress_apply(normalize, lowercase=True, remove_stopwords=True)

    tfidf_wm = tfidfvectorizer.fit_transform(train_set['item_description_new'].values.astype('U')) # https://stackoverflow.com/a/39308809/3675086

    tfidf_tokens = tfidfvectorizer.get_feature_names_out()

    df_tfidfvect = pd.DataFrame(
        data = tfidf_wm.toarray(),
        columns = tfidf_tokens
    )

    testing_set['item_description_new'] = testing_set['item_description'].progress_apply(normalize, lowercase=True, remove_stopwords=True)

    tfidf_wm_2 = tfidfvectorizer.transform(testing_set['item_description_new'].values.astype('U')) # https://stackoverflow.com/a/39308809/3675086

    tfidf_tokens_2 = tfidfvectorizer.get_feature_names_out()

    df_tfidfvect_2 = pd.DataFrame(
        data = tfidf_wm_2.toarray(),
        columns = tfidf_tokens_2
    )

    return (df_tfidfvect, df_tfidfvect_2)