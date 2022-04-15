# Authors: William Wiemann, Tyler Carr, Benjamin Ranew
# Project title: Mercari Price Prediction Project
# File description: This file cleans up the text (title + description) before sending it through tfidf in the pipeline. It converts text to lowercase, lemmatizes, removes stopwords and punctuation, and removes most words that are only numbers. 

from tqdm import tqdm
import spacy
from nltk.corpus import stopwords
import nltk
import ssl
import re

tqdm.pandas()

ssl._create_default_https_context = ssl._create_unverified_context

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('stopwords')
stops = stopwords.words("english")

# Function: Apply Normalize
# Description: Run normalize function for each row in entire dataframe on the combined_desc column
# Params: df (DataFrame)
# Outputs: df (DataFrame)
def apply_normalize(df):
    df['combined_desc'].progress_apply(normalize, lowercase=True, remove_stopwords=True)
    return df

# Function: Normalize
# Description: Normalize the provided string based on provided params. This cleans up the text so model can make better predictions when running on TFIDF matrix. Force string to be lowercase, remove blank descriptions, lemmatize words, remove stopwords, remove punctuation, remove words less than 2 charas, remove words that are only number digits
# Params: comment (string), lowercase (bool), remove_stopwords (bool)
# Outputs: normalized (string)
def normalize(comment, lowercase, remove_stopwords):
    if "No description yet" in comment:
        comment = comment.replace("No description yet", "")
    if lowercase:
        comment = comment.lower()
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops) and not word.is_punct:
                lemmatized.append(lemma)

    cut = [word for word in lemmatized if len(word) > 2 and not re.search("^[0-9]*$", word)] # Only keep words that are more than 2 characters long, remove words that are only number digits

    normalized = " ".join(cut)
    return normalized