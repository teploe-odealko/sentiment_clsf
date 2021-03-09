from typing import Dict, List
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import logging



def lemmatize_str(text):
    tokenizer = RegexpTokenizer("[A-Za-zА-Яа-я]+")
    tokens = tokenizer.tokenize(text)

    # Lowercase and lemmatize
    morph = pymorphy2.MorphAnalyzer()
    tokens_norm = [morph.parse(t.lower())[0].normal_form for t in tokens]
    # tokens_clean = [t for t in tokens_norm if t not in stop_word_list]
    return " ".join(tokens_norm)


def preprocess_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the reviews (lemmatize with spacy)

        Args:
            reviews: Source data.
        Returns:
            Preprocessed (lemmatized) data.
    """

    reviews.text = reviews.text.apply(lemmatize_str)
    # print(reviews.text)
    return reviews


def generate_bow(preprocessed, params):
    vectorizer = CountVectorizer(
        token_pattern=r"[A-Za-zА-Яа-я]+", min_df=params["min_vectorizer_freq"]
    )
    vectorizer.fit(preprocessed.text)
    features = vectorizer.transform(preprocessed.text).toarray()
    return features, vectorizer


def generate_tfidf(preprocessed, params):
    vectorizer = TfidfVectorizer(
        token_pattern=r"[A-Za-zА-Яа-я]+", min_df=params["min_vectorizer_freq"]
    )
    vectorizer.fit(preprocessed.text)
    features = vectorizer.transform(preprocessed.text).toarray()
    return features, vectorizer


def generate_w2v(preproccessed, params):
    w2v_model = Word2Vec(
        min_count=params["w2v"]["min_count"],
        window=params["w2v"]["window"],
        size=params["w2v"]["size"],
        sample=params["w2v"]["sample"],
        alpha=params["w2v"]["alpha"],
        min_alpha=params["w2v"]["min_alpha"],
        negative=params["w2v"]["negative"],
    )
    w2v_model.build_vocab(
        preproccessed.text.apply(lambda x: x.split()), progress_per=1000
    )
    w2v_model.train(
        preproccessed.text.apply(lambda x: x.split()),
        total_examples=w2v_model.corpus_count,
        epochs=params["w2v"]["epochs"],
        report_delay=1,
    )

    sent_emb = preproccessed.text.apply(
        lambda text: np.mean(
            [w2v_model.wv[w] for w in text.split() if w in w2v_model.wv], axis=0
        )
    )
    return np.array([sen.tolist() for sen in sent_emb]), w2v_model


def generate_features(preprocessed: pd.DataFrame, params: Dict) -> List:
    """Generate features from reviews

        Args:
            preprocessed: preprocessed data.
            params: Parameters defined in parameters.yml.
        Returns:
            [Generated features,
            labels,
            trained bag of words vectorizer,
            trained tf-idf vectorizer,
            trained word2vec model]

    """
    # if len(preprocessed) == 0:
    #     raise ValueError('Empty preprocessed dataset')
    log = logging.getLogger(__name__)
    try:
        bow_features, bow_victorizer = generate_bow(preprocessed, params)
        tfidf_features, tfidf_vevtorizer = generate_tfidf(preprocessed, params)
    except ValueError as e:
        log.error(e)
        return [np.ndarray(0), np.ndarray(0)]
    w2v_features, w2v_model = generate_w2v(preprocessed, params)

    stacked_features = np.concatenate(
        [bow_features, tfidf_features, w2v_features], axis=1
    )
    # fs = SelectKBest(score_func=f_regression, k=params["k_best_features"])
    # best_features_selector = fs.fit(stacked_features, preprocessed.label)
    # best_features = fs.transform(stacked_features)
    return [stacked_features,
            preprocessed.label.values,
            bow_victorizer,
            tfidf_vevtorizer,
            w2v_model,
            # best_features_selector
            ]
