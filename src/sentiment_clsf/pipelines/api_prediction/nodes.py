import numpy as np
from sentiment_clsf.pipelines.data_engineering.nodes import lemmatize_str


# def get_api_date(data):
#     return data.headers['date'], \
#            data.json()['index'], \
#            pd.DataFrame.from_dict(data.json()['data'], orient='index').T


def generate_features(preprocessed,
                      bow_vectorizer,
                      tfidf_vectorizer,
                      w2v_model,
                      # best_features_selector
                      ):
    bow_features = bow_vectorizer.transform([preprocessed]).toarray()
    tfidf_features = tfidf_vectorizer.transform([preprocessed]).toarray()

    sent_emb = np.mean(
            [w2v_model.wv[w] for w in preprocessed.split() if w in w2v_model.wv], axis=0
        )
    w2v_features = np.array(sent_emb.tolist())
    # print(type(bow_features))
    # print(bow_features.reshape(-1).shape, tfidf_features.shape, w2v_features.shape)
    stacked_features = np.concatenate(
        [bow_features, tfidf_features, w2v_features.reshape(1, -1)], axis=1
    )
    # best_features = best_features_selector.transform(stacked_features)
    # log = logging.getLogger(__name__)
    # log.warning(best_features)
    # print(stacked_features.shape)
    return stacked_features


def preprocess(text):
    lemmatized = lemmatize_str(text)
    return lemmatized


def make_prediction(features, model):

    predict = model.predict(features)
    return predict[0]


# def serve_result(predict_date, row_index, predict):
#     """
#     answer = {"predict_date": _90.headers['date'],
#               "row_index": _90.json()['index'],
#               "predict": float(rf_model.predict(df))
#               }
#     """
#
#     return predict_date, row_index, float(predict) if len(predict) == 1 else list(predict)
