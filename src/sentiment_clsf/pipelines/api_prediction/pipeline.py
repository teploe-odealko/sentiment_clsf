from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            # node(
            #     get_api_date,
            #     "review_from_user",
            #     ["predict_date", "row_index", "to_predict"]
            # ),
            node(
                preprocess,
                ["review_from_user"],
                "preprocessed"
            ),
            node(
                generate_features,
                ["preprocessed",
                 "bow_vectorizer",
                 "tfidf_vectorizer",
                 "w2v_model",
                 # "best_features_selector"
                 ],
                "features"
            ),
            node(
                make_prediction,
                ["features", "model_logreg"],
                "prediction"
            )
            # node(
            #     serve_result,
            #     ["predict_date", "row_index", "predict"],
            #     ["predict_date_r", "row_index_r", "predict_r"]
            # )
        ]
    )
