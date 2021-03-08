from kedro.pipeline import Pipeline, node

from .nodes import preprocess_reviews, generate_features


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_reviews,
                inputs="reviews",
                outputs="reviews_preprocessed",
                name="preprocessing_reviews",
            ),
            node(
                func=generate_features,
                inputs=["reviews_preprocessed", "parameters"],
                outputs=["generated_features", "labels"],
                name="generating_features",
            ),
        ]
    )
