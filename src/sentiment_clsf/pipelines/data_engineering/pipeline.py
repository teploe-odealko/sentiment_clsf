from kedro.pipeline import Pipeline, node

from .nodes import preprocess_reviews


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_reviews,
                inputs="reviews",
                outputs="reviews_preprocessed",
                name="preprocessing_reviews",
            ),
        ]
    )
