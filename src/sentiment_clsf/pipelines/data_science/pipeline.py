from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["reviews_preprocessed", "parameters"],
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="splitting_data",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="training_model",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs=None,
                name="evaluating_model",
            ),
        ]
    )
