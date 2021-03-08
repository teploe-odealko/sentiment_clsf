import logging
from typing import Dict, List

import numpy as np
import pandas as pd

# from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def split_data(features: np.ndarray, labels: np.ndarray, params: Dict) -> List:
    """Splits data into training and test sets.

        Args:
            features: generated features from reviews.
            labels: graund-truth labels
            params: Parameters defined in parameters.yml.
        Returns:
            A list containing split data.

    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=labels,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train,
    )

    return [X_train, X_val, X_test, y_train, y_val, y_test]


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, params: Dict
) -> LogisticRegression:
    """Train the regression models and choose the best one by cv.

        Args:
            X_train: train data.
            y_train: tain labels.

        Returns:
            Trained model.

    """
    parameters = {"C": [0.1, 0.4, 0.5, 1, 1.2, 10]}
    lgreg = LogisticRegression(random_state=params["seed"], max_iter=10000)
    clf = GridSearchCV(lgreg, parameters, scoring="f1")
    clf.fit(X_train, y_train)
    return clf.best_estimator_


def assess_pred_target(target, pred):
    pr, rec, f1, _ = precision_recall_fscore_support(target, pred, average="binary")
    res = pd.DataFrame()
    res["Precision"] = [pr]
    res["Recall"] = [rec]
    res["f1"] = [f1]
    res["accuracy"] = [accuracy_score(target, pred)]
    return res


def evaluate_model(
    regressor: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray
):
    """Calculate the coefficient of determination and log the result.

        Args:
            regressor: Trained model.
            X_test: test data.
            y_test: test labels.

    """
    y_pred = regressor.predict(X_test)

    # score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(assess_pred_target(y_test, y_pred).T)
