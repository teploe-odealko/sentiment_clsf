import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    """Splits data into training and test sets.

        Args:
            data: preprocessed data.
            parameters: Parameters defined in parameters.yml.
        Returns:
            A list containing split data.

    """
    data, data_test, = train_test_split(data, test_size=parameters['test_size'], random_state=parameters['seed'],
                                        stratify=data.label)
    data_train, data_val = train_test_split(data, test_size=parameters['test_size'], random_state=parameters['seed'],
                                            stratify=data.label)
    X_train = data_train.text
    y_train = data_train.label
    X_val = data_val.text
    y_val = data_val.label
    X_test = data_test.text
    y_test = data_test.label

    return [X_train, X_val, X_test, y_train, y_val, y_test]


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train the linear regression model.

        Args:
            X_train: Training data of independent features.
            y_train: Training data for price.

        Returns:
            Trained model.

    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(regressor: LinearRegression, X_test: np.ndarray, y_test: np.ndarray):
    """Calculate the coefficient of determination and log the result.

        Args:
            regressor: Trained model.
            X_test: Testing data of independent features.
            y_test: Testing data for price.

    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f.", score)
