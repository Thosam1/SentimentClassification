from typing import Tuple

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from config.config import RANDOM_SEED


def train_and_predict_logistic_regression(
    X_train: csr_matrix,
    y_train: pd.Series,
    X_val: csr_matrix,
    X_test: csr_matrix,
    C: float = 0.1,
    max_iter: int = 300,
) -> Tuple[pd.Series, pd.Series, pd.Series, LogisticRegression]:
    """
    Trains a logistic regression model and returns predictions on train, validation, and test sets.

    Args:
        X_train (csr_matrix): Sparse feature matrix for training data.
        y_train (pd.Series): Labels for training data.
        X_val (csr_matrix): Sparse feature matrix for validation data.
        X_test (csr_matrix): Sparse feature matrix for test data.
        C (float): Inverse of regularization strength; smaller values specify stronger regularization.
        max_iter (int): Maximum number of iterations taken for the solvers to converge.

    Returns:
        Tuple containing:
            - y_train_pred (pd.Series): Predictions on training data.
            - y_val_pred (pd.Series): Predictions on validation data.
            - y_test_pred (pd.Series): Predictions on test data.
            - model (LogisticRegression): Trained LogisticRegression model.
    """
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    return y_train_pred, y_val_pred, y_test_pred, model


def train_and_predict_random_forest(
    X_train: csr_matrix,
    y_train: pd.Series,
    X_val: csr_matrix,
    X_test: csr_matrix,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.Series, pd.Series, pd.Series, RandomForestClassifier]:
    """
    Trains a Random Forest classifier and returns predictions on train, validation, and test sets.

    Args:
        X_train (csr_matrix): Sparse feature matrix for training data.
        y_train (pd.Series): Labels for training data.
        X_val (csr_matrix): Sparse feature matrix for validation data.
        X_test (csr_matrix): Sparse feature matrix for test data.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the tree. None means nodes are expanded until all leaves are pure.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple containing:
            - y_train_pred (pd.Series): Predictions on training data.
            - y_val_pred (pd.Series): Predictions on validation data.
            - y_test_pred (pd.Series): Predictions on test data.
            - model (RandomForestClassifier): Trained Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    return y_train_pred, y_val_pred, y_test_pred, model


def train_and_predict_xgboost(
    X_train: csr_matrix,
    y_train: pd.Series,
    X_val: csr_matrix,
    X_test: csr_matrix,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.3,
    random_state: int = RANDOM_SEED,
    eval_metric: str = "logloss",
) -> Tuple[pd.Series, pd.Series, pd.Series, XGBClassifier]:
    """
    Trains an XGBoost classifier and returns predictions on train, validation, and test sets.

    Args:
        X_train (csr_matrix): Sparse feature matrix for training data.
        y_train (pd.Series): Labels for training data.
        X_val (csr_matrix): Sparse feature matrix for validation data.
        X_test (csr_matrix): Sparse feature matrix for test data.
        n_estimators (int): Number of gradient boosted trees.
        max_depth (int): Maximum tree depth for base learners.
        learning_rate (float): Boosting learning rate.
        random_state (int): Seed for reproducibility.
        eval_metric (str): Evaluation metric for early stopping.

    Returns:
        Tuple containing:
            - y_train_pred (pd.Series): Predictions on training data.
            - y_val_pred (pd.Series): Predictions on validation data.
            - y_test_pred (pd.Series): Predictions on test data.
            - model (XGBClassifier): Trained XGBoost model.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric=eval_metric,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    return y_train_pred, y_val_pred, y_test_pred, model


def train_and_predict_mlp(
    X_train: csr_matrix,
    y_train: pd.Series,
    X_val: csr_matrix,
    X_test: csr_matrix,
    hidden_layer_sizes: Tuple[int, ...] = (100,),
    activation: str = "relu",
    max_iter: int = 200,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.Series, pd.Series, pd.Series, MLPClassifier]:
    """
    Trains an MLP classifier and returns predictions on train, validation, and test sets.

    Args:
        X_train (csr_matrix): Sparse feature matrix for training data.
        y_train (pd.Series): Labels for training data.
        X_val (csr_matrix): Sparse feature matrix for validation data.
        X_test (csr_matrix): Sparse feature matrix for test data.
        hidden_layer_sizes (Tuple[int, ...]): The ith element represents the number of neurons in the ith hidden layer.
        activation (str): Activation function for the hidden layer.
        max_iter (int): Maximum number of iterations.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple containing:
            - y_train_pred (pd.Series): Predictions on training data.
            - y_val_pred (pd.Series): Predictions on validation data.
            - y_test_pred (pd.Series): Predictions on test data.
            - model (MLPClassifier): Trained MLP model.
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    return y_train_pred, y_val_pred, y_test_pred, model
