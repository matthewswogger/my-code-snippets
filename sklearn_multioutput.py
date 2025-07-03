import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class InvoiceRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """
    A custom regressor that enforces monotonicity on the predicted values.
    This is useful for cashflow forecasting, where the predicted values should
    be non-decreasing over time.
    """

    def __init__(self, enforce_monotonicity: bool = True):
        self.enforce_monotonicity = enforce_monotonicity

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.pairwise = False
        return tags

    def fit(self, X, y):
        X, y = validate_data(
            self, X, y, accept_sparse=("csr", "csc"), multi_output=False, y_numeric=True
        )
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=("csr", "csc"), reset=False)

        if self.enforce_monotonicity:
            return np.sort(X, axis=1)

        return X
