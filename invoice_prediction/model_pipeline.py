from typing import Self, overload
import pandas as pd
import numpy as np
from dataclasses import dataclass#, asdict

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


@dataclass
class DataConfig:
    reg_target_name: str
    row_id_cols: list[str]
    train_date: str
    features_cols: list[str]

@dataclass
class ModelConfig:
    quantiles: list[float]
    days_of_week_order: list[str]
    cat_features: list[str]
    cat_features_to_encode: list[str]

@dataclass
class ModelParams:
    loss: str = 'quantile'
    max_iter: int = 100
    learning_rate: float = 0.05
    max_leaf_nodes: int = 20
    l2_regularization: float = 0.1
    random_state: int = 42
    categorical_features: list[str] | str = 'from_dtype'


class ProcessData:
    def __init__(self, data: pd.DataFrame, config: DataConfig):
        self.data = data
        self.config = config

        if set(self.data.index.names) != set(self.config.row_id_cols):
            self.data = self.data.set_index(self.config.row_id_cols)

    # def _remove_negative_target(self) -> Self:
    #     self.data = self.data.loc[self.data[self.config.reg_target_name] >= 0,:]
    #     return self

    @overload
    def make_X_y(self, train: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        ...

    @overload
    def make_X_y(self, train: bool = False) -> tuple[pd.DataFrame, pd.Series]:
        ...

    def make_X_y(
        self, train: bool | None = None
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series] | tuple[pd.DataFrame, pd.Series]:

        if train:
            # self._remove_negative_target()
            train_set = self.data.loc[self.data.invoice_date <= self.config.train_date,:]
            test_set = self.data.loc[self.data.invoice_date > self.config.train_date,:]

            X_train = train_set.loc[:, self.config.features_cols]
            y_train = train_set.loc[:, self.config.reg_target_name]

            X_test = test_set.loc[:, self.config.features_cols]
            y_test = test_set.loc[:, self.config.reg_target_name]
            return X_train, y_train, X_test, y_test
        else:
            X = self.data.loc[:, self.config.features_cols]

            if self.config.reg_target_name in self.data.columns:
                y = self.data.loc[:, self.config.reg_target_name]
            else:
                y = pd.Series(np.nan, index=X.index, name=self.config.reg_target_name)
            return X, y

def make_transformation_pipeline(config: ModelConfig) -> ColumnTransformer:
    transform_days_of_week = OrdinalEncoder(
        categories=[config.days_of_week_order],  # type: ignore[arg-type]
        dtype=np.float64,
        handle_unknown='use_encoded_value',
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    column_transformers = ColumnTransformer(
        transformers=[
            ('transform_days_of_week', transform_days_of_week, config.cat_features_to_encode),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )
    column_transformers.set_output(transform='pandas')
    return column_transformers
