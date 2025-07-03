import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib

from cashflow_feature_engineering import (
    cashflow_feature_engineering
)
from model_pipeline import (
    DataConfig,
    ModelConfig,
    ModelParams,
    ProcessData,
    make_transformation_pipeline
)
from evaluation_metrics import crps_score_sklearn


# df.to_parquet('data_files/sample_data.parquet')
# df = pd.read_parquet('data_files/sample_data.parquet')

# df = cashflow_feature_engineering(df)
# df.to_parquet('data_files/feature_engineering_df.parquet')
df = pd.read_parquet('data_files/feature_engineering_df.parquet')


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_config = DataConfig(**config['data_config'])
model_config = ModelConfig(**config['model_config'])
params = ModelParams(**config['params'])


def classify_day(day, day_groups):
    for c in day_groups:
        if day < 0: return None
        elif day <= c: return c
    return None

days_group = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120])
df[data_config.reg_target_name] = df.loc[:, data_config.reg_target_name].apply(lambda x: classify_day(x, days_group))

df_for_training = df.loc[df[data_config.reg_target_name].notna(), :]

# print(df.shape, df_for_training.shape)

process_data = ProcessData(
    data=df_for_training,
    config=data_config
)

X_train, y_train, X_test, y_test = process_data.make_X_y(train=True)

column_transformers = make_transformation_pipeline(config=model_config)
hgb_cassifier = HistGradientBoostingClassifier(
    loss='log_loss',
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_features=1.0,
    max_bins=255,
    categorical_features=model_config.cat_features,  # type: ignore[misc]
    verbose=2,
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ('feature_selector', column_transformers),
        ('classifier', hgb_cassifier)
    ]
)

pipeline.fit(X_train, y_train)
# joblib.dump(pipeline, 'hgbr_classifier_pipeline.joblib')

print(pipeline)


def get_quantiles(cdfs, quantiles):
    fake_quants = []
    for q in quantiles:
        fake_quants.append(np.argmax(cdfs > q, axis=1))
    return np.stack(fake_quants, axis=1)


y_train_pmfs = pipeline.predict_proba(X_train)
y_test_pmfs = pipeline.predict_proba(X_test)

y_train_cdfs = np.cumsum(y_train_pmfs, axis=1)
y_test_cdfs = np.cumsum(y_test_pmfs, axis=1)

y_train_quantiles = get_quantiles(y_train_cdfs, model_config.quantiles)
y_test_quantiles = get_quantiles(y_test_cdfs, model_config.quantiles)

print(
    '-'*100,'\n',
    f'num classes: {days_group.shape[0]}\n',
    'CRPS scores:\n',
    f'train: {crps_score_sklearn(y_train, y_train_quantiles, model_config.quantiles)}\n',
    f'test: {crps_score_sklearn(y_test, y_test_quantiles, model_config.quantiles)}'
)


# X_test.groupby(level=0).count().sort_values(by='invoice_amt', ascending=False)

vendor_id = 39980
# vendor_id = 740751
# vendor_id = 36738
# vendor_id = 754640

y_test_pmfs = pipeline.predict_proba(X_test.loc[vendor_id,:])
y_test_cdfs = np.cumsum(y_test_pmfs, axis=1)
y_test_quantiles = get_quantiles(y_test_cdfs, model_config.quantiles)

print(
    '-'*100,'\n',
    f'num classes: {days_group.shape[0]}\n',
    'CRPS scores:\n',
    f'test: {crps_score_sklearn(y_test.loc[vendor_id,:], y_test_quantiles, model_config.quantiles)}'
)

########################################################################################################################
########################################################################################################################

y_test_pmfs = pd.DataFrame(y_test_pmfs, index=X_test.loc[vendor_id,:].index, columns=days_group)
y_test_cdfs = pd.DataFrame(y_test_cdfs, index=X_test.loc[vendor_id,:].index, columns=days_group)
y_test_quantiles = pd.DataFrame(
    y_test_quantiles,
    index=X_test.loc[vendor_id,:].index,
    columns=model_config.quantiles  # type: ignore[misc]
)

########################################################################################################################
########################################################################################################################

filtered_vendor_df = df_for_training.loc[
    df_for_training.supplier_number == vendor_id, :
].set_index(['unique_invoice_id', 'unique_payer_id']).loc[:, ['invoice_amt', 'invoice_date', 'clear_date']]

combined_df = y_test_pmfs.merge(
    filtered_vendor_df,
    how='left',
    left_index=True, right_index=True
)
combined_df.loc[:, :120] = combined_df.loc[:, :120].multiply(combined_df.loc[:, 'invoice_amt'], axis='index')

dates_from_invoices = []
for days_from_invoice in combined_df.loc[:, :120].columns:
    dates_from_invoices.append(combined_df.invoice_date + pd.Timedelta(days=days_from_invoice))

dates_from_invoices = pd.DataFrame(dates_from_invoices).T
dates_from_invoices.columns = combined_df.loc[:, :120].columns

########################################################################################################################
########################################################################################################################

time_stuff = dates_from_invoices.loc[:,:120].melt(
    var_name='days_from_invoice',
    value_name='date_from_invoice'
)
money_stuff = combined_df.loc[:,:120].melt(
    var_name='days_from_invoice',
    value_name='expected_cashflow_forecast'
)

forecasts = money_stuff.merge(
    time_stuff,
    left_index=True, right_index=True
).groupby('date_from_invoice').agg(
    {'expected_cashflow_forecast': 'sum'}
).sort_index().cumsum()

truths = combined_df.groupby('clear_date').agg(
    {'invoice_amt': 'sum'}
).sort_index().cumsum().rename(
    columns={'invoice_amt': 'actual_cashflow'}
)

forecast_truths = forecasts.merge(
    truths,
    how='outer',
    left_index=True, right_index=True
).sort_index()

print(forecast_truths.interpolate())

# forecast_truths.interpolate().plot(figsize=(15,5))
