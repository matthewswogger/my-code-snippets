import pandas as pd
import holidays


def cashflow_feature_engineering(forecasting_df: pd.DataFrame) -> pd.DataFrame:
    us_holidays = holidays.US()  # type: ignore[misc]

    forecasting_df = forecasting_df.sort_values(['unique_payer_id', 'clear_date']).reset_index().drop(columns=['index'])

    payer_min_max_df = forecasting_df[['unique_payer_id', 'clear_date', 'invoice_date']].groupby(['unique_payer_id']) \
        .agg(
            clear_min=('clear_date', 'min'),
            invoice_min=('invoice_date', 'min'),
            clear_max=('clear_date', 'max'),
            invoice_max=('invoice_date', 'max')
        )
    payer_min_max_df['all_min'] = payer_min_max_df[['clear_min', 'invoice_min']].T.min() - pd.Timedelta(days=1)  # type: ignore[misc]
    payer_min_max_df['all_max'] = payer_min_max_df[['clear_max', 'invoice_max']].T.max()  # type: ignore[misc]
    payer_min_max_df = payer_min_max_df[['all_min', 'all_max']].reset_index()  # type: ignore[misc]

    payer_blank_dfs = []

    for _, row in payer_min_max_df.iterrows():
        new_df = pd.DataFrame(pd.date_range(start=row['all_min'], end=row['all_max']), columns=['join_date'])  # type: ignore[misc]
        new_df['unique_payer_id'] = row['unique_payer_id']
        payer_blank_dfs.append(new_df)

    payer_date_join_df = pd.concat(payer_blank_dfs).reset_index().drop(columns=['index'])

    payer_daily_dupes = payer_date_join_df.merge(
        forecasting_df[['unique_payer_id', 'clear_date', 'days_to_pay']],
        left_on=['unique_payer_id', 'join_date'],
        right_on=['unique_payer_id', 'clear_date'],
        how='left'
    ).drop(columns='clear_date').sort_values(['unique_payer_id', 'join_date'])

    payer_prev_stats = payer_daily_dupes.groupby(['unique_payer_id']) \
        .rolling(window='547d', on='join_date', closed='both').days_to_pay.agg(['mean', 'std']).round(1).reset_index()

    payer_prev_stats = payer_prev_stats[
        (payer_prev_stats.index.isin(
            payer_prev_stats.reset_index().groupby(['unique_payer_id', 'join_date'])['index'].agg('max')))
    ].rename(columns={'mean': 'avg_payer_prev_days_to_pay', 'std': 'std_payer_prev_days_to_pay'})

    prev_days_df = payer_prev_stats.merge(
        forecasting_df[['unique_payer_id', 'clear_date', 'days_to_pay']].groupby(['unique_payer_id', 'clear_date'])
        .days_to_pay.max()
        .reset_index(),
        left_on=['unique_payer_id', 'join_date'],
        right_on=['unique_payer_id', 'clear_date'],
        how='left'
    ).drop(columns=['clear_date'])

    prev_days_df.days_to_pay = prev_days_df.groupby(['unique_payer_id']).days_to_pay.ffill()
    prev_days_df = prev_days_df.rename(columns={'days_to_pay': 'payer_prev_days_to_pay'})

    payer_issued_df = forecasting_df[['unique_payer_id', 'invoice_date', 'clear_date', 'invoice_amt']].copy()
    payer_issued_df['min_date'] = payer_issued_df[['invoice_date', 'clear_date']].min(axis=1)

    payer_issued_df = (payer_issued_df.groupby(['unique_payer_id', 'min_date']).invoice_amt.agg(['count', 'sum'])
        .reset_index().sort_values(['unique_payer_id', 'min_date'])
        .rename(columns={'count': 'invoice_count', 'sum': 'invoice_sum'}))

    payer_issued_groupby = payer_issued_df.set_index('min_date').groupby(['unique_payer_id'])

    payer_issued_df[
        'issued_invoice_count'] = payer_issued_groupby.invoice_count.expanding().sum().reset_index().invoice_count
    payer_issued_df['issued_total_amt'] = payer_issued_groupby.invoice_sum.expanding().sum().reset_index().invoice_sum

    payer_collected_df = forecasting_df[['unique_payer_id', 'clear_date', 'invoice_amt']] \
        .groupby(['unique_payer_id', 'clear_date']).invoice_amt.agg(['count', 'sum']).reset_index() \
        .sort_values(['unique_payer_id', 'clear_date']).rename(columns={'count': 'invoice_count', 'sum': 'invoice_sum'})

    payer_collected_groupby = payer_collected_df.set_index('clear_date').groupby(['unique_payer_id'])

    payer_collected_df[
        'collected_invoice_count'] = payer_collected_groupby.invoice_count.expanding().sum().reset_index().invoice_count
    payer_collected_df[
        'collected_total_amt'] = payer_collected_groupby.invoice_sum.expanding().sum().reset_index().invoice_sum

    payer_outstanding_df = prev_days_df.merge(
        payer_issued_df[['unique_payer_id', 'min_date', 'issued_invoice_count', 'issued_total_amt']],
        left_on=['unique_payer_id', 'join_date'],
        right_on=['unique_payer_id', 'min_date'],
        how='left'
    ).merge(
        payer_collected_df[['unique_payer_id', 'clear_date', 'collected_invoice_count', 'collected_total_amt']],
        left_on=['unique_payer_id', 'join_date'],
        right_on=['unique_payer_id', 'clear_date'],
        how='left'
    ).drop(columns=['min_date', 'clear_date'])

    payer_outstanding_groupby = payer_outstanding_df.groupby(['unique_payer_id'])

    payer_outstanding_df.issued_invoice_count = payer_outstanding_groupby.issued_invoice_count.ffill().fillna(0)
    payer_outstanding_df.issued_total_amt = payer_outstanding_groupby.issued_total_amt.ffill().fillna(0)
    payer_outstanding_df.collected_invoice_count = payer_outstanding_groupby.collected_invoice_count.ffill().fillna(0)
    payer_outstanding_df.collected_total_amt = payer_outstanding_groupby.collected_total_amt.ffill().fillna(0)

    payer_outstanding_df['payer_outstanding_invoices'] = payer_outstanding_df.issued_invoice_count \
        - payer_outstanding_df.collected_invoice_count
    payer_outstanding_df['payer_outstanding_dollars'] = payer_outstanding_df.issued_total_amt \
        - payer_outstanding_df.collected_total_amt

    payer_outstanding_df['next_day'] = payer_outstanding_df.join_date + pd.Timedelta(days=1)

    forecasting_df = forecasting_df.merge(
        payer_outstanding_df[[
            'unique_payer_id', 'next_day', 'payer_prev_days_to_pay', 'avg_payer_prev_days_to_pay',
            'std_payer_prev_days_to_pay',
            'payer_outstanding_invoices', 'payer_outstanding_dollars'
        ]],
        left_on=['unique_payer_id', 'invoice_date'],
        right_on=['unique_payer_id', 'next_day']
    ).drop(columns=['next_day'])

    forecasting_df = forecasting_df.sort_values(['supplier_number', 'clear_date']).reset_index().drop(columns=['index'])

    vendor_min_max_df = (forecasting_df[['supplier_number', 'clear_date', 'invoice_date']].groupby(['supplier_number'])
        .agg(
            clear_min=('clear_date', 'min'),
            invoice_min=('invoice_date', 'min'),
            clear_max=('clear_date', 'max'),
            invoice_max=('invoice_date', 'max')
        ))
    vendor_min_max_df['all_min'] = vendor_min_max_df[['clear_min', 'invoice_min']].T.min() - pd.Timedelta(days=1)  # type: ignore[misc]
    vendor_min_max_df['all_max'] = vendor_min_max_df[['clear_max', 'invoice_max']].T.max()  # type: ignore[misc]
    vendor_min_max_df = vendor_min_max_df[['all_min', 'all_max']].reset_index()  # type: ignore[misc]

    vendor_blank_dfs = []

    for _, row in vendor_min_max_df.iterrows():
        new_df = pd.DataFrame(pd.date_range(start=row['all_min'], end=row['all_max']), columns=['join_date'])  # type: ignore[misc]
        new_df['supplier_number'] = row['supplier_number']
        vendor_blank_dfs.append(new_df)

    vendor_date_join_df = pd.concat(vendor_blank_dfs).reset_index().drop(columns=['index'])

    vendor_daily_dupes = vendor_date_join_df.merge(
        forecasting_df[['supplier_number', 'clear_date', 'days_to_pay']],
        left_on=['supplier_number', 'join_date'],
        right_on=['supplier_number', 'clear_date'],
        how='left'
    ).drop(columns='clear_date').sort_values(['supplier_number', 'join_date'])

    vendor_prev_stats = vendor_daily_dupes.groupby(['supplier_number']) \
        .rolling(window='547d', on='join_date', closed='both').days_to_pay.agg(['mean', 'std']).round(1).reset_index()

    vendor_prev_stats = vendor_prev_stats[
        (vendor_prev_stats.index.isin(vendor_prev_stats.reset_index().groupby(['supplier_number', 'join_date'])
                                      ['index'].agg('max')))
    ].rename(columns={'mean': 'avg_vendor_prev_days_to_pay', 'std': 'std_vendor_prev_days_to_pay'})

    prev_days_df = vendor_prev_stats.merge(
        forecasting_df[['supplier_number', 'clear_date', 'days_to_pay']].groupby(
            ['supplier_number', 'clear_date']).days_to_pay.max()
        .reset_index(),
        left_on=['supplier_number', 'join_date'],
        right_on=['supplier_number', 'clear_date'],
        how='left'
    ).drop(columns=['clear_date'])

    prev_days_df.days_to_pay = prev_days_df.groupby(['supplier_number']).days_to_pay.ffill()
    prev_days_df = prev_days_df.rename(columns={'days_to_pay': 'vendor_prev_days_to_pay'})

    vendor_issued_df = forecasting_df[['supplier_number', 'invoice_date', 'clear_date', 'invoice_amt']].copy()
    vendor_issued_df['min_date'] = vendor_issued_df[['invoice_date', 'clear_date']].min(axis=1)

    vendor_issued_df = vendor_issued_df.groupby(['supplier_number', 'min_date']) \
        .invoice_amt.agg(['count', 'sum']).reset_index().sort_values(['supplier_number', 'min_date']) \
        .rename(columns={'count': 'invoice_count', 'sum': 'invoice_sum'})

    vendor_issued_groupby = vendor_issued_df.set_index('min_date').groupby(['supplier_number'])

    vendor_issued_df['issued_invoice_count'] = vendor_issued_groupby.invoice_count.expanding().sum().reset_index() \
        .invoice_count
    vendor_issued_df['issued_total_amt'] = vendor_issued_groupby.invoice_sum.expanding().sum().reset_index() \
        .invoice_sum

    vendor_collected_df = forecasting_df[['supplier_number', 'clear_date', 'invoice_amt']] \
        .groupby(['supplier_number', 'clear_date']).invoice_amt.agg(['count', 'sum']).reset_index() \
        .sort_values(['supplier_number', 'clear_date']).rename(columns={'count': 'invoice_count', 'sum': 'invoice_sum'})

    vendor_collected_groupby = vendor_collected_df.set_index('clear_date').groupby(['supplier_number'])

    vendor_collected_df['collected_invoice_count'] = vendor_collected_groupby.invoice_count.expanding().sum() \
        .reset_index().invoice_count
    vendor_collected_df['collected_total_amt'] = vendor_collected_groupby.invoice_sum.expanding().sum().reset_index() \
        .invoice_sum

    vendor_outstanding_df = prev_days_df.merge(
        vendor_issued_df[['supplier_number', 'min_date', 'issued_invoice_count', 'issued_total_amt']],
        left_on=['supplier_number', 'join_date'],
        right_on=['supplier_number', 'min_date'],
        how='left'
    ).merge(
        vendor_collected_df[['supplier_number', 'clear_date', 'collected_invoice_count', 'collected_total_amt']],
        left_on=['supplier_number', 'join_date'],
        right_on=['supplier_number', 'clear_date'],
        how='left'
    ).drop(columns=['min_date', 'clear_date'])

    vendor_outstanding_groupby = vendor_outstanding_df.groupby(['supplier_number'])

    vendor_outstanding_df.issued_invoice_count = vendor_outstanding_groupby.issued_invoice_count.ffill().fillna(0)
    vendor_outstanding_df.issued_total_amt = vendor_outstanding_groupby.issued_total_amt.ffill().fillna(0)
    vendor_outstanding_df.collected_invoice_count = vendor_outstanding_groupby.collected_invoice_count.ffill().fillna(0)
    vendor_outstanding_df.collected_total_amt = vendor_outstanding_groupby.collected_total_amt.ffill().fillna(0)

    vendor_outstanding_df['vendor_outstanding_invoices'] = vendor_outstanding_df.issued_invoice_count \
        - vendor_outstanding_df.collected_invoice_count
    vendor_outstanding_df['vendor_outstanding_dollars'] = vendor_outstanding_df.issued_total_amt \
        - vendor_outstanding_df.collected_total_amt

    vendor_outstanding_df['next_day'] = vendor_outstanding_df.join_date + pd.Timedelta(days=1)

    forecasting_df = forecasting_df.merge(
        vendor_outstanding_df[[
            'supplier_number', 'next_day', 'vendor_prev_days_to_pay', 'avg_vendor_prev_days_to_pay',
            'std_vendor_prev_days_to_pay',
            'vendor_outstanding_invoices', 'vendor_outstanding_dollars'
        ]],
        left_on=['supplier_number', 'invoice_date'],
        right_on=['supplier_number', 'next_day']
    ).drop(columns=['next_day'])

    forecasting_df['invoice_month'] = forecasting_df['invoice_date'].dt.month
    forecasting_df['invoice_dayofweek'] = forecasting_df['invoice_date'].dt.day_name()
    forecasting_df['is_us_holiday'] = forecasting_df['invoice_date'].isin(us_holidays).astype(int)

    month_dummies = pd.get_dummies(forecasting_df['invoice_month'], prefix='month')
    dow_dummies = pd.get_dummies(forecasting_df['invoice_dayofweek'], prefix='dow')

    forecasting_df = pd.concat([forecasting_df, month_dummies, dow_dummies], axis=1)

    return forecasting_df
