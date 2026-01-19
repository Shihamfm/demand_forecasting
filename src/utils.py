import os
import sys
import pickle

import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

import warnings
warnings.filterwarnings("ignore")

# Reindexing Function to handle missing dates
def clean_and_reindex(group):
    try:

        group = group.set_index('date')
        full_range = pd.date_range(start=group.index.min(), end=group.index.max(), freq='D')
        group = group.reindex(full_range)
        
        # Impute missing values
        group['units_sold'] = group['units_sold'].fillna(0)
        group['revenue'] = group['revenue'].fillna(0)
        group['promotion_discount_pct'] = group['promotion_discount_pct'].fillna(0)
        group['promotion_type'] = group['promotion_type'].fillna('no_promotion')
        group['avg_price'] = group['avg_price'].ffill().bfill() # Carry price forward
        group['store_id'] = group['store_id'].ffill().bfill()
        group['category'] = group['category'].ffill().bfill()
        group['sub_category'] = group['sub_category'].ffill().bfill()
        group['footfall'] = group['footfall'].ffill().bfill()
        group['item_id'] = group['item_id'].ffill().bfill()

        return group.reset_index().rename(columns={'index': 'date'})
    
    except Exception as e:
        raise CustomException(e, sys)

    

def data_preprocessing(df):
    try:

        logging.info("Creating date, store_id and item level values")
        df['month'] = df['date'].dt.month

        daily_df = (
        df.groupby(["store_id", "item_id", "date"], as_index=False)
        .agg({
            "units_sold": "sum",   # critical
            "revenue": "max",
            "avg_price": "mean",
            "footfall": "mean",
            "promotion_discount_pct": "mean",
            "is_holiday": "max",
            "category": "first",
            "sub_category": "first",
            "is_weekend": "first",
            "promotion_type": "first",
            "holiday_type": "first",
            "weather_condition": "first",
            "temperature_c": "first",
            "month": "first"
        })
    )
        logging.info("Get the date range")
        
        all_dates = pd.date_range(
        start=daily_df['date'].min(),
        end=daily_df['date'].max(),
        freq='D'
    )

        stores = daily_df['store_id'].unique()
        items = daily_df['item_id'].unique()

        full_index = pd.MultiIndex.from_product(
            [all_dates, stores, items],
            names=['date', 'store_id', 'item_id']
        )

        df_full = (
        df
        .set_index(['date', 'store_id', 'item_id'])
        .reindex(full_index)
        .reset_index()
    )
        
        logging.info("Define the calendar features")

        df_calendar = df_full.groupby("date", as_index=False).agg({
        'is_weekend': "first",
        'is_holiday': "first",
        'holiday_type': "first",
        'weather_condition': "first", 
        'temperature_c': "first",
        "month": "first"
    })
        logging.info("Reindexing Function to handle missing dates")
        
        df_filled = df_full.groupby(['store_id', 'item_id'], group_keys=False).apply(clean_and_reindex)

        df_filled = df_filled.drop(columns=['is_weekend', 'is_holiday', 'holiday_type','weather_condition', 'temperature_c', 'month'], axis=1)

        df_filled = df_filled.merge(
            df_calendar,
            on = 'date',
            how = 'left')
        
        logging.info("Create timeseries feature engineering")

        df_filled['day_of_week'] = df_filled["date"].dt.dayofweek
        df_filled['week_of_year'] = df_filled["date"].dt.isocalendar().week.astype(int)
        df_filled['is_month_start'] = df_filled['date'].dt.is_month_start.astype(int)
        df_filled['is_month_end'] = df_filled['date'].dt.is_month_end.astype(int)

        logging.info("Sort & Group")
        df_filled = df_filled.sort_values(
            ['store_id', 'item_id', 'date']
        ).reset_index(drop=True)

        
        logging.info("Apply lag features")
        LAGS = [1, 3, 7, 14, 28]

        for lag in LAGS:
            df_filled[f'units_sold_lag_{lag}'] = (
                df_filled
                .groupby(['store_id', 'item_id'])['units_sold']
                .shift(lag)
            )
        
        # Rolling Window Statistics for stability and trend

        logging.info("Apply rolling means")
        # Rolling means
        WINDOWS = [7, 14, 28]

        for w in WINDOWS:
            df_filled[f'rolling_mean_{w}'] = (
                df_filled
                .groupby(['store_id', 'item_id'])['units_sold']
                .shift(1)
                .rolling(w)
                .mean()
            )
        
        logging.info("Apply rolling volatality")

        for w in WINDOWS:
            df_filled[f'rolling_std_{w}'] = (
                df_filled
                .groupby(['store_id', 'item_id'])['units_sold']
                .shift(1)
                .rolling(w)
                .std()
            )

        logging.info("Apply Rolling max (captures spikes)")
        df_filled['rolling_max_7'] = (
            df_filled
            .groupby(['store_id', 'item_id'])['units_sold']
            .shift(1)
            .rolling(7)
            .max()
        )

        logging.info("Get intermediate demand features")
        # Zero-sales flag
        df_filled['zero_sales_flag'] = (df_filled['units_sold'] == 0).astype(int)

        logging.info("Get price & promotion dynamics")
        # Price & Promotion Dynamics
        df_filled['price_lag_7'] = (
            df_filled
            .groupby(['store_id', 'item_id'])['avg_price']
            .shift(7)
        )

        logging.info("Get Relative price change")
        # Relative price change
        df_filled['price_change_pct_7'] = (
            (df_filled['avg_price'] - df_filled['price_lag_7'])
            / df_filled['price_lag_7']
        )

        logging.info("Get promotion frequency")
        # Promotion frequency (last 7 days)
        df_filled['promo_flag'] = (
            df_filled['promotion_discount_pct'] > 0
        ).astype(int)

        df_filled['promo_days_last_7'] = (
            df_filled
            .groupby(['store_id', 'item_id'])['promo_flag']
            .shift(1)
            .rolling(7)
            .sum()
        )

        logging.info("Create calendar cyclin encoding")
        # Calendar Cyclic Encoding

        # Day of week
        df_filled['dow_sin'] = np.sin(2 * np.pi * df_filled['day_of_week'] / 7)
        df_filled['dow_cos'] = np.cos(2 * np.pi * df_filled['day_of_week'] / 7)

        # Month
        df_filled['month_sin'] = np.sin(2 * np.pi * df_filled['month'] / 12)
        df_filled['month_cos'] = np.cos(2 * np.pi * df_filled['month'] / 12)


        logging.info("Create 7-Day Forecast Targets (Multi-Horizon)")
        # Create 7-Day Forecast Targets (Multi-Horizon)

        for h in range(1, 8):
            df_filled[f'y_t_plus_{h}'] = (
                df_filled
                .groupby(['store_id', 'item_id'])['units_sold']
                .shift(-h)
            )

        df_preprocessed = df_filled.dropna().reset_index(drop=True)

        logging.info("Data preprocessing is completed")

        return df_preprocessed
        
    except Exception as e:
        raise CustomException(e, sys)
    

def train_test_split(df, train_cutoff_pct):
    # Convert the column to datetime
    df['date'] = pd.to_datetime(df['date'])

    train_cutoff = df['date'].quantile(train_cutoff_pct)

    train_df = df[df['date'] <= train_cutoff]
    test_df   = df[df['date'] > train_cutoff]

    return train_df, test_df

# WAPE (Weighted Absolute Percentage Error)
def wape(y_true, y_pred):
    '''
    Calculating the Weighted Absolute Percentage Error WAPE
    '''
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

def evaluate_model_lgb(model_lgb, X_train, y_train, X_test, y_test, TARGET_COLS):
    import lightgbm as lgb
    models_lgb = {}
    preds_lgb = {}
    wape_lgb = []

    for i, target in enumerate(TARGET_COLS):
        model_lgb.fit(
            X_train, y_train[target],
            eval_set=[(X_test, y_test[target])],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(50)],
        )

        models_lgb[target] = model_lgb
        preds_lgb[target] = model_lgb.predict(X_test)
        
    for target in TARGET_COLS:
        score_lgb = wape(y_test[target].values, preds_lgb[target])
        wape_lgb.append(score_lgb)
        #print(f"{target} WAPE: {score_lgb:.3f} for LGBM")

    return np.mean(wape_lgb)    

def evaluate_model_xgb(model_xgb, X_train, y_train, X_test, y_test, TARGET_COLS):
    models_xgb = {}
    preds_xgb = {}
    wape_xgb = []

    for i, target in enumerate(TARGET_COLS):
        model_xgb.fit(
            X_train, y_train[target],
            eval_set=[(X_test, y_test[target])],
            verbose=False
        )

        models_xgb[target] = model_xgb
        preds_xgb[target] = model_xgb.predict(X_test)
        
    for target in TARGET_COLS:
        score_xgb = wape(y_test[target].values, preds_xgb[target])
        wape_xgb.append(score_xgb)
        #print(f"{target} WAPE: {score_xgb:.3f} for XGBoost")
    
    return np.mean(wape_xgb)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def train_and_evaluate_models(X_train, y_train, X_test, y_test, TARGET_COLS, best_model):
    '''
    From the chosen best model derive seperate seven models for each day
    '''
    models = {}
    scores = {}

    for target in TARGET_COLS:
        model = best_model
        model.fit(X_train, y_train[target])

        preds = model.predict(X_test)
        score = wape(y_test[target].values, preds)

        models[target] = model
        scores[target] = score

        logging.info(f"{target} WAPE: {score}")
        print(f"{target} WAPE: {score:.3f}")

    return models, scores



if __name__ == "__main__":
    print("Hello")
    df_preprocessed = data_preprocessing("E:/Projects/DS/Demand_Forecasting/demand_forecasting/notebook/data/walmart_daily_sales_2025_realistic.csv")
    print(df_preprocessed.head())

