import sys
import os
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

import numpy as np
import pandas as pd

class DataTransformation:
    def __init__(self):
         
        self.FEATURE_COLS = [
            # Lag features
            'units_sold_lag_1', 'units_sold_lag_3', 'units_sold_lag_7',
            'units_sold_lag_14', 'units_sold_lag_28',

            # Rolling stats
            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
            'rolling_std_7', 'rolling_std_14', 'rolling_std_28',
            'rolling_max_7',

            # Intermittent demand
            'zero_sales_flag',

            # Price & promo
            'avg_price', 'price_change_pct_7',
            'promotion_discount_pct', 'promo_days_last_7',

            # Calendar
            'dow_sin', 'dow_cos',
            'month_sin', 'month_cos',
            'is_weekend', 'is_holiday'
        ]
            # Targets (7-day horizon)
        self.TARGET_COLS = [f'y_t_plus_{i}' for i in range(1, 8)]


    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info(f"Reading data from {train_path} and {test_path}")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            X_train = train_df[self.FEATURE_COLS]
            y_train = train_df[self.TARGET_COLS]

            X_test = test_df[self.FEATURE_COLS]
            y_test = test_df[self.TARGET_COLS]

            logging.info("Applying Features and next 7 days target on trainig dataframe and testing dataframe")

            target_columns = self.TARGET_COLS

            return(
                X_train,
                y_train,
                X_test,
                y_test,
                target_columns
            )

        except Exception as e:
            raise CustomException(e, sys)
