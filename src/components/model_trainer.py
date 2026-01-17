import os
import sys
from dataclasses import dataclass

from datetime import datetime

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model_lgb, evaluate_model_xgb, save_object, wape, train_and_evaluate_models
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "models")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_input, train_target, test_input, test_target, TARGET_COLS):
        try:
            logging.info("Get train feature input, train target, test feature input & test target")
            X_train, y_train, X_test, y_test = (
                train_input, 
                train_target, 
                test_input, 
                test_target
            )

            # 1. Baseline Performance (Lag-7 Seasonal Naive)
            logging.info("Calculating Seasonal Naive baseline (Lag-7)")
            # Using .values and repeating for the 7-day horizon
            baseline_preds = np.tile(X_test[['units_sold_lag_7']].values, (1, 7))
            
            # Calculate baseline error (e.g., WAPE or RMSE)
            # This serves as the 'minimum acceptable performance' threshold
            baseline_score = np.mean(np.abs(y_test.values - baseline_preds))
            logging.info(f"Baseline Mean Absolute Error: {baseline_score}")


            # 2. Model Definitions
            logging.info("Get the 'Light Gradient-Boosting Machine' & 'eXtreme Gradient Boosting' models as multi-output model")
            models = {
                "model_lgb": LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=8,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                    ),

                "model_xgb": XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    random_state=42
                )
            }


            # 3. Model Evaluation
            logging.info("fit, evaluate the models and get the best model")
            model_score = {
                "model_lgb": evaluate_model_lgb(models['model_lgb'], X_train, y_train, X_test, y_test, TARGET_COLS),
                "model_xgb": evaluate_model_xgb(models['model_xgb'], X_train, y_train, X_test, y_test, TARGET_COLS),
            }


            # 4. Identify Best Model
            best_model_score = min(list(model_score.values()))
            print(f'The best model score {best_model_score}')

            best_model_name = list(model_score.keys())[
                list(model_score.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            
            # 5. Threshold Validation
            # If the model isn't better than the baseline, it's not ready for production

            if best_model_score > baseline_score:
                logging.warning("No model outperformed the Seasonal Naive baseline.")
                raise CustomException("No best model found: Model performance is worse than baseline.")
            
            logging.info(f"Best Model found: {best_model_name} with score: {best_model_score}")

            # 6. Trains 7 independent best models which derived
            logging.info("Trains 7 independent LightGBM models where each model learns horizon-specific patterns")

            models, scores = train_and_evaluate_models(X_train, y_train, X_test, y_test, TARGET_COLS, best_model)

            # 7. Save the Best Model
            config = ModelTrainerConfig()
            run_date = datetime.today().strftime("%Y%m%d")

            for target, model in models.items():
                model_path = os.path.join(
                    config.trained_model_file_path,
                    f"demand_forecast_{target}_v1_{run_date}.pkl"
                )
                save_object(model_path, model)
                logging.info("Successfully saved {model} in {model_path}")
            

        except Exception as e:
            raise CustomException(e, sys)



