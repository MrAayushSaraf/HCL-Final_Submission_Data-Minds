import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.loader import DataLoader
from src.features.preprocessor import DataPreprocessor
from src.evaluation.metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_pipeline(data_dir="data/store-sales", sample_frac=0.1):
    """
    Executes the training pipeline from loading data to saving the model.
    Using sample_frac to train on a fraction of data for speed in testing.
    """
    logger.info("Initializing Data Loader...")
    loader = DataLoader(data_dir)
    
    # Load data
    train_df = loader.load_csv("train.csv", parse_dates=['date'])
    if sample_frac < 1.0:
        logger.info(f"Sampling {sample_frac*100}% of the data for faster iteration.")
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        
    stores_df = loader.load_csv("stores.csv")
    oil_df = loader.load_csv("oil.csv", parse_dates=['date'])
    holidays_df = loader.load_csv("holidays_events.csv", parse_dates=['date'])
    transactions_df = loader.load_csv("transactions.csv", parse_dates=['date'])
    
    logger.info("Preprocessing Data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(train_df, stores_df, oil_df, transactions_df, holidays_df, is_train=True)
    
    logger.info("Preparing Features and Target...")
    # Target variable is 'sales'
    X = df.drop(columns=['sales', 'id'])
    y = df['sales']
    
    # Fill any remaining NaNs (e.g., from lags or specific merges)
    X = X.fillna(0)
    
    logger.info("Splitting Data into Train and Validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    # Model 1a: Linear Regression (Baseline 1)
    logger.info("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_val)
    lr_preds = np.maximum(lr_preds, 0)
    results['LinearRegression'] = evaluate_model(y_val, lr_preds, "Linear Regression")

    # Model 1b: Ridge Regression (Baseline 2)
    logger.info("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_val)
    ridge_preds = np.maximum(ridge_preds, 0)
    results['Ridge'] = evaluate_model(y_val, ridge_preds, "Ridge Regression")

    # Model 1b: Lasso Regression (Baseline 2)
    logger.info("Training Lasso Regression...")
    lasso = Lasso(alpha=1.0, max_iter=2000)
    lasso.fit(X_train, y_train)
    lasso_preds = lasso.predict(X_val)
    lasso_preds = np.maximum(lasso_preds, 0)
    results['Lasso'] = evaluate_model(y_val, lasso_preds, "Lasso Regression")
    
    # Model 2: RandomForest
    logger.info("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    rf_preds = np.maximum(rf_preds, 0)
    results['RandomForest'] = evaluate_model(y_val, rf_preds, "Random Forest")
    
    # Model 3: XGBoost
    logger.info("Training XGBoost Regressor...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_val)
    xgb_preds = np.maximum(xgb_preds, 0)
    results['XGBoost'] = evaluate_model(y_val, xgb_preds, "XGBoost")
    
    # Select best model based on MAE
    best_model_name = min(results, key=lambda k: results[k]['mae'])
    logger.info(f"Best Model Selected: {best_model_name}")
    
    if best_model_name == 'LinearRegression':
        best_model = lr
    elif best_model_name == 'Ridge':
        best_model = ridge
    elif best_model_name == 'Lasso':
        best_model = lasso
    elif best_model_name == 'RandomForest':
        best_model = rf
    else:
        best_model = xgb
        
    # Serialize model and preprocessor
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.info("Saved best model and preprocessor to models/ directory.")

    # Save per-store average transactions for use by the API (Option B)
    full_transactions_df = loader.load_csv("transactions.csv", parse_dates=['date'])
    store_avg_tx = full_transactions_df.groupby("store_nbr")["transactions"].mean().round(2)
    store_avg_tx.to_json("models/store_avg_transactions.json")
    logger.info("Saved per-store average transactions to models/store_avg_transactions.json.")

    # Save last known oil price (forward-fill friendly)
    full_oil_df = loader.load_csv("oil.csv", parse_dates=['date'])
    last_oil_price = float(full_oil_df['dcoilwtico'].dropna().iloc[-1])
    import json
    with open("models/last_oil_price.json", "w") as f:
        json.dump({"last_oil_price": last_oil_price}, f)
    logger.info(f"Saved last known oil price: {last_oil_price:.2f} to models/last_oil_price.json.")

    # Save holidays lookup indexed by date string
    full_holidays_df = loader.load_csv("holidays_events.csv", parse_dates=['date'])
    holidays_lookup = {}
    for _, row in full_holidays_df.iterrows():
        date_key = row['date'].strftime("%Y-%m-%d")
        holidays_lookup[date_key] = {
            "holiday_type": row.get('type', 'None'),
            "holiday_transferred": bool(row.get('transferred', False))
        }
    with open("models/holidays_lookup.json", "w") as f:
        json.dump(holidays_lookup, f)
    logger.info(f"Saved {len(holidays_lookup)} holiday entries to models/holidays_lookup.json.")
    
    print("\n--- Summary of Results ---")
    for r in results.values():
        print(f"{r['model']}: RMSE={r['rmse']:.2f}, MAE={r['mae']:.2f}, R2={r['r2']:.4f}")
        
if __name__ == "__main__":
    # For a hackathon timeline natively on a PC, 10-20% data sampled is practical for initial runs
    run_training_pipeline(sample_frac=0.1)
