from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory for module loading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.features.preprocessor import DataPreprocessor

app = FastAPI(title="Store Sales Forecasting API", description="API to predict sales using the best performing regression model.")

# Global variables to hold model and preprocessor
model = None
preprocessor = None
store_avg_transactions = {}  # per-store average transactions lookup
last_oil_price = 50.0        # last known oil price fallback
holidays_lookup = {}         # date -> {holiday_type, holiday_transferred}

# Model inputs defined by a simple schema for online inference
class SalesPredictionRequest(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int
    date: str # format YYYY-MM-DD
    # Stores Context
    city: str
    state: str
    type: str # store type A-E
    cluster: int
    
class SalesPredictionResponse(BaseModel):
    predicted_sales: float
    model_used: str

class ForecastRequest(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int
    start_date: str  # format YYYY-MM-DD
    end_date: str    # format YYYY-MM-DD
    city: str
    state: str
    type: str
    cluster: int

class DailyForecast(BaseModel):
    date: str
    predicted_sales: float

class ForecastResponse(BaseModel):
    forecasts: list
    total_sales: float
    model_used: str

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, store_avg_transactions, last_oil_price, holidays_lookup
    import json
    model_path = "models/best_model.pkl"
    prep_path = "models/preprocessor.pkl"
    avg_tx_path = "models/store_avg_transactions.json"
    oil_path = "models/last_oil_price.json"
    holidays_path = "models/holidays_lookup.json"
    
    if os.path.exists(model_path) and os.path.exists(prep_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        print("Model and Preprocessor loaded successfully.")
    else:
        print("Warning: Model or Preprocessor not found. Train the model first.")

    if os.path.exists(avg_tx_path):
        with open(avg_tx_path) as f:
            store_avg_transactions = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded average transactions for {len(store_avg_transactions)} stores.")
    else:
        print("Warning: store_avg_transactions.json not found. Re-run training to generate it.")

    if os.path.exists(oil_path):
        with open(oil_path) as f:
            last_oil_price = json.load(f).get("last_oil_price", 50.0)
        print(f"Loaded last oil price: {last_oil_price:.2f}")
    else:
        print("Warning: last_oil_price.json not found. Using default $50.0.")

    if os.path.exists(holidays_path):
        with open(holidays_path) as f:
            holidays_lookup = json.load(f)
        print(f"Loaded {len(holidays_lookup)} holiday entries.")
    else:
        print("Warning: holidays_lookup.json not found. All dates will be treated as non-holiday.")

@app.get("/")
def home():
    return {"message": "Welcome to the Store Sales Forecasting API. Use /predict for inference."}

@app.post("/predict", response_model=SalesPredictionResponse)
def predict_sales(request: SalesPredictionRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Convert request dict to DataFrame
        data = request.dict()
        
        # Auto-fill transactions from historical store average
        store_nbr = data['store_nbr']
        global_avg_tx = sum(store_avg_transactions.values()) / len(store_avg_transactions) if store_avg_transactions else 1500.0
        data['transactions'] = store_avg_transactions.get(store_nbr, global_avg_tx)
        
        # Auto-fill oil price from last known historical value
        data['dcoilwtico'] = last_oil_price
        
        # Auto-fill holiday type from date-based lookup
        date_key = data['date']
        holiday_info = holidays_lookup.get(date_key, {"holiday_type": "None", "holiday_transferred": False})
        data['holiday_type'] = holiday_info.get('holiday_type', 'None')
        data['holiday_transferred'] = holiday_info.get('holiday_transferred', False)
        
        df = pd.DataFrame([data])
        
        # Format the 'date' field to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Because our preprocessor was designed for batch merging,
        # we bypass the merge_datasets step explicitly, as the API receives 
        # a fully denormalized record via JSON payload.
        # So we manually sequence: create_time_features -> encode -> scale
        
        df = preprocessor.create_time_features(df)
        df = preprocessor.encode_categorical(df, is_train=False)
        df = preprocessor.scale_numerical(df, is_train=False)
        
        # Drop columns not needed (must match training exactly)
        drop_cols = ['date', 'holiday_locale', 'holiday_locale_name', 'holiday_desc', 'id']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        # To prevent missing columns from breaking prediction, ensure column alignment
        # This occurs if order matters. Tree models rely on order for numpy prediction.
        expected_cols = getattr(model, 'feature_names_in_', None)
        if expected_cols is not None:
            # Reorder columns to match train set
            df = df[expected_cols]
            
        # Predict
        prediction = model.predict(df)[0]
        prediction = max(0, prediction) # sales cannot be negative
        
        return SalesPredictionResponse(
            predicted_sales=float(prediction),
            model_used=type(model).__name__
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastResponse)
def forecast_sales(request: ForecastRequest):
    """Predicts daily sales for every day in a given date range, then aggregates."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        from datetime import datetime, timedelta
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")

        if end < start:
            raise HTTPException(status_code=400, detail="end_date must be after start_date.")
        if (end - start).days > 366:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 366 days.")

        store_nbr = request.store_nbr
        global_avg_tx = sum(store_avg_transactions.values()) / len(store_avg_transactions) if store_avg_transactions else 1500.0

        rows = []
        date_labels = []
        current = start
        while current <= end:
            date_key = current.strftime("%Y-%m-%d")
            holiday_info = holidays_lookup.get(date_key, {"holiday_type": "None", "holiday_transferred": False})

            row = {
                "store_nbr": store_nbr,
                "family": request.family,
                "onpromotion": request.onpromotion,
                "date": current,
                "transactions": store_avg_transactions.get(store_nbr, global_avg_tx),
                "dcoilwtico": last_oil_price,
                "holiday_type": holiday_info.get("holiday_type", "None"),
                "holiday_transferred": holiday_info.get("holiday_transferred", False),
                "city": request.city,
                "state": request.state,
                "type": request.type,
                "cluster": request.cluster,
            }
            rows.append(row)
            date_labels.append(date_key)
            current += timedelta(days=1)

        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = preprocessor.create_time_features(df)
        df = preprocessor.encode_categorical(df, is_train=False)
        df = preprocessor.scale_numerical(df, is_train=False)

        drop_cols = ['date', 'holiday_locale', 'holiday_locale_name', 'holiday_desc', 'id']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        expected_cols = getattr(model, 'feature_names_in_', None)
        if expected_cols is not None:
            df = df[expected_cols]

        predictions = model.predict(df)
        predictions = np.maximum(predictions, 0)

        forecasts = [{"date": date_labels[i], "predicted_sales": float(predictions[i])} for i in range(len(predictions))]
        total = float(predictions.sum())

        return ForecastResponse(
            forecasts=forecasts,
            total_sales=total,
            model_used=type(model).__name__
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
