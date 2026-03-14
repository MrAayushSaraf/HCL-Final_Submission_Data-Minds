# 📈 Retail Sales Forecasting Engine

An end-to-end Machine Learning ecosystem designed to predict daily, weekly, and monthly retail sales based on hierarchical store data, time-series features, and macroeconomic indicators. Built primarily for the Kaggle *Store Sales - Time Series Forecasting* dataset (Corporación Favorita).

## 🚀 Key Features

*   **Robust ML Pipeline**: Implements advanced preprocessing (handling missing values, encoding, and scaling) and feature engineering (time-series extraction, holiday lookups, and macroeconomic oil price integration).
*   **Optimal Algorithm**: Uses a highly-tuned **Random Forest Regressor** to capture non-linear retail trends, significantly outperforming linear baselines.
*   **FastAPI Backend**: A lightweight, high-performance REST API that serves model inferences instantly and handles date-range aggregations.
*   **Streamlit Dashboard**: A beautifully designed, interactive UI allowing users to dynamically select stores (by authentic Ecuadorian branch names), product families, and dates to visualize **Estimated Revenue Projections**.
*   **Smart Automation**: The system auto-fills complex inputs like rolling average store transactions, WTI crude oil prices, and local/national holiday flags, reducing user input friction to just 5 essential parameters.

---

## 📂 Project Architecture

```text
sales_forecasting/
├── app/
│   ├── api.py                   # FastAPI application serving ML inferences
│   └── ui.py                    # Streamlit Dashboard for user interaction
├── data/
│   └── store-sales/             # Raw Kaggle CSV datasets
├── models/
│   ├── best_model.pkl           # Serialized Random Forest model
│   ├── preprocessor.pkl         # Serialized Scaler and Encoder
│   └── *.json                   # Cached lookups (oil prices, holidays, avg transactions)
├── notebooks/
│   └── eda.py                   # Exploratory Data Analysis & visual generation
├── reports/
│   └── figures/                 # Saved EDA and Feature Importance charts
└── src/
    ├── data/
    │   └── loader.py            # Data ingestion and schema validation
    ├── evaluation/
    │   ├── metrics.py           # RMSE, MAE, R² calculators
    │   └── visuals.py           # Feature importance plotting
    ├── features/
    │   └── preprocessor.py      # Master data transformation pipeline
    └── models/
        └── train.py             # Model training, evaluation, orchestration
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/retail-sales-forecasting.git
   cd retail-sales-forecasting
   ```

2. **Set up the Python Environment:**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit fastapi uvicorn matplotlib seaborn requests joblib
   ```

3. **Data Preparation:**
   Download the [Store Sales - Time Series Forecasting dataset from Kaggle](https://www.kaggle.com/c/store-sales-time-series-forecasting/data) and place the CSV files (`train.csv`, `stores.csv`, `oil.csv`, `transactions.csv`, `holidays_events.csv`) into the `data/store-sales/` directory.

4. **Train the Model:**
   Generate the machine learning models and cache files:
   ```bash
   python src/models/train.py
   ```

---

## 🖥️ Usage

This project uses a decoupled architecture. You need to run both the backend (API) and the frontend (UI) simultaneously in two separate terminal windows.

### Step 1: Start the API Server
In your first terminal, start the FastAPI backend:
```bash
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```
*The API handles all the heavy machine learning computation and will be available locally at `http://localhost:8000`.*

### Step 2: Launch the UI Dashboard
In your second terminal, start the Streamlit frontend:
```bash
streamlit run app/ui.py
```
*This will automatically open the interactive dashboard in your default web browser.*

---

## 📊 Model Performance

During our automated algorithm evaluation, multiple models were tested. The **Random Forest Regressor** was selected for production due to its superior capability in handling non-linear retail trends and minimizing error.

| Algorithm | Mean Absolute Error (MAE) | R² Score |
| :--- | :--- | :--- |
| Baseline (Linear Regression) | 409.93 | 24.79% |
| **Production (Random Forest)** | **101.60** | **90.14%** |
| XGBoost | 108.62 | 90.34% |

---

## 🔮 Future Enhancements
- [ ] Implement robust NLP integration to analyze the free-text `holiday_desc` column instead of completely dropping it.
- [ ] Add Docker configuration (`Dockerfile` & `docker-compose.yml`) for containerized orchestration.
- [ ] Add `pytest` automation scripts for CI/CD integration.
