import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data.loader import DataLoader
from src.features.preprocessor import DataPreprocessor

# Ensure figures output directory exists
os.makedirs("reports/figures", exist_ok=True)

def perform_eda():
    print("Loading data for EDA...")
    loader = DataLoader("data/store-sales")
    
    # We load a subset of train data for EDA to save memory
    train_df = loader.load_csv("train.csv", parse_dates=['date'])
    # Downsample for EDA
    train_df = train_df.sample(frac=0.1, random_state=42)
    
    stores_df = loader.load_csv("stores.csv")
    oil_df = loader.load_csv("oil.csv", parse_dates=['date'])
    holidays_df = loader.load_csv("holidays_events.csv", parse_dates=['date'])
    transactions_df = loader.load_csv("transactions.csv", parse_dates=['date'])
    
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(train_df, stores_df, oil_df, transactions_df, holidays_df)
    
    print("Generating Sales Distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['sales'] > 0]['sales'], bins=50, kde=True, log_scale=True)
    plt.title('Log-Scaled Distribution of Sales (Excluding Zeros)')
    plt.xlabel('Sales (Log Scale)')
    plt.ylabel('Frequency')
    plt.savefig('reports/figures/sales_distribution.png')
    plt.close()
    
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    # Select only numeric columns for correlation
    num_cols = df.select_dtypes(include=['number']).columns
    corr_matrix = df[num_cols].corr()
    
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png')
    plt.close()
    
    print("EDA Visualizations saved to reports/figures/")

if __name__ == "__main__":
    perform_eda()
