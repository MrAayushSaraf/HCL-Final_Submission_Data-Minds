import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import joblib

def plot_feature_importance():
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train the model first.")
        return
        
    model = joblib.load(model_path)
    
    # Check if the model has feature_importances_
    if hasattr(model, 'feature_importances_'):
        # Usually from Tree models (RF, XGBoost)
        importances = model.feature_importances_
        # Feature names should technically be extracted from preprocessing step,
        # but for simplicity, we fallback to numbered generic names if we can't extract them.
        feature_names = getattr(model, 'feature_names_in_', [f"Feature {i}" for i in range(len(importances))])
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        top_features = 15 # show top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importances (Top 15)')
        plt.bar(range(min(top_features, len(importances))), importances[indices][:top_features], align="center")
        plt.xticks(range(min(top_features, len(importances))), [feature_names[i] for i in indices][:top_features], rotation=45, ha='right')
        plt.xlim([-1, min(top_features, len(importances))])
        plt.tight_layout()
        
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig('reports/figures/feature_importance.png')
        plt.close()
        print("Feature importance plot saved to reports/figures/feature_importance.png")
    else:
        print("The loaded model doesn't support generic feature_importances_ plotting (e.g. Ridge).")

if __name__ == "__main__":
    plot_feature_importance()
