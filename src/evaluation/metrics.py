from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import logging

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculates RMSE, MAE, and R2 score for given predictions.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    logger.info(f"Results for {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    return metrics
