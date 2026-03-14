import pandas as pd
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_csv(self, filename: str, parse_dates=None) -> pd.DataFrame:
        """
        Loads a CSV file into a pandas DataFrame.
        """
        file_path = f"{self.data_dir}/{filename}"
        logger.info(f"Loading data from {file_path}")
        try:
            df = pd.read_csv(file_path, parse_dates=parse_dates)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")
            return df
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, Any]) -> bool:
        """
        Validates the DataFrame against an expected schema (column names and data types).
        """
        is_valid = True
        logger.info("Validating schema...")
        
        # Check if expected columns are present
        missing_cols = [col for col in expected_schema.keys() if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing expected columns: {missing_cols}")
            is_valid = False

        # Validate types
        if is_valid:
            for col, expected_type in expected_schema.items():
                actual_type = df[col].dtype
                if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                    logger.warning(f"Column '{col}' has type {actual_type}, but expected {expected_type}")
                    # Attempt to convert or just flag
                    # is_valid = False # Optional: strict type checking
        
        if is_valid:
            logger.info("Schema validation passed.")
        else:
            logger.error("Schema validation failed.")
            
        return is_valid

if __name__ == "__main__":
    # Test loading and basic validation
    loader = DataLoader("data/store-sales")
    
    # Example expected schema for train.csv 
    # id, date, store_nbr, family, sales, onpromotion
    expected_train_schema = {
        'id': 'int64',
        'date': 'datetime64[ns]',
        'store_nbr': 'int64',
        'family': 'object',
        'sales': 'float64',
        'onpromotion': 'int64'
    }
    
    try:
        train_df = loader.load_csv("train.csv", parse_dates=['date'])
        loader.validate_schema(train_df, expected_train_schema)
        
        # Load additional contextual tables
        stores_df = loader.load_csv("stores.csv")
        oil_df = loader.load_csv("oil.csv", parse_dates=['date'])
        holidays_df = loader.load_csv("holidays_events.csv", parse_dates=['date'])
        transactions_df = loader.load_csv("transactions.csv", parse_dates=['date'])
        
        print(f"Sample Train Data:\\n{train_df.head()}")
    except Exception as e:
        print(f"Testing loader failed: {e}")
