import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def merge_datasets(self, train, stores, oil, transactions, holidays):
        """Merges relational datasets based on their common keys for store sales forecasing."""
        logger.info("Merging datasets...")
        
        # Merge with stores
        df = train.merge(stores, on='store_nbr', how='left')
        
        # Merge with transactions
        df = df.merge(transactions, on=['date', 'store_nbr'], how='left')
        # Fill missing transactions with 0
        df['transactions'] = df['transactions'].fillna(0)
        
        # Merge with oil
        df = df.merge(oil, on='date', how='left')
        
        # Merge with holidays (simplification: just take the first event for a given date)
        holidays_unique = holidays.drop_duplicates(subset=['date'])
        
        # We rename columns to prevent collisions
        holidays_unique = holidays_unique.rename(columns={
            'type': 'holiday_type', 'locale': 'holiday_locale',
            'locale_name': 'holiday_locale_name', 'description': 'holiday_desc',
            'transferred': 'holiday_transferred'
        })
        
        df = df.merge(holidays_unique, on='date', how='left')
        
        # Fill missing holiday types as 'None'
        df['holiday_type'] = df['holiday_type'].fillna('None')
        df['holiday_transferred'] = df['holiday_transferred'].fillna(False)

        # Impute missing oil values using forward and backward fill (as oil prices don't update on weekends)
        df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()
        
        logger.info(f"Merged memory usage: {df.memory_usage().sum() / 1e9:.2f} GB")
        return df

    def create_time_features(self, df):
        """Extracts date-related features."""
        logger.info("Extracting time features...")
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['quarter'] = df['date'].dt.quarter
        # Is payday assuming 15th and end of month
        df['is_payday'] = ((df['day_of_month'] == 15) | (df['date'].dt.is_month_end)).astype(int)
        return df

    def encode_categorical(self, df, is_train=True):
        """Encodes categorical columns."""
        logger.info("Encoding categorical columns...")
        cat_cols = ['family', 'city', 'state', 'type', 'holiday_type']
        
        # Convert non-string columns just to be safe
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        if is_train:
            encoded_vals = self.encoder.fit_transform(df[cat_cols])
        else:
            encoded_vals = self.encoder.transform(df[cat_cols])
            
        df[cat_cols] = encoded_vals
        return df

    def scale_numerical(self, df, is_train=True):
        """Standardizes numerical columns like oil price and transactions."""
        logger.info("Scaling numerical columns...")
        num_cols = ['transactions', 'dcoilwtico']
        
        if is_train:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
            
        return df

    def preprocess(self, train, stores, oil, transactions, holidays, is_train=True):
        """Full pipeline execution."""
        df = self.merge_datasets(train, stores, oil, transactions, holidays)
        df = self.create_time_features(df)
        df = self.encode_categorical(df, is_train)
        df = self.scale_numerical(df, is_train)
        
        # Drop columns not needed for modeling
        drop_cols = ['date', 'holiday_locale', 'holiday_locale_name', 'holiday_desc']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        logger.info("Preprocessing complete.")
        return df
