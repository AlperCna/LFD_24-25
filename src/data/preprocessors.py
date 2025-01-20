import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handle data preprocessing tasks."""

    def __init__(self):
        self.scalers = {}

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Linear interpolation for numeric columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        # Fill remaining NaN values with 0 as fallback
        df = df.fillna(0)
        return df

    def detect_and_interpolate_outliers(self, df: pd.DataFrame, method: str = 'iqr', order: int = 2) -> pd.DataFrame:

        numeric_cols = df.select_dtypes(include=[np.number]).columns  # Sayısal sütunları seç
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR

                # Aykırı değerleri tespit et
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

                # Aykırı değerleri interpolasyonla doldur
                if outliers.any():
                    df.loc[outliers, col] = df[col].interpolate(method='polynomial', order=order,
                                                                limit_direction='both')

        return df

    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """
        Scale features using the specified method.

        Args:
            df (pd.DataFrame): The dataframe to process.
            columns (List[str]): The columns to scale.
            method (str): The method to use for scaling ('standard').

        Returns:
            pd.DataFrame: The dataframe with scaled features.
        """
        if method == 'standard':
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            self.scalers['standard'] = scaler
        return df
