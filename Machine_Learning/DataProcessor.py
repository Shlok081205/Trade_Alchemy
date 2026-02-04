import numpy as np
import pandas as pd
from typing import Tuple, List


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass


class FeatureCalculator:
    def __init__(self):
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    def validate_input(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input dataframe before processing"""
        errors = []
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors

        missing = set(self.required_columns) - set(df.columns)
        if missing and 'AdjClose' not in df.columns and 'Close' not in df.columns:
            errors.append(f"Missing critical columns: {missing}")

        if len(df) < 200:
            errors.append(f"Need at least 200 rows, got {len(df)}")

        return len(errors) == 0, errors

    def calculate_features(self, df, threshold=0.002, regime="volatile"):
        """
        Calculates features and dynamic targets based on market regime.

        Args:
            df: Input OHLCV DataFrame
            threshold: Fixed % used if regime is 'volatile'
            regime: 'stable' or 'volatile' (from Gemini)
        """
        is_valid, errors = self.validate_input(df)
        if not is_valid:
            raise DataValidationError(f"Invalid input: {'; '.join(errors)}")

        try:
            df = df.copy()

            # Ensure price columns exist
            if 'AdjClose' not in df.columns:
                df['AdjClose'] = df['Close'] if 'Close' in df.columns else None

            # Basic returns
            df['Ret'] = df['AdjClose'].pct_change()

            # ============= VOLATILITY FEATURES =============
            # ATR (14-day)
            hl = df['High'] - df['Low']
            hc = np.abs(df['High'] - df['AdjClose'].shift())
            lc = np.abs(df['Low'] - df['AdjClose'].shift())
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
            df['Vol_Regime'] = df['ATR'] / df['ATR'].rolling(100).mean()

            # ============= MOMENTUM & TECHNICALS =============
            # RSI
            delta = df['AdjClose'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = df['AdjClose'].ewm(span=12).mean()
            ema26 = df['AdjClose'].ewm(span=26).mean()
            macd = ema12 - ema26
            df['MACD_Hist'] = macd - macd.ewm(span=9).mean()

            # Moving Average Distances
            df['MA_20'] = df['AdjClose'].rolling(20).mean()
            df['MA_50'] = df['AdjClose'].rolling(50).mean()
            df['Dist_MA50'] = (df['AdjClose'] - df['MA_50']) / df['MA_50']

            # ============= REGIME-BASED TARGET LOGIC =============
            # Shift returns to get 'tomorrow's' result for today's row
            df['Next_Ret'] = df['Ret'].shift(-1)

            if regime == "stable":
                # Strategy: Target moves > 50% of typical daily volatility (ATR)
                # Best for JNJ, KO, PG
                df['Dynamic_Threshold'] = (df['ATR'] / df['AdjClose']) * 0.5
                df['Target'] = (df['Next_Ret'].abs() > df['Dynamic_Threshold']).astype(int)
            else:
                # Strategy: Target fixed momentum breakouts
                # Best for TSLA, NVDA, TCS.NS
                df['Target'] = (df['Next_Ret'].abs() > threshold).astype(int)

            df['Target_Direction'] = (df['Next_Ret'] > 0).astype(int)

            # Clean up
            df = df.replace([np.inf, -np.inf], 0).ffill().fillna(0)

            return df

        except Exception as e:
            raise DataValidationError(f"Feature calculation failed: {str(e)}")

    @staticmethod
    def get_sample_weights(df, recent_weeks=12, multiplier=3.5):
        """
        Generates hybrid weights for training.
        Ensures recent data has more impact on the loss function.
        """
        train_size = len(df)
        weights = np.ones(train_size)

        # Calculate cutoff for "recent" data (5 trading days per week)
        recent_days = recent_weeks * 5
        cutoff = max(0, train_size - recent_days)

        # Apply higher weight to the most recent period
        weights[cutoff:] = multiplier

        # Linear smoothing for the transition to avoid abrupt weight jumps
        decay_window = min(40, cutoff)
        if decay_window > 0:
            weights[cutoff - decay_window:cutoff] = np.linspace(1.0, multiplier, decay_window)

        return weights