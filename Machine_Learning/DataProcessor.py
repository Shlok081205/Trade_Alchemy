import numpy as np
import pandas as pd
from typing import Tuple, List


# Custom exception for data validation failures
class DataValidationError(Exception):
    pass


class FeatureCalculator:

    def __init__(self):
        # Define required columns for stock data processing
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Validate input DataFrame before processing features
    def validate_input(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors

        missing = set(self.required_columns) - set(df.columns)
        if missing and 'AdjClose' not in df.columns and 'Close' not in df.columns:
            errors.append(f"Missing critical columns: {missing}")

        if len(df) < 60:
            errors.append(f"Need at least 60 rows, got {len(df)}")

        return len(errors) == 0, errors

    # --- NEW: CONTEXT FEATURE LOGIC ---
    def add_context_features(self, target_df, market_map, context):
        """
        Calculates Relative Strength vs Peers and Partners.
        """
        df = target_df.copy()

        # Helper to calculate group index
        def get_group_index(tickers):
            prices = pd.DataFrame(index=df.index)
            for t in tickers:
                if t in market_map and not market_map[t].empty:
                    # Align dates and forward fill
                    if 'AdjClose' in market_map[t].columns:
                        clean_series = market_map[t]['AdjClose'].reindex(df.index).ffill()
                        prices[t] = clean_series

            if prices.empty: return None
            # Normalize each stock to start at 1.0, then average them
            return (prices / prices.iloc[0]).mean(axis=1)

        # 1. Peer Analysis (Relative Strength vs Competitors)
        peer_idx = get_group_index(context.get('peers', []))
        if peer_idx is not None:
            target_norm = df['AdjClose'] / df['AdjClose'].iloc[0]
            df['Rel_Str_Peers'] = target_norm / peer_idx
        else:
            df['Rel_Str_Peers'] = 1.0

        # 2. Partner Analysis (Relative Strength vs Supply Chain)
        partner_idx = get_group_index(context.get('partners', []))
        if partner_idx is not None:
            target_norm = df['AdjClose'] / df['AdjClose'].iloc[0]
            df['Rel_Str_Partners'] = target_norm / partner_idx
        else:
            df['Rel_Str_Partners'] = 1.0

        return df

    # Calculate technical indicators
    def calculate_features(self, df, threshold=0.01, regime="volatile"):
        # Validate input data first
        is_valid, errors = self.validate_input(df)
        if not is_valid: return None

        df = df.copy()
        # Fallback for AdjClose
        if 'AdjClose' not in df.columns: df['AdjClose'] = df['Close']

        # 1. Returns
        df['Ret'] = df['AdjClose'].pct_change()

        # 2. Volatility (ATR)
        h_l = df['High'] - df['Low']
        h_c = np.abs(df['High'] - df['AdjClose'].shift())
        l_c = np.abs(df['Low'] - df['AdjClose'].shift())
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # Volatility regime
        df['Vol_Regime'] = df['ATR'] / df['ATR'].rolling(100).mean()

        # 3. RSI
        delta = df['AdjClose'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 4. MACD
        ema12 = df['AdjClose'].ewm(span=12).mean()
        ema26 = df['AdjClose'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Hist'] = df['MACD'] - df['MACD'].ewm(span=9).mean()

        # 5. Moving Averages
        df['MA_50'] = df['AdjClose'].rolling(50).mean()
        df['Dist_MA50'] = (df['AdjClose'] - df['MA_50']) / df['MA_50']

        # --- TARGET GENERATION (Volatility Focus) ---
        # Predict if price moves > threshold (1.0%)
        df['Next_Ret'] = df['AdjClose'].pct_change().shift(-1)
        df['Target'] = (df['Next_Ret'].abs() > threshold).astype(int)

        # Store direction for UI display (even if we train on volatility)
        df['Target_Direction'] = (df['Next_Ret'] > 0).astype(int)

        df = df.replace([np.inf, -np.inf], 0).dropna()
        return df

    # Flask-specific method
    def calculate_features_for_api(self, df, regime="volatile"):
        df_with_features = self.calculate_features(df, regime=regime)
        if df_with_features is None: return None
        return df_with_features