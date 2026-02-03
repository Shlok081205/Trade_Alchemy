import numpy as np
import pandas as pd

class FeatureCalculator:

    def calculate_features(self,df,threshold=0.002):
        """Calculate comprehensive feature set"""

        # Basic returns
        df['Ret'] = df['AdjClose'].pct_change(fill_method=None)

        # ============= VOLATILITY FEATURES =============
        # ATR (Average True Range)
        hl = df['High'] - df['Low']
        hc = np.abs(df['High'] - df['AdjClose'].shift())
        lc = np.abs(df['Low'] - df['AdjClose'].shift())
        df['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

        # Volatility regime (current ATR vs long-term average)
        df['Vol_Regime'] = df['ATR'] / df['ATR'].rolling(100).mean()

        # Realized volatility (std of returns)
        df['Realized_Vol'] = df['Ret'].rolling(20).std()

        # ============= MOMENTUM FEATURES =============
        # MACD
        ema12 = df['AdjClose'].ewm(span=12).mean()
        ema26 = df['AdjClose'].ewm(span=26).mean()
        macd = ema12 - ema26
        df['MACD_Hist'] = macd - macd.ewm(span=9).mean()

        # RSI
        delta = df['AdjClose'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # RSI momentum (is RSI rising or falling?)
        df['RSI_Change'] = df['RSI'].diff()
        df['RSI_Acceleration'] = df['RSI_Change'].diff()

        # Multiple timeframe momentum
        df['Mom_5'] = df['AdjClose'].pct_change(5,fill_method=None)
        df['Mom_20'] = df['AdjClose'].pct_change(20,fill_method=None)
        df['Mom_60'] = df['AdjClose'].pct_change(60,fill_method=None)

        # ============= VOLUME FEATURES =============
        # Volume ratio (current vs average)
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # Volume trend
        df['Volume_Trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()

        # Price-Volume divergence
        df['PV_Divergence'] = (df['Ret'] > 0).astype(int) - (df['Volume_Ratio'] > 1).astype(int)

        # Money flow (price * volume)
        df['Money_Flow'] = df['AdjClose'] * df['Volume']
        df['MF_Ratio'] = df['Money_Flow'] / df['Money_Flow'].rolling(20).mean()

        # ============= PRICE POSITION FEATURES =============
        # Distance from highs/lows
        df['Distance_From_High_20'] = (df['AdjClose'] - df['High'].rolling(20).max()) / df['AdjClose']
        df['Distance_From_Low_20'] = (df['AdjClose'] - df['Low'].rolling(20).min()) / df['AdjClose']
        df['Distance_From_High_60'] = (df['AdjClose'] - df['High'].rolling(60).max()) / df['AdjClose']
        df['Distance_From_Low_60'] = (df['AdjClose'] - df['Low'].rolling(60).min()) / df['AdjClose']

        # Moving average distances
        df['MA_20'] = df['AdjClose'].rolling(20).mean()
        df['MA_50'] = df['AdjClose'].rolling(50).mean()
        df['MA_200'] = df['AdjClose'].rolling(200).mean()
        df['Distance_MA20'] = (df['AdjClose'] - df['MA_20']) / df['MA_20']
        df['Distance_MA50'] = (df['AdjClose'] - df['MA_50']) / df['MA_50']
        df['Distance_MA200'] = (df['AdjClose'] - df['MA_200']) / df['MA_200']

        # MA crossovers
        df['MA_Cross_20_50'] = (df['MA_20'] > df['MA_50']).astype(int)
        df['MA_Cross_50_200'] = (df['MA_50'] > df['MA_200']).astype(int)

        # ============= PATTERN FEATURES =============
        # Higher highs, higher lows (trend strength)
        df['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['Higher_Low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
        df['Trend_Strength'] = df['Higher_High'].rolling(5).sum() + df['Higher_Low'].rolling(5).sum()

        # Candle patterns (simplified)
        df['Body_Size'] = np.abs(df['AdjClose'] - df['Open']) / df['AdjClose']
        df['Upper_Shadow'] = (df['High'] - df[['AdjClose', 'Open']].max(axis=1)) / df['AdjClose']
        df['Lower_Shadow'] = (df[['AdjClose', 'Open']].min(axis=1) - df['Low']) / df['AdjClose']

        # ============= TARGET =============
        # Volatility-adjusted target (more sophisticated)
        df['Next_Ret'] = df['Ret'].shift(-1)
        df['Target'] = (df['Next_Ret'] > threshold).astype(int)

        # Clean up
        #df.dropna(inplace=True)

        # Select feature columns
        feature_cols = [
            'Ret', 'ATR', 'Vol_Regime', 'Realized_Vol',
            'MACD_Hist', 'RSI', 'RSI_Change', 'RSI_Acceleration',
            'Mom_5', 'Mom_20', 'Mom_60',
            'Volume_Ratio', 'Volume_Trend', 'PV_Divergence', 'MF_Ratio',
            'Distance_From_High_20', 'Distance_From_Low_20',
            'Distance_From_High_60', 'Distance_From_Low_60',
            'Distance_MA20', 'Distance_MA50', 'Distance_MA200',
            'MA_Cross_20_50', 'MA_Cross_50_200',
            'Trend_Strength', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow',
            'Target'
        ]
        return df
        #return df[feature_cols].copy()


if __name__ == "__main__":
    from yahoo_scraper import YahooScraper
    ys  = YahooScraper()
    fc =  FeatureCalculator()
    data = ys.scrape("TCS.NS", time_range="5d", use_proxy=False, v10=True, v8=True, v7=True,v10_full_access=True)
    data = ys.v8_formatter(data)
    print(fc.calculate_features(data))
