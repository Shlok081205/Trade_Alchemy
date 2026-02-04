import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
tf.keras.backend.clear_session()

# Import custom modules
from AccountServices import PortfolioRiskManager
from Machine_Learning import MultiTimeframeLSTM, FeatureCalculator
from Web_Scraping import YahooScraper, Gemini
from Web_Scraping.config_setup import GEMINI_API_KEY

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    LOOKBACK_SHORT = 20
    LOOKBACK_MEDIUM = 40
    LOOKBACK_LONG = 60
    RECENT_DATA_WEIGHT = 3.5
    RECENT_WEEKS_EMPHASIZED = 12
    MIN_CONFIDENCE = 0.55
    INITIAL_CAPITAL = 100000
    MAX_POSITION_SIZE = 0.25
    MAX_PORTFOLIO_RISK = 0.02
    MIN_DAYS_REQUIRED = 200

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text, char="="):
    print(f"\n{char * 70}\n  {text}\n{char * 70}\n")

def calculate_sample_weights(train_size, recent_weeks=12, weight_multiplier=3.5):
    weights = np.ones(train_size)
    recent_days = recent_weeks * 5
    recent_cutoff = max(0, train_size - recent_days)
    weights[recent_cutoff:] = weight_multiplier
    return weights * (train_size / weights.sum())

# ============================================================================
# DATA COLLECTION MODULE
# ============================================================================

class DataCollector:
    def __init__(self, gemini_api_key):
        self.ys = YahooScraper()
        self.gem = Gemini()
        self.gemini_api_key = gemini_api_key
        self.fc = FeatureCalculator()

    def get_market_context(self, ticker):
        """Fetches peers and regime from Gemini"""
        print(f"🔍 Analyzing {ticker} market personality...")
        try:
            response = self.gem.retrive_data(ticker, self.gemini_api_key)
            data = json.loads(response)

            partners = [x['ticker'] for x in data.get("partners", []) if x['ticker'] != 'Private']
            peers = [x['ticker'] for x in data.get("peers", [])]
            regime = data.get("market_regime", "volatile")

            idx = data.get("sectoral_index") or data.get("market_index")
            related = list(set(partners + peers))
            if idx: related.append(idx)

            return related, idx, regime
        except:
            return [], None, "volatile"

    def build_dataset(self, ticker, related, regime):
        print(f"🔄 Collecting data for {ticker} ({regime.upper()} mode)...")
        raw = self.ys.scrape(ticker, v8=True, time_range="5y")
        main_df = self.ys.v8_formatter(raw)

        # Core change: Use Regime-aware Feature Calculation
        main_df = self.fc.calculate_features(main_df, regime=regime)

        for rel in related:
            try:
                rel_raw = self.ys.scrape(rel, v8=True, time_range="5y")
                rel_df = self.ys.v8_formatter(rel_raw)
                if rel_df is not None:
                    main_df = main_df.join(rel_df[['Close']].rename(columns={'Close': f'{rel}_Close'}), how='left')
            except: continue

        return main_df.ffill().fillna(0).dropna(subset=['Target'])

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class StockPredictor:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.scaler = RobustScaler()
        self.val_accuracy = 0.0

    def _prepare_sequences(self, X_scaled, y_raw, is_training=True):
        X_seq, y_seq, weights = [], [], []
        raw_weights = calculate_sample_weights(len(X_scaled))

        for i in range(self.config.LOOKBACK_MEDIUM, len(X_scaled)):
            X_seq.append(X_scaled[i-self.config.LOOKBACK_MEDIUM:i])
            y_seq.append(y_raw[i])
            weights.append(raw_weights[i] if is_training else 1.0)

        return np.array(X_seq), np.array(y_seq), np.array(weights)

    def train_production_model(self, df):
        y_all = df['Target'].values
        X_all = df.drop(['Target', 'AdjClose', 'Next_Ret'], axis=1, errors='ignore').values

        # PASS 1: Validation and Epoch Tuning
        split = int(len(X_all) * 0.85)
        self.scaler.fit(X_all[:split])
        X_train_seq, y_train_seq, w_train_seq = self._prepare_sequences(self.scaler.transform(X_all[:split]), y_all[:split])
        X_val_seq, y_val_seq, _ = self._prepare_sequences(self.scaler.transform(X_all[split:]), y_all[split:], False)

        self.model = MultiTimeframeLSTM()
        self.model.build_model((X_train_seq.shape[1], X_train_seq.shape[2]))

        print("🔍 PASS 1: Tuning model...")
        history = self.model.model.fit(
            X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq),
            epochs=40, verbose=0, sample_weight=w_train_seq
        )

        best_epoch = np.argmin(history.history['val_loss']) + 1
        self.val_accuracy = history.history['val_accuracy'][best_epoch-1] * 100

        # PASS 2: Retrain on Full Data
        print(f"🚀 PASS 2: Retraining (Epochs: {best_epoch})...")
        X_full_scaled = self.scaler.fit_transform(X_all)
        X_f, y_f, w_f = self._prepare_sequences(X_full_scaled, y_all)
        self.model.build_model((X_f.shape[1], X_f.shape[2]))
        self.model.model.fit(X_f, y_f, epochs=best_epoch, verbose=1, sample_weight=w_f)

        return X_full_scaled

# ============================================================================
# MAIN APP
# ============================================================================

class StockPredictionApp:
    def __init__(self):
        self.collector = DataCollector(GEMINI_API_KEY)
        self.predictor = StockPredictor()
        self.risk = PortfolioRiskManager(Config.INITIAL_CAPITAL)

    def run(self, ticker):
        related, idx, regime = self.collector.get_market_context(ticker)
        df = self.collector.build_dataset(ticker, related, regime)

        X_scaled = self.predictor.train_production_model(df)

        # Last Sequence Prediction
        last_seq = X_scaled[-Config.LOOKBACK_MEDIUM:].reshape(1, Config.LOOKBACK_MEDIUM, -1)
        prob = self.predictor.model.model.predict(last_seq, verbose=0)[0][0]

        # Sizing and Display
        price = df['AdjClose'].iloc[-1]
        vol = (df['ATR'].iloc[-1] / price)
        conf = prob if prob > 0.5 else 1 - prob

        print_header(f"RESULTS FOR {ticker}")
        print(f"🌡️  Market Regime:    {regime.upper()}")
        print(f"🎯 Model Accuracy:   {self.predictor.val_accuracy:.2f}%")
        print(f"📈 Direction:        {'UP 🟢' if prob > 0.5 else 'DOWN 🔴'}")
        print(f"🤝 Confidence:       {conf:.2%}")

        if conf >= Config.MIN_CONFIDENCE:
            shares = self.risk.calculate_position_size(ticker, conf, vol, price)
            print(f"💼 Recommendation:   BUY {shares} shares (${shares*price:,.2f})")
        else:
            print("⚠️  Recommendation:   NO TRADE (Low Confidence)")

if __name__ == "__main__":
    app = StockPredictionApp()
    while True:
        symbol = input("\n📈 Enter Ticker (or 'q'): ").strip().upper()
        if symbol == 'Q': break
        app.run(symbol)