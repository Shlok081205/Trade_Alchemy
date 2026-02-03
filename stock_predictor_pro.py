"""
Production Stock Prediction System
====================================
Hybrid Weighted Strategy with Real-Time Predictions

Features:
- Trains on ALL historical data (no artificial test split)
- Uses Hybrid weighting (2x for recent 10 weeks)
- Real-time prediction after training
- Multi-ticker correlation analysis
- User-friendly interface
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import custom modules
from AccountServices import PortfolioRiskManager
from Machine_Learning import MultiTimeframeLSTM, ConfidenceFilter, FeatureCalculator
from Web_Scraping import YahooScraper, Gemini
from Web_Scraping.config_setup import GEMINI_API_KEY

import tensorflow as tf

tf.keras.backend.clear_session()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for the prediction system"""

    # Model Parameters
    LOOKBACK_SHORT = 20
    LOOKBACK_MEDIUM = 40
    LOOKBACK_LONG = 60

    # Hybrid Strategy Parameters
    RECENT_DATA_WEIGHT = 2.0
    RECENT_WEEKS_EMPHASIZED = 10

    # Trading Parameters
    THRESHOLD = 0.002  # 0.2% price movement threshold
    MIN_CONFIDENCE = 0.55  # Minimum confidence to trade

    # Risk Management
    INITIAL_CAPITAL = 100000
    MAX_POSITION_SIZE = 0.25
    MAX_PORTFOLIO_RISK = 0.02

    # Data Requirements
    MIN_DAYS_REQUIRED = 200  # Minimum data for training


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text, char="=", width=70):
    """Print formatted header"""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def print_section(text, char="-", width=70):
    """Print formatted section"""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def calculate_sample_weights(train_size, recent_weeks=10, weight_multiplier=2.0):
    """
    Calculate sample weights for hybrid strategy

    Args:
        train_size: Number of training samples
        recent_weeks: Number of recent weeks to emphasize
        weight_multiplier: Weight multiplier for recent data

    Returns:
        Array of sample weights
    """
    weights = np.ones(train_size)

    # Calculate cutoff for "recent" data
    recent_days = recent_weeks * 5  # 5 trading days per week
    recent_cutoff = max(0, train_size - recent_days)

    # Apply higher weight to recent data
    weights[recent_cutoff:] = weight_multiplier

    # Smooth transition with gradual decay
    decay_window = min(50, recent_cutoff)
    if decay_window > 0 and recent_cutoff > decay_window:
        decay_start = recent_cutoff - decay_window
        decay_weights = np.linspace(1.0, weight_multiplier, decay_window)
        weights[decay_start:recent_cutoff] = decay_weights

    # Normalize weights
    weights = weights * (train_size / weights.sum())

    return weights


# ============================================================================
# DATA COLLECTION MODULE
# ============================================================================

class DataCollector:
    """Handles data collection from Yahoo Finance and Gemini"""

    def __init__(self, gemini_api_key):
        self.ys = YahooScraper()
        self.gem = Gemini()
        self.gemini_api_key = gemini_api_key
        self.fc = FeatureCalculator()

    def get_related_tickers(self, ticker):
        """
        Get related tickers (partners, peers, sectoral index) using Gemini
        """
        print(f"🔍 Analyzing {ticker} to find related companies...")

        try:
            response = self.gem.retrive_data(ticker, self.gemini_api_key)
            data = json.loads(response)

            # Extract tickers
            partners = [x['ticker'] for x in data.get("partners", []) if x['ticker'] != 'Private']
            peers = [x['ticker'] for x in data.get("peers", [])]
            sectoral_index = data.get("sectoral_index", "")
            market_index = data.get("market_index", "")

            # Use sectoral index if available, otherwise market index
            index_ticker = sectoral_index if sectoral_index else market_index

            # Combine all tickers
            related_tickers = list(set(partners + peers))
            if index_ticker:
                related_tickers.append(index_ticker)

            print(f"✓ Found {len(partners)} partners, {len(peers)} peers")
            if sectoral_index:
                print(f"✓ Sectoral Index: {sectoral_index}")
            elif market_index:
                print(f"✓ Market Index: {market_index}")

            return related_tickers, index_ticker

        except Exception as e:
            print(f"⚠ Error getting related tickers: {str(e)}")
            print("  Continuing with ticker only...")
            return [], None

    def fetch_ticker_data(self, ticker, time_range="5y"):
        """Fetch historical data for a single ticker"""
        try:
            raw_data = self.ys.scrape(ticker, use_proxy=False, v8=True, time_range=time_range)

            if raw_data is None or 'v8' not in raw_data:
                print(f"  ⚠️  {ticker}: No data returned from API")
                return None

            formatted_data = self.ys.v8_formatter(raw_data)

            if formatted_data is None:
                print(f"  ⚠️  {ticker}: Failed to format data")
                return None

            if len(formatted_data) < Config.MIN_DAYS_REQUIRED:
                print(
                    f"  ⚠️  {ticker}: Insufficient data ({len(formatted_data)} days, need {Config.MIN_DAYS_REQUIRED})")
                return None

            return formatted_data

        except Exception as e:
            print(f"  ✗ {ticker}: {str(e)[:80]}")
            return None

    def build_feature_dataset(self, main_ticker, related_tickers=None, time_range="5y"):
        """
        Build complete feature dataset with main ticker and related tickers
        """
        print_section(f"DATA COLLECTION FOR {main_ticker}")

        # Fetch main ticker
        print(f"Fetching main ticker: {main_ticker}...")
        main_df = self.fetch_ticker_data(main_ticker, time_range)

        if main_df is None:
            raise ValueError(f"Failed to fetch data for {main_ticker}")

        print(f"✓ Main ticker: {len(main_df)} days of data")

        # Calculate features for main ticker
        print(f"Calculating features...")
        main_df = self.fc.calculate_features(main_df, threshold=Config.THRESHOLD)
        print(f"✓ Main ticker features: {main_df.shape}")

        # Add related tickers if provided
        if related_tickers:
            print(f"\nProcessing {len(related_tickers)} related tickers...")

            successfully_merged = []
            for related_ticker in related_tickers:
                try:
                    related_df = self.fetch_ticker_data(related_ticker, time_range)

                    if related_df is None:
                        continue

                    # Select essential columns
                    available_cols = [col for col in ['Close', 'Volume', 'High', 'Low']
                                      if col in related_df.columns]

                    if not available_cols:
                        continue

                    related_subset = related_df[available_cols].copy()
                    related_subset.columns = [f"{related_ticker}_{col}"
                                              for col in related_subset.columns]

                    # Join to main dataframe
                    main_df = main_df.join(related_subset, how='left')
                    successfully_merged.append(related_ticker)
                    print(f"  ✓ {related_ticker}: {len(available_cols)} features")

                except Exception as e:
                    print(f"  ✗ {related_ticker}: {str(e)[:50]}")
                    continue

            if successfully_merged:
                print(f"\n✓ Successfully merged: {len(successfully_merged)} tickers")

        # Clean data
        print(f"\n✓ Cleaning missing values...")
        main_df = main_df.ffill().fillna(0)
        main_df = main_df.replace([np.inf, -np.inf], 0)

        # Remove NaN in target
        main_df = main_df.dropna(subset=['Target'])

        print_section("DATA PREPARATION COMPLETE")
        print(f"Final dataset shape:  {main_df.shape}")
        print(f"Total features:       {main_df.shape[1]}")
        print(f"Date range:           {main_df.index[0]} to {main_df.index[-1]}")

        return main_df


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class StockPredictor:
    """Main prediction engine with Hybrid strategy"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.model = None
        self.scaler = None

    def train_with_hybrid_weights(self, df):
        """
        Train model using ALL available data with Hybrid weighting
        """
        print_section("TRAINING MODEL - HYBRID WEIGHTED STRATEGY")

        if len(df) < self.config.LOOKBACK_LONG + 50:
            raise ValueError(f"Insufficient data. Need at least {self.config.LOOKBACK_LONG + 50} days")

        # Prepare data
        y = df['Target'].values
        X = df.drop('Target', axis=1).values

        # Validate data
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("⚠️  Warning: Found NaN or Inf values in features. Cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("Target variable contains NaN or Inf values!")

        print(f"Training samples: {len(X)}")
        print(f"Features: {X.shape[1]}")
        print(f"Class distribution: UP={np.sum(y == 1)}, DOWN={np.sum(y == 0)}")

        # Check for severe class imbalance
        class_ratio = np.sum(y == 1) / len(y)
        if class_ratio < 0.1 or class_ratio > 0.9:
            print(f"⚠️  Warning: Severe class imbalance detected ({class_ratio:.1%} positive class)")

        # Calculate sample weights for Hybrid strategy
        print(f"\n📊 Applying Hybrid Weighting...")
        print(f"   Recent {self.config.RECENT_WEEKS_EMPHASIZED} weeks: "
              f"{self.config.RECENT_DATA_WEIGHT}x weight")

        sample_weights = calculate_sample_weights(
            len(X),
            recent_weeks=self.config.RECENT_WEEKS_EMPHASIZED,
            weight_multiplier=self.config.RECENT_DATA_WEIGHT
        )

        print(f"   Weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
        print(f"   Average weight: {sample_weights.mean():.2f}")

        # Initialize and train model
        print(f"\n🤖 Initializing LSTM model...")
        self.model = MultiTimeframeLSTM(
            lookback_short=self.config.LOOKBACK_SHORT,
            lookback_medium=self.config.LOOKBACK_MEDIUM,
            lookback_long=self.config.LOOKBACK_LONG
        )

        # Scale data
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Prepare sequences
        print(f"📈 Preparing sequences (lookback={self.config.LOOKBACK_MEDIUM})...")
        X_seq, y_seq, weights_seq = [], [], []

        for i in range(self.config.LOOKBACK_MEDIUM, len(X_scaled)):
            X_seq.append(X_scaled[i - self.config.LOOKBACK_MEDIUM:i])
            y_seq.append(y[i])
            weights_seq.append(sample_weights[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        weights_seq = np.array(weights_seq)

        print(f"✓ Sequence shape: {X_seq.shape}")

        # Validate minimum samples
        if len(X_seq) < 100:
            raise ValueError(f"Insufficient sequences for training: {len(X_seq)} (need at least 100)")

        # Train model
        print(f"\n🚀 Training model (this may take a few minutes)...")

        # Build model
        self.model.build_model((X_seq.shape[1], X_seq.shape[2]))

        # Calculate class weights and combine with sample weights
        class_counts = np.bincount(y_seq)
        total = len(y_seq)
        class_weight_0 = total / (2 * class_counts[0])
        class_weight_1 = total / (2 * class_counts[1])

        # Combine class weights with sample weights
        # Apply class weight based on each sample's label
        combined_weights = weights_seq.copy()
        for i in range(len(y_seq)):
            if y_seq[i] == 0:
                combined_weights[i] *= class_weight_0
            else:
                combined_weights[i] *= class_weight_1

        # Normalize combined weights
        combined_weights = combined_weights * (len(combined_weights) / combined_weights.sum())

        print(f"✓ Combined weights applied (class balance + recency)")
        print(f"   Final weight range: {combined_weights.min():.2f} - {combined_weights.max():.2f}")

        # Callbacks
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        # Train with combined weights (NO class_weight parameter)
        history = self.model.model.fit(
            X_seq, y_seq,
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=callbacks,
            sample_weight=combined_weights  # Combined class + recency weights!
        )

        print_section("TRAINING COMPLETE")

        return X_scaled, y_seq

    def predict_next(self, df, X_scaled):
        """
        Predict next day's movement
        """
        # Get last sequence
        last_seq = X_scaled[-self.config.LOOKBACK_MEDIUM:].reshape(
            1, self.config.LOOKBACK_MEDIUM, X_scaled.shape[1]
        )

        # Predict
        prob = self.model.model.predict(last_seq, verbose=0)[0][0]

        # Determine direction
        direction = 'UP' if prob > 0.5 else 'DOWN'
        confidence = prob if prob > 0.5 else 1 - prob

        return {
            'direction': direction,
            'probability': float(prob),
            'confidence': float(confidence),
            'signal_strength': 'STRONG' if confidence > 0.65 else 'MODERATE' if confidence > 0.55 else 'WEAK'
        }


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class StockPredictionApp:
    """Main application class"""

    def __init__(self, gemini_api_key):
        self.gemini_api_key = gemini_api_key
        self.data_collector = DataCollector(gemini_api_key)
        self.predictor = StockPredictor()
        self.portfolio_manager = PortfolioRiskManager(
            initial_capital=Config.INITIAL_CAPITAL,
            max_position_size=Config.MAX_POSITION_SIZE,
            max_portfolio_risk=Config.MAX_PORTFOLIO_RISK
        )

    def analyze_ticker(self, ticker, use_related_tickers=True):
        """
        Complete analysis pipeline for a ticker
        """
        try:
            print_header(f"STOCK PREDICTION SYSTEM - {ticker}")
            print(
                f"Strategy: Hybrid Weighted (Recent {Config.RECENT_WEEKS_EMPHASIZED} weeks @ {Config.RECENT_DATA_WEIGHT}x)")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Step 1: Get related tickers
            related_tickers = []
            index_ticker = None

            if use_related_tickers:
                related_tickers, index_ticker = self.data_collector.get_related_tickers(ticker)

            # Step 2: Fetch and prepare data
            main_df = self.data_collector.build_feature_dataset(
                ticker,
                related_tickers=related_tickers if use_related_tickers else None,
                time_range="5y"
            )

            # Get current price info
            current_price = main_df['Close'].iloc[-1] if 'Close' in main_df.columns else None
            current_date = main_df.index[-1]

            # Step 3: Train model
            X_scaled, y_seq = self.predictor.train_with_hybrid_weights(main_df)

            # Step 4: Make prediction
            print_section("GENERATING PREDICTION")
            prediction = self.predictor.predict_next(main_df, X_scaled)

            # Step 5: Calculate position sizing
            volatility = main_df['ATR'].iloc[-1] / current_price if 'ATR' in main_df.columns else 0.02

            trade_recommendation = None
            if prediction['confidence'] >= Config.MIN_CONFIDENCE:
                shares = self.portfolio_manager.calculate_position_size(
                    ticker,
                    prediction['confidence'],
                    volatility,
                    current_price
                )

                position_value = shares * current_price
                position_weight = position_value / Config.INITIAL_CAPITAL * 100

                trade_recommendation = {
                    'action': 'BUY' if prediction['direction'] == 'UP' else 'SELL',
                    'shares': shares,
                    'position_value': position_value,
                    'position_weight': position_weight
                }

            # Step 6: Display results
            self._display_results(
                ticker,
                current_date,
                current_price,
                volatility,
                prediction,
                trade_recommendation,
                main_df,
                related_tickers,
                index_ticker
            )

            return {
                'ticker': ticker,
                'date': current_date,
                'price': current_price,
                'prediction': prediction,
                'trade': trade_recommendation,
                'success': True
            }

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'ticker': ticker,
                'success': False,
                'error': str(e)
            }

    def _display_results(self, ticker, date, price, volatility, prediction,
                         trade, df, related_tickers, index_ticker):
        """Display prediction results"""

        print_header("PREDICTION RESULTS", "=")

        # Basic Info
        print(f"📊 Ticker:               {ticker}")
        print(f"📅 Latest Date:          {date}")
        print(f"💵 Current Price:        ${price:.2f}")
        print(f"📈 Volatility (ATR/Price): {volatility:.2%}")

        # Prediction
        print(f"\n🎯 PREDICTION FOR NEXT TRADING DAY:")
        print(f"   Direction:            {prediction['direction']} {'🟢' if prediction['direction'] == 'UP' else '🔴'}")
        print(f"   Confidence:           {prediction['confidence']:.2%}")
        print(f"   Signal Strength:      {prediction['signal_strength']}")
        print(f"   Probability:          {prediction['probability']:.2%}")

        # Trade Recommendation
        if trade:
            print(f"\n💼 TRADE RECOMMENDATION:")
            print(f"   Action:               {trade['action']}")
            print(f"   Shares:               {trade['shares']}")
            print(f"   Position Value:       ${trade['position_value']:,.2f}")
            print(f"   Portfolio Weight:     {trade['position_weight']:.1f}%")
        else:
            print(f"\n⚠️  NO TRADE RECOMMENDED")
            print(
                f"   Reason: Confidence ({prediction['confidence']:.1%}) below threshold ({Config.MIN_CONFIDENCE:.1%})")

        # Model Info
        print(f"\n🤖 MODEL INFORMATION:")
        print(f"   Strategy:             Hybrid Weighted")
        print(f"   Training Data:        {len(df)} days")
        print(f"   Features Used:        {df.shape[1]}")
        print(f"   Recent Weight:        {Config.RECENT_DATA_WEIGHT}x for last {Config.RECENT_WEEKS_EMPHASIZED} weeks")

        if related_tickers or index_ticker:
            print(f"\n🔗 CORRELATION DATA:")
            if related_tickers:
                print(f"   Related Tickers:      {', '.join(related_tickers[:5])}")
                if len(related_tickers) > 5:
                    print(f"                         + {len(related_tickers) - 5} more")
            if index_ticker:
                print(f"   Benchmark Index:      {index_ticker}")

        print("\n" + "=" * 70 + "\n")


# ============================================================================
# USER INTERFACE
# ============================================================================

def main():
    """Main function with user interface"""

    print_header("🚀 STOCK PREDICTION SYSTEM - PRODUCTION VERSION", "=")
    print("Hybrid Weighted Strategy | Real-Time Predictions")
    print("=" * 70)

    # Get API key
    try:
        api_key = GEMINI_API_KEY
    except ImportError:
        print("⚠️  config_setup.py not found. Please enter API key manually.")
        api_key = input("Enter Gemini API Key: ").strip()

    # Initialize app
    app = StockPredictionApp(api_key)

    # Main loop
    while True:
        print("\n" + "=" * 70)
        ticker = input("📈 Enter stock ticker (or 'quit' to exit): ").strip().upper()

        if ticker in ['QUIT', 'EXIT', 'Q']:
            print("\n👋 Thank you for using Stock Prediction System!")
            break

        if not ticker:
            print("⚠️  Please enter a valid ticker symbol")
            continue

        # Ask about related tickers
        use_related = input("🔗 Include related tickers for correlation analysis? (Y/n): ").strip().lower()
        use_related_tickers = use_related != 'n'

        # Run analysis
        print("\n🔄 Starting analysis...")
        result = app.analyze_ticker(ticker, use_related_tickers=use_related_tickers)

        if result['success']:
            # Ask if user wants to continue
            continue_input = input("\n📊 Analyze another ticker? (Y/n): ").strip().lower()
            if continue_input == 'n':
                print("\n👋 Thank you for using Stock Prediction System!")
                break
        else:
            retry = input("\n⚠️  Analysis failed. Try another ticker? (Y/n): ").strip().lower()
            if retry == 'n':
                break

    print("\n" + "=" * 70)
    print("Session ended at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()