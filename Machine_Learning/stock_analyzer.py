from typing import Dict
from Web_Scraping.yahoo_scraper import YahooScraper
from Web_Scraping.gemini import Gemini
from Web_Scraping.config_setup import GEMINI_API_KEY
from Machine_Learning.LSTMConfidenceModel import MultiTimeframeLSTM
from Machine_Learning.DataProcessor import FeatureCalculator


class StockAnalyzer:
    """Handles stock data fetching and analysis"""

    def __init__(self):
        self.scraper = YahooScraper()
        self.gemini = Gemini()
        self.feature_calc = FeatureCalculator()
        self.lstm = MultiTimeframeLSTM()

    def get_deep_data(self, ticker: str):
        print(f"\n🔍 Fetching deep data for {ticker}...")
        try:
            data = self.scraper.scrape(ticker, v7=True)
            if not data or 'v7' not in data:
                print(f"❌ Failed to fetch data for {ticker}")
                return

            v7 = data['v7']
            print("\n" + "=" * 80)
            print(f"📊 DEEP STOCK DATA: {ticker}")
            print(f"Current Price:       ${v7.get('Current Price', 'N/A')}")
            print(f"Market Cap:          ${v7.get('Market Cap', 'N/A'):,.0f}" if v7.get(
                'Market Cap') else "Market Cap:          N/A")
            print(f"P/E Ratio:           {v7.get('Trailing PE', 'N/A')}")
            print("=" * 80)
        except Exception as e:
            print(f"❌ Error fetching deep data: {e}")

    def ai_prediction(self, ticker: str):
        print(f"\n🤖 Running AI analysis for {ticker}...")
        try:
            # Step A: Gemini Context
            print("📡 Fetching market regime...")
            context_data = self.gemini.get_info(ticker, GEMINI_API_KEY)
            regime = context_data.get('market_regime', 'volatile') if context_data else 'volatile'

            # Step B: Historical Data
            print("📊 Fetching historical data...")
            raw_data = self.scraper.scrape(ticker, v8=True, time_range="5y")
            if not raw_data: return None
            df = self.scraper.v8_formatter(raw_data)

            # Step C: Features & Prediction
            df = self.feature_calc.calculate_features(df, regime=regime)
            if len(df) < 200: return None

            result = self.lstm.train_and_predict(df, verbose=0)
            if not result: return None

            prob, accuracy, _ = result
            direction = "UP 🟢" if prob > 0.5 else "DOWN 🔴"
            confidence = prob if prob > 0.5 else (1 - prob)

            print(f"\n🎯 Result: {direction} (Conf: {confidence:.1%}) | Regime: {regime}")

            return {
                'ticker': ticker,
                'direction': direction,
                'confidence': confidence,
                'price': df['AdjClose'].iloc[-1],
                'atr': df['ATR'].iloc[-1]
            }
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None

    def calculate_position_size(self, prediction_result: Dict):
        try:
            capital = float(input("\n💰 Enter capital: $").replace(',', ''))
            atr = prediction_result['atr']
            price = prediction_result['price']

            # Logic: Risk 2% of capital, Stop Loss at 2x ATR
            risk_amount = capital * 0.02
            risk_per_share = atr * 2
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

            # Adjust by confidence
            shares = int(shares * prediction_result['confidence'])

            print(f"\n📋 RECOMMENDATION: Buy {shares} shares (${shares * price:,.2f})")
            print(f"   (Based on Volatility ${atr:.2f} and Risk ${risk_amount:.2f})")
        except:
            print("❌ Invalid input")