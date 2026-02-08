from typing import Dict, Optional
from Web_Scraping import YahooScraper
from Web_Scraping import Gemini
from Machine_Learning import MultiTimeframeLSTM
from Machine_Learning import FeatureCalculator


class StockAnalyzer:

    def __init__(self, gemini_api_key=None):
        self.scraper = YahooScraper()
        self.gemini = Gemini()
        self.feature_calc = FeatureCalculator()
        self.lstm = MultiTimeframeLSTM()
        self.gemini_api_key = gemini_api_key

    # Main Analysis Method
    def ai_prediction(self, ticker: str, gemini_api_key: str = None) -> Optional[Dict]:

        api_key = gemini_api_key or self.gemini_api_key
        print(f"\n🤖 Running AI analysis for {ticker}...")

        try:
            # 1. GET CONTEXT (Peers & Partners)
            # We need this to calculate Relative Strength features
            context = {
                'peers': [],
                'partners': [],
                'market_regime': 'volatile'
            }

            if api_key:
                gemini_data = self.gemini.get_info(ticker, api_key)
                if gemini_data:
                    context['peers'] = gemini_data.get('peers', [])[:3]
                    context['partners'] = gemini_data.get('partners', [])[:3]
                    context['market_regime'] = gemini_data.get('market_regime', 'volatile')

            # 2. DATA FETCHING (Ecosystem)
            # We fetch data for the main ticker AND its ecosystem
            tickers_to_fetch = [ticker]
            # Extract tickers from context (handling dicts if Gemini returns dicts)
            for p in context['peers']:
                if isinstance(p, dict) and 'ticker' in p:
                    tickers_to_fetch.append(p['ticker'])
                elif isinstance(p, str):
                    tickers_to_fetch.append(p)

            for p in context['partners']:
                if isinstance(p, dict) and 'ticker' in p:
                    tickers_to_fetch.append(p['ticker'])
                elif isinstance(p, str):
                    tickers_to_fetch.append(p)

            # Deduplicate
            tickers_to_fetch = list(set(tickers_to_fetch))
            print(f"📊 Fetching data for ecosystem: {tickers_to_fetch}")

            market_map = {}
            for t in tickers_to_fetch:
                # Use v8 for historical data
                df = self.scraper.scrape(t, v8=True, time_range="5y")
                if df and 'v8' in df:
                    formatted = self.scraper.v8_formatter(df)
                    if formatted is not None and not formatted.empty:
                        market_map[t] = formatted

            if ticker not in market_map:
                print("❌ Main ticker data not found.")
                return None

            # 3. FEATURE ENGINEERING
            # A. Basic Features
            df_main = self.feature_calc.calculate_features(
                market_map[ticker],
                regime=context['market_regime']
            )
            if df_main is None: return None

            # B. Context Features (Relative Strength)
            # Simplify context list for processor
            simple_context = {
                'peers': [t for t in tickers_to_fetch if t in context['peers'] or any(
                    p.get('ticker') == t for p in context['peers'] if isinstance(p, dict))],
                'partners': [t for t in tickers_to_fetch if t in context['partners'] or any(
                    p.get('ticker') == t for p in context['partners'] if isinstance(p, dict))]
            }
            df_final = self.feature_calc.add_context_features(df_main, market_map, simple_context)

            # 4. PREDICTION
            result = self.lstm.train_and_predict(df_final)
            if not result: return None

            prob, acc, _ = result

            # 5. INTERPRETATION
            # Prob > 0.5 means High Volatility (Risk)
            # Prob < 0.5 means Stable
            is_risky = prob > 0.5
            confidence = abs(prob - 0.5) * 2

            return {
                'ticker': ticker,
                'direction': "DOWN" if is_risky else "UP",
                'probability': float(prob),
                'confidence': float(confidence),
                'regime': context['market_regime'],
                'accuracy': float(acc * 100),
                'atr': float(df_final['ATR'].iloc[-1]),
                'current_price': float(df_final['AdjClose'].iloc[-1])
            }
        except Exception as e:
            print(f"Analyzer Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Flask API calls this
    def analyze_for_api(self, ticker: str):
        return self.ai_prediction(ticker)

    def get_fundamentals(self, ticker: str):
        data = self.scraper.scrape(ticker, v10=True)
        return {'success': True, 'data': data['v10']} if data and 'v10' in data else {'success': False}

    def get_quote(self, ticker: str):
        data = self.scraper.scrape(ticker, v7=True)
        return {'success': True, 'data': data['v7']} if data and 'v7' in data else {'success': False}