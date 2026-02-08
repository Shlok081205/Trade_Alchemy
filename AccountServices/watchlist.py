import sqlite3
import time
from typing import List, Dict, Optional
from datetime import datetime
import yfinance as yf
import pandas as pd

# Global in-memory cache
# Structure: { 'TICKER': { 'data': {...}, 'timestamp': time.time() } }
PRICE_CACHE = {}
CACHE_DURATION = 300  # 5 minutes


class WatchlistManager:

    def __init__(self, db_manager, user_id: int):
        self.db = db_manager
        self.user_id = user_id

    def get_watchlist(self) -> List[Dict]:
        """Fetch basic watchlist data from DB."""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """SELECT ticker, buy_price, added_at
               FROM watchlist
               WHERE user_id = ?
               ORDER BY ticker""",
            (self.user_id,)
        )

        results = [
            {
                'ticker': row[0],
                'buy_price': row[1],
                'added_at': row[2]
            }
            for row in cursor.fetchall()
        ]

        conn.close()
        return results

    def add_stock(self, ticker: str, buy_price: Optional[float] = None, added_at: Optional[str] = None) -> Dict:
        """Add a stock to the watchlist."""
        try:
            if not added_at:
                added_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Clean ticker (remove spaces, uppercase)
            clean_ticker = ticker.upper().strip()

            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """INSERT INTO watchlist (user_id, ticker, buy_price, added_at)
                   VALUES (?, ?, ?, ?)""",
                (self.user_id, clean_ticker, buy_price, added_at)
            )

            conn.commit()
            conn.close()

            return {'success': True, 'message': f'{clean_ticker} added to watchlist'}

        except sqlite3.IntegrityError:
            return {'success': False, 'message': f'{ticker.upper()} is already in your watchlist'}
        except Exception as e:
            return {'success': False, 'message': f'Failed to add stock: {str(e)}'}

    def remove_stock(self, ticker: str) -> Dict:
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
                (self.user_id, ticker.upper())
            )

            count = cursor.rowcount
            conn.commit()
            conn.close()

            if count > 0:
                return {'success': True, 'message': f'{ticker.upper()} removed.'}
            return {'success': False, 'message': 'Stock not found.'}

        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}'}

    def _get_cached_data(self, ticker: str):
        """Retrieve data from cache if valid."""
        if ticker in PRICE_CACHE:
            entry = PRICE_CACHE[ticker]
            if time.time() - entry['timestamp'] < CACHE_DURATION:
                return entry['data']
        return None

    def _update_cache(self, ticker: str, data: Dict):
        """Update cache with new data."""
        PRICE_CACHE[ticker] = {
            'data': data,
            'timestamp': time.time()
        }

    def get_watchlist_with_prices(self) -> List[Dict]:
        """
        Fetches watchlist data.
        FIXED: Robust handling for Single vs Multi-ticker yfinance structures.
        """
        stocks = self.get_watchlist()
        if not stocks:
            return []

        detailed_stocks = []
        tickers_to_fetch = []

        # 1. Check Cache First
        for stock in stocks:
            cached = self._get_cached_data(stock['ticker'])
            if cached:
                stock_data = cached.copy()
                stock_data['added_at'] = stock['added_at']
                detailed_stocks.append(stock_data)
            else:
                tickers_to_fetch.append(stock['ticker'])

        # 2. Fetch missing tickers
        if tickers_to_fetch:
            print(f"📊 Fetching fresh data for: {', '.join(tickers_to_fetch)}")

            try:
                # Download data
                bulk_data = yf.download(
                    tickers_to_fetch,
                    period="1mo",
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    threads=False
                )

                for ticker in tickers_to_fetch:
                    data = {
                        'ticker': ticker,
                        'current_price': "N/A",
                        'change_percent': 0.0,
                        'sparkline_data': []
                    }

                    try:
                        hist = pd.DataFrame()

                        # --- FIX START: Robust DataFrame Extraction ---
                        # Check if bulk_data has a MultiIndex column structure (e.g. ('AAPL', 'Close'))
                        if isinstance(bulk_data.columns, pd.MultiIndex):
                            try:
                                # Extract specific ticker data from MultiIndex
                                hist = bulk_data[ticker]
                            except KeyError:
                                # Ticker might be missing in the response
                                pass
                        else:
                            # It is a flat DataFrame (single ticker download often results in this)
                            # Only use it if we are sure it matches our single requested ticker
                            if len(tickers_to_fetch) == 1:
                                hist = bulk_data
                        # --- FIX END ---

                        if not hist.empty and 'Close' in hist.columns:
                            hist = hist.dropna(subset=['Close'])

                            if not hist.empty:
                                closes = hist['Close'].tolist()
                                current_price = closes[-1]

                                if len(closes) > 1:
                                    prev_close = closes[-2]
                                    change = ((current_price - prev_close) / prev_close) * 100
                                else:
                                    change = 0.0

                                data['current_price'] = f"{current_price:,.2f}"
                                data['change_percent'] = round(change, 2)
                                data['sparkline_data'] = closes[-20:]

                    except Exception as e:
                        print(f"⚠️ Error processing {ticker}: {e}")

                    # If price is valid, cache it
                    if data['current_price'] != "N/A" and data['current_price'] != "Error":
                        self._update_cache(ticker, data)

                    # Prepare final record
                    final_stock_record = data.copy()
                    original = next((s for s in stocks if s['ticker'] == ticker), None)
                    if original:
                        final_stock_record['added_at'] = original['added_at']

                    detailed_stocks.append(final_stock_record)

            except Exception as e:
                print(f"❌ Bulk download critical error: {e}")
                # Fallback for all fetched tickers
                for ticker in tickers_to_fetch:
                    detailed_stocks.append({
                        'ticker': ticker,
                        'added_at': datetime.now().strftime("%Y-%m-%d"),
                        'current_price': "Error",
                        'change_percent': 0.0,
                        'sparkline_data': []
                    })

        # Sort alphabetically
        detailed_stocks.sort(key=lambda x: x['ticker'])
        return detailed_stocks