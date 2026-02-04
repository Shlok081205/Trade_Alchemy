import sqlite3
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime
from Database import DatabaseManager


class WatchlistManager:
    """Manages user's stock watchlist"""

    def __init__(self, db_manager: DatabaseManager, user_id: int):
        self.db = db_manager
        self.user_id = user_id

    def get_watchlist(self) -> List[Dict]:
        self.db.cursor.execute(
            """SELECT ticker, buy_price, buy_date
               FROM watchlist
               WHERE user_id = ?
               ORDER BY ticker""",
            (self.user_id,)
        )
        return [{'ticker': r[0], 'buy_price': r[1], 'buy_date': r[2]} for r in self.db.cursor.fetchall()]

    def add_stock(self, ticker: str, buy_price: Optional[float] = None, buy_date: Optional[str] = None):
        try:
            self.db.cursor.execute(
                """INSERT INTO watchlist (user_id, ticker, buy_price, buy_date)
                   VALUES (?, ?, ?, ?)""",
                (self.user_id, ticker.upper(), buy_price, buy_date)
            )
            self.db.conn.commit()
            print(f"✅ {ticker.upper()} added to watchlist")
        except sqlite3.IntegrityError:
            print(f"❌ {ticker.upper()} is already in your watchlist")
        except Exception as e:
            print(f"❌ Failed to add stock: {e}")

    def remove_stock(self, ticker: str):
        try:
            self.db.cursor.execute(
                "DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
                (self.user_id, ticker.upper())
            )
            self.db.conn.commit()
            print(f"✅ {ticker.upper()} removed")
        except Exception as e:
            print(f"❌ Failed to remove stock: {e}")

    def display_watchlist(self):
        stocks = self.get_watchlist()
        if not stocks:
            print("\n📋 Your watchlist is empty")
            return

        print(f"\n{'Ticker':<10} {'Live Price':<15} {'Buy Price':<15} {'P/L %':<15}")
        print("-" * 60)

        for stock in stocks:
            ticker = stock['ticker']
            buy_price = stock['buy_price']
            try:
                data = yf.Ticker(ticker)
                # Fast fetch using fast_info or regular info
                live_price = data.fast_info.last_price if hasattr(data, 'fast_info') else data.info.get(
                    'regularMarketPrice')

                if live_price:
                    pl_str = "N/A"
                    if buy_price:
                        pl_pct = ((live_price - buy_price) / buy_price) * 100
                        pl_str = f"{pl_pct:+.2f}%"
                        pl_str = f"🟢 {pl_str}" if pl_pct > 0 else f"🔴 {pl_str}"

                    print(
                        f"{ticker:<10} ${live_price:<14.2f} {f'${buy_price:.2f}' if buy_price else 'N/A':<15} {pl_str:<15}")
                else:
                    print(f"{ticker:<10} Error fetching price")
            except:
                print(f"{ticker:<10} Error")