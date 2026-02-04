"""
Stock Analysis CLI Application
"""
import sys
from datetime import datetime

# Import Custom Modules from new directories
try:
    from Database import DatabaseManager
    from AccountServices import AuthManager
    from AccountServices import WatchlistManager
    from Machine_Learning import StockAnalyzer
except ImportError as e:
    print(f"❌ Missing module: {e}")
    sys.exit(1)


class StockAnalysisApp:
    """Main application controller"""

    def __init__(self):
        # Initialize the managers
        self.db = DatabaseManager()
        self.auth = AuthManager(self.db)
        self.analyzer = StockAnalyzer()

        self.current_user = None
        self.watchlist = None

    def authenticate(self) -> bool:
        while True:
            print("\n=== STOCK CLI AUTHENTICATION ===")
            print("1. Sign In\n2. Sign Up\n3. Exit")
            choice = input("Select: ").strip()

            if choice == '1':
                result = self.auth.sign_in()
                if result:
                    user_id, username = result
                    self.current_user = {'id': user_id, 'username': username}
                    # Initialize watchlist for this specific user
                    self.watchlist = WatchlistManager(self.db, user_id)
                    return True
            elif choice == '2':
                self.auth.sign_up()
            elif choice == '3':
                return False
            else:
                print("Invalid option")

    def main_menu(self):
        while True:
            print(f"\n=== MAIN MENU ({self.current_user['username']}) ===")
            print("1. Watchlist (Fast View)")
            print("2. Deep Stock Data")
            print("3. AI Prediction")
            print("4. Log Out")

            choice = input("Select: ").strip()

            if choice == '1':
                self.run_watchlist_menu()
            elif choice == '2':
                t = input("Ticker: ").strip().upper()
                if t: self.analyzer.get_deep_data(t)
            elif choice == '3':
                t = input("Ticker: ").strip().upper()
                if t:
                    res = self.analyzer.ai_prediction(t)
                    if res and input("Calculate Size? (y/n): ").lower() == 'y':
                        self.analyzer.calculate_position_size(res)
            elif choice == '4':
                break

    def run_watchlist_menu(self):
        self.watchlist.display_watchlist()
        print("\nOptions: [A]dd, [R]emove, [B]ack")
        opt = input("Choice: ").upper()

        if opt == 'A':
            t = input("Ticker: ").upper()
            p = input("Buy Price (optional): ")
            price = float(p) if p else None
            date = datetime.now().strftime('%Y-%m-%d') if p else None
            self.watchlist.add_stock(t, price, date)
        elif opt == 'R':
            t = input("Ticker: ").upper()
            self.watchlist.remove_stock(t)

    def run(self):
        try:
            if self.authenticate():
                self.main_menu()
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.db.close()


if __name__ == "__main__":
    app = StockAnalysisApp()
    app.run()