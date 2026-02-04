import sqlite3
import sys

class DatabaseManager:
    """Handles all database operations with persistent storage"""

    def __init__(self, db_path: str = "app.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._initialize_tables()

    def _connect(self):
        """Establish connection to persistent database file"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"✅ Database connected: {self.db_path}")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            sys.exit(1)

    def _initialize_tables(self):
        """Create tables if they don't exist"""
        try:
            # Users table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Watchlist table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    buy_price REAL,
                    buy_date TEXT,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(user_id, ticker)
                )
            """)

            self.conn.commit()
            print("✅ Database tables initialized")
        except Exception as e:
            print(f"❌ Table initialization failed: {e}")
            sys.exit(1)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
