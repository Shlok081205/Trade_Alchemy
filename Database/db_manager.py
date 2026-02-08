import sqlite3
from datetime import datetime


class DatabaseManager:
    # Initialize database connection and create tables if they don't exist
    def __init__(self, db_path: str = "app.db"):
        self.db_path = db_path
        # Create all necessary tables when the database manager is initialized
        self._initialize_tables()

    # Creates a thread-safe connection for Flask
    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=True)
        conn.row_factory = sqlite3.Row
        return conn

    # Create all database tables if they don't already exist
    def _initialize_tables(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # OTP verification table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS otp_verification (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp_code TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                is_used INTEGER DEFAULT 0
            )
        """)

        # Watchlist table - CORRECTED SCHEMA with added_at
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                buy_price REAL,
                added_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, ticker)
            )
        """)

        conn.commit()
        conn.close()

    def cleanup_expired_otps(self):
        """Remove expired OTP codes from the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM otp_verification WHERE expires_at < ?",
            (datetime.now().isoformat(),)
        )
        conn.commit()
        conn.close()