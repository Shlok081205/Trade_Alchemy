import hashlib
import random
import getpass
import sqlite3
from typing import Optional, Tuple
from Database import DatabaseManager


class AuthManager:
    """Handles user authentication and verification"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def human_verification(self) -> bool:
        """Simple CAPTCHA-like verification"""
        print("\n" + "=" * 60)
        print("🤖 HUMAN VERIFICATION")
        print("=" * 60)

        numbers = [random.randint(0, 9) for _ in range(4)]
        order = random.choice(['Ascending', 'Descending'])

        print(f"Numbers: {', '.join(map(str, numbers))}")
        print(f"Task: Enter these numbers in {order.upper()} order")

        if order == 'Ascending':
            correct = ''.join(map(str, sorted(numbers)))
        else:
            correct = ''.join(map(str, sorted(numbers, reverse=True)))

        try:
            user_input = input("\nYour answer: ").strip()
            if user_input == correct:
                print("✅ Verification successful!")
                return True
            else:
                print(f"❌ Incorrect! Expected: {correct}")
                return False
        except KeyboardInterrupt:
            return False

    def sign_up(self) -> bool:
        """Create new user account"""
        print("\n" + "=" * 60)
        print("📝 SIGN UP")
        print("=" * 60)

        try:
            username = input("Username: ").strip()
            if not username:
                return False

            email = input("Email: ").strip()
            password = input("Password: ")

            if not self.human_verification():
                return False

            hashed_pw = self.hash_password(password)
            self.db.cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed_pw)
            )
            self.db.conn.commit()
            print(f"\n✅ Account created successfully! Welcome, {username}!")
            return True

        except sqlite3.IntegrityError:
            print("❌ Username or email already exists")
            return False
        except Exception as e:
            print(f"❌ Sign up failed: {e}")
            return False

    def sign_in(self) -> Optional[Tuple[int, str]]:
        """Authenticate existing user"""
        print("\n" + "=" * 60)
        print("🔐 SIGN IN")
        print("=" * 60)

        try:
            login = input("Username or Email: ").strip()
            password = input("Password: ")
            hashed_pw = self.hash_password(password)

            self.db.cursor.execute(
                """SELECT id, username
                   FROM users
                   WHERE (username = ? OR email = ?)
                     AND password = ?""",
                (login, login, hashed_pw)
            )

            result = self.db.cursor.fetchone()
            if result:
                user_id, username = result
                print(f"\n✅ Welcome back, {username}!")
                return user_id, username
            else:
                print("❌ Invalid credentials")
                return None
        except Exception as e:
            print(f"❌ Sign in failed: {e}")
            return None
    def sign_in_web(self, login: str, password: str) -> Optional[Tuple[int, str]]:
        """Authenticate existing user for web interface"""
        try:
            hashed_pw = self.hash_password(password)

            self.db.cursor.execute(
                """SELECT id, username
                   FROM users
                   WHERE (username = ? OR email = ?)
                     AND password = ?""",
                (login, login, hashed_pw)
            )

            result = self.db.cursor.fetchone()
            if result:
                user_id, username = result
                return user_id, username
            else:
                return None
        except Exception as e:
            print(f"❌ Sign in failed: {e}")
            return None
