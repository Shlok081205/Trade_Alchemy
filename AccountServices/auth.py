import hashlib
import sqlite3
import smtplib
import random
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple, Dict

try:
    from config import EMAIL_SENDER, EMAIL_PASSWORD
except ImportError:
    EMAIL_SENDER = "phoniexblaze5@gmail.com"
    EMAIL_PASSWORD = "your-app-password-here"


class EmailVerification:
    """Handles OTP generation, database storage, and SMTP email delivery."""

    def __init__(self, db_manager):
        self.db = db_manager
        self.sender = EMAIL_SENDER
        self.password = EMAIL_PASSWORD

    def generate_otp(self):
        return str(random.randint(1000, 9999))

    def store_otp(self, email, otp_code):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()

        cursor.execute(
            """INSERT INTO otp_verification (email, otp_code, expires_at)
               VALUES (?, ?, ?)""",
            (email, otp_code, expires_at)
        )
        conn.commit()
        conn.close()

    def verify_otp(self, email, otp_code):
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """SELECT id
               FROM otp_verification
               WHERE email = ?
                 AND otp_code = ?
                 AND is_used = 0
                 AND expires_at > ?
               ORDER BY created_at DESC LIMIT 1""",
            (email, otp_code, datetime.now().isoformat())
        )

        result = cursor.fetchone()

        if result:
            cursor.execute("UPDATE otp_verification SET is_used = 1 WHERE id = ?", (result[0],))
            conn.commit()
            conn.close()
            return True

        conn.close()
        return False

    def send_otp_email(self, receiver_email, otp_code):
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Stock Analysis App - Verification Code"
        msg['From'] = self.sender
        msg['To'] = receiver_email

        html_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2 style="color: #2c3e50;">Stock Analysis App</h2>
                <p>Your verification code is:</p>
                <h1 style="color: #3498db; font-size: 36px; letter-spacing: 5px;">{otp_code}</h1>
                <p style="color: #7f8c8d;">This code will expire in 10 minutes.</p>
            </body>
        </html>
        """
        msg.attach(MIMEText(html_body, 'html'))

        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(self.sender, self.password)
            server.sendmail(self.sender, receiver_email, msg.as_string())
            try:
                server.quit()
            except:
                pass
            return True
        except Exception as e:
            print(f"CRITICAL MAIL ERROR: {e}")
            return False

    def send_verification_code(self, email):
        otp_code = self.generate_otp()
        self.store_otp(email, otp_code)
        return self.send_otp_email(email, otp_code)

    def verify_and_activate_user(self, email, otp_code):
        if self.verify_otp(email, otp_code):
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET is_verified = 1 WHERE email = ?", (email,))
            conn.commit()
            conn.close()
            return True
        return False


class AuthManager:
    """Handles user account lifecycle: Sign-up, Sign-in, and Validation."""

    def __init__(self, db_manager):
        self.db = db_manager
        self.email_verifier = EmailVerification(db_manager)

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def validate_email(email: str) -> bool:
        return '@' in email and '.' in email.split('@')[1]

    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        if not any(char.isdigit() for char in password):
            return False, "Password must contain at least one number"
        return True, ""

    def sign_up(self, username: str, email: str, password: str) -> Dict:
        try:
            if not username or not email or not password:
                return {'success': False, 'message': 'All fields are required', 'requires_verification': False}

            if not self.validate_email(email):
                return {'success': False, 'message': 'Invalid email format', 'requires_verification': False}

            is_valid, msg = self.validate_password(password)
            if not is_valid:
                return {'success': False, 'message': msg, 'requires_verification': False}

            hashed_pw = self.hash_password(password)
            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO users (username, email, password, is_verified) VALUES (?, ?, ?, 0)",
                (username.strip(), email.strip().lower(), hashed_pw)
            )
            conn.commit()
            conn.close()

            email_sent = self.email_verifier.send_verification_code(email.strip().lower())

            return {
                'success': True,
                'message': f'Account created! Check {email} for code.' if email_sent else 'Account created, but email failed.',
                'requires_verification': True
            }

        except sqlite3.IntegrityError:
            return {'success': False, 'message': 'Username or email already exists', 'requires_verification': False}
        except Exception as e:
            return {'success': False, 'message': f'Sign up failed: {str(e)}', 'requires_verification': False}

    def verify_email(self, email: str, otp_code: str) -> Dict:
        try:
            if self.email_verifier.verify_and_activate_user(email.strip().lower(), otp_code):
                return {'success': True, 'message': 'Email verified successfully!'}
            return {'success': False, 'message': 'Invalid or expired verification code'}
        except Exception as e:
            return {'success': False, 'message': f'Verification failed: {str(e)}'}

    def sign_in(self, login: str, password: str) -> Dict:
        try:
            hashed_pw = self.hash_password(password)
            conn = self.db.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """SELECT id, username, is_verified
                   FROM users
                   WHERE (username = ? OR email = ?)
                     AND password = ?""",
                (login.strip(), login.strip().lower(), hashed_pw)
            )

            result = cursor.fetchone()
            conn.close()

            if result:
                user_id, username, is_verified = result
                if is_verified == 0:
                    return {'success': False, 'message': 'Please verify your email', 'user_id': None, 'username': None}

                return {'success': True, 'message': f'Welcome, {username}!', 'user_id': user_id, 'username': username}
            return {'success': False, 'message': 'Invalid credentials', 'user_id': None, 'username': None}
        except Exception as e:
            return {'success': False, 'message': f'Sign in failed: {str(e)}', 'user_id': None, 'username': None}

    def resend_verification_code(self, email: str) -> Dict:
        """Trigger a new OTP email for the user."""
        email_sent = self.email_verifier.send_verification_code(email.strip().lower())
        return {
            'success': email_sent,
            'message': 'Verification code resent!' if email_sent else 'Failed to send email.'
        }

    # THIS WAS THE MISSING PART CAUSING YOUR ISSUE
    def get_user_info(self, user_id: int) -> Optional[Dict]:
        """Fetch basic user details for the frontend profile."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, email, is_verified FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return dict(row)
            return None
        except Exception as e:
            print(f"Error fetching user info: {e}")
            return None

    def change_password(self, user_id: int, old_password: str, new_password: str) -> Dict:
        """Verifies old password and updates to new password."""
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()

            # 1. Verify old password
            cursor.execute("SELECT password FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()

            if not result:
                return {'success': False, 'message': 'User not found'}

            stored_hash = result[0]
            if stored_hash != self.hash_password(old_password):
                return {'success': False, 'message': 'Incorrect current password'}

            # 2. Validate new password
            is_valid, msg = self.validate_password(new_password)
            if not is_valid:
                return {'success': False, 'message': msg}

            # 3. Update password
            new_hash = self.hash_password(new_password)
            cursor.execute("UPDATE users SET password = ? WHERE id = ?", (new_hash, user_id))
            conn.commit()
            conn.close()

            return {'success': True, 'message': 'Password updated successfully'}

        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}'}