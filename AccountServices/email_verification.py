import smtplib
from email.mime.text import MIMEText
import random

try:
    from config import EMAIL_SENDER, EMAIL_PASSWORD
except ImportError:
    EMAIL_SENDER = "phoniexblaze5@gmail.com"
    EMAIL_PASSWORD = "your-app-password-here"


class Email_Verfication:

    def generate_otp(self):
        return str(random.randint(1000, 9999))

    def send_otp_email(self,receiver_email, otp_code):
        sender = EMAIL_SENDER
        password = EMAIL_PASSWORD

        msg = MIMEText(f"Your Verification Code is: {otp_code}")
        msg['Subject'] = "Stock Analysis App - Verification Code"
        msg['From'] = sender
        msg['To'] = receiver_email

        try:
            # 1. Connect and Login
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(sender, password)

            # 2. Send the Email
            server.sendmail(sender, receiver_email, msg.as_string())

            # 3. SUCCESS!
            # The email is sent. Now we try to close the connection politely.
            # If closing fails (because Gmail already kicked us), we simply ignore it.
            try:
                server.quit()
            except Exception:
                pass

            return True

        except Exception as e:
            # If we get here, the email actually failed to send (Login error, etc.)
            print(f"CRITICAL MAIL ERROR: {e}")
            return False



if __name__ == "__main__":
    ev = Email_Verfication()
    ev.send_otp_email("patelshlok2119@gmail.com",ev.generate_otp())