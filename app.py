import os
import secrets
import math
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv  # Import the loader
load_dotenv()


from Database import DatabaseManager
from AccountServices import AuthManager
from AccountServices import WatchlistManager
from Machine_Learning import StockAnalyzer

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# --- INITIALIZE SYSTEMS ---
try:
    db = DatabaseManager()
    auth = AuthManager(db)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_ACTUAL_API_KEY_HERE")
    analyzer = StockAnalyzer(gemini_api_key=GEMINI_API_KEY)
    print("✅ System modules initialized successfully.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")


# --- HELPER: Handle NaN for JSON ---
def clean_data(data):
    """Recursively replace NaN/Infinity with None for valid JSON serialization."""
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    return data


# --- AUTH DECORATOR ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('landing'))
        return f(*args, **kwargs)

    return decorated_function


# --- ROUTES ---

@app.route('/')
def landing():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return send_from_directory('templates', 'landing.html')


@app.route('/dashboard')
@app.route('/dashboard.html')
@login_required
def dashboard():
    return send_from_directory('templates', 'dashboard.html')


@app.route('/search')
@app.route('/search.html')
@login_required
def search_page():
    return send_from_directory('templates', 'search.html')


@app.route('/ai_prediction')
@app.route('/ai_prediction.html')
@login_required
def ai_prediction_page():
    return send_from_directory('templates', 'ai_prediction.html')


@app.route('/watchlist')
@app.route('/watchlist.html')
@login_required
def watchlist_page():
    return send_from_directory('templates', 'watchlist.html')


@app.route('/account')
@app.route('/account.html')
@login_required
def account_page():
    return send_from_directory('templates', 'account.html')


@app.route('/style.css')
def serve_css():
    return send_from_directory('static/css', 'style.css')


@app.route('/static/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('static/images', filename)


# --- API ENDPOINTS ---

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    result = auth.sign_up(data.get('username'), data.get('email'), data.get('password'))
    return jsonify(result)


@app.route('/api/verify', methods=['POST'])
def verify_otp():
    data = request.json
    result = auth.verify_email(data.get('email'), data.get('otp'))
    return jsonify(result)


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    result = auth.sign_in(data.get('login'), data.get('password'))

    if result.get('success'):
        session['user_id'] = result['user_id']
        session['username'] = result['username']
        return jsonify({'success': True, 'message': 'Login successful'})

    return jsonify(result)


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/user_info', methods=['GET'])
@login_required
def user_info():
    user = auth.get_user_info(session['user_id'])
    if user:
        return jsonify({'success': True, 'user': clean_data(dict(user))})
    return jsonify({'success': False, 'message': 'User not found'})


@app.route('/api/change_password', methods=['POST'])
@login_required
def change_password():
    data = request.json
    result = auth.change_password(
        session['user_id'],
        data.get('old_password'),
        data.get('new_password')
    )
    return jsonify(result)


@app.route('/api/watchlist', methods=['GET'])
@login_required
def get_watchlist():
    wm = WatchlistManager(db, session['user_id'])
    return jsonify(clean_data({'success': True, 'data': wm.get_watchlist_with_prices()}))


@app.route('/api/watchlist/add', methods=['POST'])
@login_required
def add_to_watchlist():
    data = request.json
    wm = WatchlistManager(db, session['user_id'])
    result = wm.add_stock(data.get('ticker'), 0.0, data.get('date'))
    return jsonify(result)


@app.route('/api/watchlist/remove', methods=['POST'])
@login_required
def remove_from_watchlist():
    data = request.json
    wm = WatchlistManager(db, session['user_id'])
    result = wm.remove_stock(data.get('ticker'))
    return jsonify(result)


# --- SEARCH DATA API (UPDATED) ---
@app.route('/api/search_data', methods=['GET'])
@login_required
def get_search_data():
    ticker = request.args.get('ticker')
    # Allow specifying period (default to max if not provided)
    period = request.args.get('period', 'max')

    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker required'})

    try:
        # 1. Fetch Fundamentals
        scraper_data = {}
        try:
            data = analyzer.scraper.scrape(
                ticker,
                v7=True,
                v10=True,
                v10_full_access=False,
                use_proxy=False
            )
            if data:
                scraper_data = data
        except Exception as e:
            print(f"⚠️ Scraper warning for {ticker}: {e}")

        # 2. Fetch Historical Data (Chart)
        stock = yf.Ticker(ticker)
        # Use the requested period (e.g., '5y' or 'max')
        hist = stock.history(period=period, interval="1d")

        chart_data = []
        if hasattr(hist, 'empty') and not hist.empty:
            hist = hist.reset_index()
            for _, row in hist.iterrows():
                if pd.isna(row['Close']): continue

                ts = int(row['Date'].timestamp() * 1000)
                chart_data.append({
                    'x': ts,
                    'y': [row['Open'], row['High'], row['Low'], row['Close']],
                    'v': row['Volume']
                })

        if not scraper_data and not chart_data:
            return jsonify({'success': False, 'message': 'No data found for this ticker'})

        response_data = {
            'success': True,
            'ticker': ticker.upper(),
            'fundamentals': scraper_data,
            'chart_data': chart_data
        }

        return jsonify(clean_data(response_data))

    except Exception as e:
        print(f"CRITICAL API ERROR: {e}")
        return jsonify({'success': False, 'message': f"Server Error: {str(e)}"})


# --- AI PREDICTION API ---
@app.route('/api/predict', methods=['GET'])
@login_required
def predict():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker required'})

    result = analyzer.analyze_for_api(ticker)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)