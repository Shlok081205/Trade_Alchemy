# TradeAlchemy 📈

> Transmuting raw data into market wealth with intelligent analysis.

A full-stack stock analysis platform built as a Semester 3 project. TradeAlchemy combines web scraping, machine learning, and real-time financial data to deliver stock insights and AI-powered volatility predictions.

---

## 🌟 Features

- **Stock Analysis** — Fundamentals, financials, and interactive candlestick/line charts
- **AI Volatility Prediction** — Bidirectional LSTM neural network predicts significant price movements
- **Ecosystem Intelligence** — Gemini AI identifies competitors and supply chain partners for context-aware ML
- **Live Watchlist** — Real-time price tracking with sparkline mini-charts and auto-refresh
- **Secure Authentication** — SHA-256 hashing, OTP email verification, session management
- **Account Management** — Change password and email with OTP verification flow

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| Python 3.x | Core language |
| Flask | Web framework + REST API |
| SQLite | Database (users, watchlist, OTP) |
| TensorFlow / Keras | Bidirectional LSTM model |
| scikit-learn | Feature scaling, class weights, accuracy |
| pandas / numpy | Data processing + feature engineering |
| yfinance | Historical stock price data |
| requests | Yahoo Finance web scraping |
| Google Gemini AI | Market intelligence (peers, partners, regime) |
| smtplib | OTP email delivery via Gmail SMTP |

### Frontend
| Technology | Purpose |
|---|---|
| HTML5 / CSS3 | Structure and styling |
| Vanilla JavaScript | DOM manipulation, async fetch, state management |
| ApexCharts | Candlestick, area, and sparkline charts |
| Google Fonts | Typography (DM Sans, Inter) |
| Bootstrap 5 | Educational pages layout |
| Tailwind CSS | Utility styling (AI/ML page) |

---

## 📁 Project Structure
```
TradeAlchemy/
│
├── app.py                  # Flask application, routes, API endpoints
│
├── Database/
│   └── db_manager.py       # SQLite connection, table initialization
│
├── AccountServices/
│   ├── auth.py             # AuthManager + EmailVerification classes
│   └── watchlist.py        # WatchlistManager + price caching
│
├── Web_Scraping/
│   ├── yahoo_scraper.py    # YahooScraper (v7, v8, v10 endpoints)
│   └── gemini.py           # Gemini AI market intelligence
│
├── Machine_Learning/
│   ├── stock_analyzer.py   # StockAnalyzer orchestrator
│   ├── DataProcessor.py    # FeatureCalculator (RSI, MACD, ATR, MA)
│   └── LSTMConfidenceModel.py  # Bidirectional LSTM model
│
├── templates/
│   ├── landing.html        # Login + Signup + OTP verification
│   ├── dashboard.html      # Search bar + mode selector
│   ├── search.html         # Stock analysis (chart + fundamentals)
│   ├── ai_prediction.html  # ML prediction results + chart
│   ├── watchlist.html      # Live watchlist table + sparklines
│   ├── account.html        # Profile, password, email management
│   ├── stock_market.html   # Educational: Stock Market basics
│   └── ai_ml.html          # Educational: AI & ML in trading
│
├── static/
│   ├── css/
│   │   └── style.css       # Global stylesheet
│   └── images/
│       └── logo.png        # TradeAlchemy logo
│
├── .env                    # Environment variables (not committed)
├── .gitignore
├── requirements.txt
└── app.db                  # SQLite database (auto-created on first run)
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/TradeAlchemy.git
cd TradeAlchemy
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

> Get your Gemini API key at: https://makersuite.google.com/app/apikey

### 5. Configure Gmail SMTP (for OTP emails)

In `AccountServices/auth.py`, update the email credentials or create a `config.py`:
```python
# config.py
EMAIL_SENDER = "your_gmail@gmail.com"
EMAIL_PASSWORD = "your_gmail_app_password"
```

> **Important:** Use a Gmail **App Password**, not your regular password.
> Enable it at: Google Account → Security → 2-Step Verification → App Passwords

### 6. Run the Application
```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

> The SQLite database (`app.db`) is created automatically on first run.

---

## 🔑 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini AI API key | Yes (for AI predictions) |
| `EMAIL_SENDER` | Gmail address for OTP emails | Yes (for auth) |
| `EMAIL_PASSWORD` | Gmail App Password | Yes (for auth) |

---

## 📦 Requirements

Create a `requirements.txt` with:
```
flask
yfinance
pandas
numpy
tensorflow
scikit-learn
requests
google-genai
python-dotenv
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🧠 How the AI Prediction Works

1. **Context Extraction** — Gemini AI identifies the stock's top 3 competitors (peers) and top 3 supply chain partners, plus classifies market regime (stable/volatile)

2. **Data Collection** — Downloads 5 years of daily OHLCV data for the target stock and all identified ecosystem stocks via yfinance

3. **Feature Engineering** — Calculates technical indicators:
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - ATR (Average True Range — volatility)
   - Dist_MA50 (Distance from 50-day Moving Average)
   - Rel_Str_Peers (Relative strength vs competitors)
   - Rel_Str_Partners (Relative strength vs supply chain)

4. **LSTM Training** — Bidirectional LSTM trained on 85% of data with:
   - Time decay weighting (recent data 20x more important)
   - Class-balanced sampling (handles rare volatility spikes)
   - Early stopping + learning rate reduction callbacks

5. **Prediction** — Model predicts tomorrow's volatility probability:
   - `> 0.5` → Direction: **DOWN** (high volatility expected)
   - `< 0.5` → Direction: **UP** (stable conditions expected)
   - Confidence = certainty in the stated direction

---

## 📊 ML Model Architecture
```
Input: (60 days × n_features)
    ↓
Bidirectional LSTM (128 units, return_sequences=True)
    ↓
BatchNormalization → Dropout (30%)
    ↓
Bidirectional LSTM (64 units)
    ↓
BatchNormalization → Dropout (30%)
    ↓
Dense (32 units, swish activation)
    ↓
Dense (1 unit, sigmoid activation)
    ↓
Output: Probability (0 to 1)
```

---

## 🔒 Security Features

- Passwords hashed with **SHA-256** (never stored in plain text)
- OTP codes expire after **10 minutes**
- One-time use OTP codes (cannot be reused)
- Session-based authentication with Flask encrypted cookies
- Database-level UNIQUE constraints prevent duplicate watchlist entries
- Parameterized SQL queries prevent SQL injection

---

## 📱 Pages Overview

| Page | URL | Auth Required | Description |
|---|---|---|---|
| Landing | `/` | No | Login and Sign Up |
| Dashboard | `/dashboard` | Yes | Search bar with mode selector |
| Search | `/search` | Yes | Stock fundamentals + charts |
| AI Prediction | `/ai_prediction` | Yes | LSTM volatility prediction |
| Watchlist | `/watchlist` | Yes | Live tracked stocks |
| Account | `/account` | Yes | Profile and security settings |
| Stock Market | `/stock_market` | No | Educational content |
| AI & ML | `/ai_ml` | No | Educational content |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/signup` | Create new account |
| POST | `/api/login` | Authenticate user |
| POST | `/api/logout` | End session |
| POST | `/api/verify` | Verify OTP + activate account |
| GET | `/api/user_info` | Get current user details |
| POST | `/api/change_password` | Update password |
| POST | `/api/request_email_change` | Send OTP to new email |
| POST | `/api/verify_email_change` | Verify and update email |
| GET | `/api/watchlist` | Get watchlist with live prices |
| POST | `/api/watchlist/add` | Add stock to watchlist |
| POST | `/api/watchlist/remove` | Remove stock from watchlist |
| GET | `/api/search_data` | Get fundamentals + chart data |
| GET | `/api/predict` | Run AI volatility prediction |

---

## ⚠️ Limitations & Known Issues

- The AI prediction retrains from scratch on every request (~30–60 seconds wait)
- SHA-256 password hashing (bcrypt/Argon2 recommended for production)
- In-memory price cache is lost on server restart
- `secret_key` regenerates on every restart (all sessions cleared)
- Yahoo Finance scraper may occasionally fail if Yahoo changes their API

---

## 🚀 Production Considerations

For deployment beyond development:

- Replace SQLite with **PostgreSQL**
- Replace SHA-256 with **Argon2** password hashing
- Load `secret_key` from environment variable (not regenerated)
- Replace in-memory cache with **Redis**
- Use **Gunicorn** or **uWSGI** instead of Flask dev server
- Add **HTTPS** with SSL certificate
- Pre-train and cache LSTM model weights (don't retrain per request)

---

## 👨‍💻 Authors

**TradeAlchemy Team** — Semester 3 Project

---

## 📄 License

This project was built for educational purposes as part of a semester project.

---

## 🙏 Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) — Financial data source
- [Google Gemini](https://deepmind.google/technologies/gemini/) — AI market intelligence
- [ApexCharts](https://apexcharts.com/) — Interactive charting library
- [yfinance](https://github.com/ranaroussi/yfinance) — Python Yahoo Finance wrapper
- [TensorFlow](https://www.tensorflow.org/) — Deep learning framework
