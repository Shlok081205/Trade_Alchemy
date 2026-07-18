# TradeAlchemy 📈

> Transmuting raw data into market wealth with intelligent analysis.

A full-stack stock analysis platform built as a Semester 3 project. TradeAlchemy combines real-time web scraping, a Bidirectional LSTM neural network, and Google Gemini AI to deliver comprehensive stock insights and volatility predictions.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 📊 **Stock Analysis** | Fundamentals, financials, and interactive candlestick/area charts via ApexCharts |
| 🤖 **AI Volatility Prediction** | Bidirectional LSTM neural network predicts next-day significant price movements |
| 🌐 **Ecosystem Intelligence** | Gemini AI identifies competitors & supply chain partners for context-aware ML |
| 📋 **Live Watchlist** | Real-time price tracking with sparkline mini-charts and auto-refresh |
| 🔐 **Secure Authentication** | SHA-256 hashing, OTP email verification, and session management |
| 👤 **Account Management** | Change password and email address with OTP verification flow |

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.12+ | Core language |
| Flask | 3.x | Web framework + REST API |
| SQLite | — | Database (users, watchlist, OTP) |
| TensorFlow / tf-keras | 2.21 | Bidirectional LSTM model |
| scikit-learn | 1.x | Feature scaling, class weights |
| pandas / numpy | Latest | Data processing + feature engineering |
| yfinance | 1.x | Historical stock price data |
| requests | 2.x | Yahoo Finance web scraping |
| Google Gemini AI (google-genai) | 2.x | Market intelligence (peers, partners) |
| smtplib | stdlib | OTP email delivery via Gmail SMTP |

### Frontend
| Technology | Purpose |
|---|---|
| HTML5 / CSS3 | Structure and styling |
| Vanilla JavaScript | DOM manipulation, async fetch, state management |
| ApexCharts | Candlestick, area, and sparkline charts |
| Google Fonts (DM Sans, Inter) | Typography |
| Bootstrap 5 | Educational pages layout |
| Tailwind CSS | Utility styling (AI/ML page) |

---

## 📁 Project Structure

```
TradeAlchemy/
│
├── app.py                      # Flask application, routes, API endpoints
├── config.py                   # All configuration constants
│
├── Database/
│   └── db_manager.py           # SQLite connection, table initialization
│
├── AccountServices/
│   ├── auth.py                 # AuthManager + EmailVerification classes
│   └── watchlist.py            # WatchlistManager + price caching
│
├── Web_Scraping/
│   ├── yahoo_scraper.py        # YahooScraper (v7, v8, v10 endpoints)
│   └── gemini.py               # Gemini AI market intelligence
│
├── Machine_Learning/
│   ├── stock_analyzer.py       # StockAnalyzer orchestrator
│   ├── DataProcessor.py        # FeatureCalculator (RSI, MACD, ATR, MA)
│   └── LSTMConfidenceModel.py  # Bidirectional LSTM model
│
├── templates/
│   ├── landing.html            # Login + Signup + OTP verification
│   ├── dashboard.html          # Search bar + mode selector
│   ├── search.html             # Stock analysis (chart + fundamentals)
│   ├── ai_prediction.html      # ML prediction results + chart
│   ├── watchlist.html          # Live watchlist table + sparklines
│   ├── account.html            # Profile, password, email management
│   ├── stock_market.html       # Educational: Stock Market basics
│   └── ai_ml.html              # Educational: AI & ML in trading
│
├── static/
│   ├── css/style.css           # Global stylesheet
│   └── images/logo.png         # TradeAlchemy logo
│
├── .env                        # Environment variables (not committed)
├── .gitignore
├── requirements.txt
└── app.db                      # SQLite database (auto-created on first run)
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
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Upgrade pip & Install Build Tools

> **Required for Python 3.12+** — `distutils` was removed from the standard library.

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install Core Dependencies

```bash
pip install flask python-dotenv yfinance requests pandas numpy scikit-learn tensorflow tf-keras google-genai
```

> **Note:** The `requirements.txt` contains pinned versions from the original development environment (Python 3.11).
> For Python 3.12, use the command above to install the latest compatible versions automatically.

### 5. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Gmail SMTP (for OTP emails)
EMAIL_SENDER=your_gmail@gmail.com
EMAIL_PASSWORD=your_gmail_app_password

# Google Gemini AI
GEMINI_API_KEY=your_gemini_api_key_here

# Flask
SECRET_KEY=your_secret_key_here
FLASK_ENV=development
FLASK_DEBUG=True

# Database
DATABASE_PATH=app.db

# Session & OTP
SESSION_LIFETIME_DAYS=7
OTP_EXPIRATION_MINUTES=10
```

- **Gemini API Key:** https://makersuite.google.com/app/apikey
- **Gmail App Password:** Google Account → Security → 2-Step Verification → App Passwords

### 6. Run the Application

```powershell
# Windows (PowerShell) — PYTHONUTF8=1 is required for emoji output in terminal
$env:PYTHONUTF8=1; .venv\Scripts\python.exe app.py
```

```bash
# macOS / Linux
python app.py
```

Visit **http://127.0.0.1:5000** in your browser.

> The SQLite database (`app.db`) is auto-created on first run.

---

## 🔑 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini AI API key | Yes (for AI predictions) |
| `EMAIL_SENDER` | Gmail address for OTP emails | Yes (for auth) |
| `EMAIL_PASSWORD` | Gmail App Password | Yes (for auth) |
| `SECRET_KEY` | Flask session secret key | Yes |
| `DATABASE_PATH` | SQLite database file path | No (default: `app.db`) |
| `SESSION_LIFETIME_DAYS` | Session cookie duration | No (default: 7) |
| `OTP_EXPIRATION_MINUTES` | OTP validity window | No (default: 10) |

---

## 🧠 How the AI Prediction Works

1. **Context Extraction** — Gemini AI identifies the stock's top 3 competitors (peers) and top 3 supply chain partners, and classifies the current market regime (stable/volatile)

2. **Data Collection** — Downloads 5 years of daily OHLCV data for the target stock and all identified ecosystem stocks via `yfinance`

3. **Feature Engineering** — Calculates 6 technical indicators:
   - **RSI** — Relative Strength Index
   - **MACD** — Moving Average Convergence Divergence
   - **ATR** — Average True Range (volatility measure)
   - **Dist_MA50** — Distance from 50-day Moving Average
   - **Rel_Str_Peers** — Relative strength vs competitors
   - **Rel_Str_Partners** — Relative strength vs supply chain

4. **LSTM Training** — Bidirectional LSTM trained on 85% of historical data with:
   - Time decay weighting (recent data weighted up to 20x more)
   - Class-balanced sampling (handles rare volatility spikes)
   - Early stopping + learning rate reduction callbacks

5. **Prediction** — Model outputs tomorrow's volatility probability:
   - `> 0.5` → **HIGH VOLATILITY** expected (bearish signal)
   - `< 0.5` → **STABLE** conditions expected (bullish signal)
   - Confidence = certainty in the stated direction

---

## 📊 ML Model Architecture

```
Input: (60 days x n_features)
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
Output: Volatility Probability (0.0 → 1.0)
```

---

## 🔌 API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/signup` | No | Create new account |
| POST | `/api/login` | No | Authenticate user |
| POST | `/api/logout` | Yes | End session |
| POST | `/api/verify` | No | Verify OTP + activate account |
| GET | `/api/user_info` | Yes | Get current user details |
| POST | `/api/change_password` | Yes | Update password |
| POST | `/api/request_email_change` | Yes | Send OTP to new email |
| POST | `/api/verify_email_change` | Yes | Verify and update email |
| GET | `/api/watchlist` | Yes | Get watchlist with live prices |
| POST | `/api/watchlist/add` | Yes | Add stock to watchlist |
| POST | `/api/watchlist/remove` | Yes | Remove stock from watchlist |
| GET | `/api/search_data` | Yes | Get fundamentals + chart data |
| GET | `/api/predict` | Yes | Run AI volatility prediction |

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

## 🔒 Security Features

- Passwords hashed with **SHA-256** (never stored in plain text)
- OTP codes expire after **10 minutes** and are single-use
- Session-based authentication with Flask encrypted cookies
- Parameterized SQL queries prevent SQL injection
- Database-level UNIQUE constraints prevent duplicate watchlist entries

---

## ⚠️ Known Limitations

| Issue | Details |
|---|---|
| Slow AI predictions | Model retrains from scratch on every request (~30–60 seconds) |
| No GPU on native Windows | TensorFlow >= 2.11 dropped native Windows GPU support; use WSL2 for GPU |
| Basic password hashing | SHA-256 used instead of bcrypt/Argon2 (fine for academic use) |
| In-memory price cache | Watchlist price cache is lost on server restart |
| Yahoo Finance fragility | Scraper may break if Yahoo changes their internal API endpoints |

---

## 🐍 Python 3.12 Compatibility Notes

The original `requirements.txt` was generated under Python 3.11. When running on Python 3.12:

1. **`distutils` removed** — Run `pip install --upgrade setuptools` first
2. **`tensorflow.keras` moved** — Install `tf-keras` for backward compatibility
3. **Unicode in terminal (Windows)** — Launch with `$env:PYTHONUTF8=1` in PowerShell
4. **No native Windows GPU** — TensorFlow >= 2.11 is CPU-only on Windows; use WSL2 for GPU acceleration

---

## 🚀 Production Considerations

- Replace **SQLite** with PostgreSQL
- Replace **SHA-256** with Argon2 password hashing
- Load `SECRET_KEY` from environment variable (already supported via `.env`)
- Replace **in-memory cache** with Redis for watchlist prices
- Use **Gunicorn** or **uWSGI** instead of Flask dev server
- Pre-train and **cache LSTM model weights** (avoid ~60s retraining per request)
- Add **HTTPS** with an SSL certificate
- Set `FLASK_DEBUG=False` in production

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
