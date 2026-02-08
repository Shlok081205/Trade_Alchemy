import requests
import time
import pandas as pd


# Custom exception classes for better error handling
class ScraperException(Exception):
    pass


class SessionSetupError(ScraperException):
    pass


class DataFetchError(ScraperException):
    pass


class InvalidTickerError(ScraperException):
    pass


class YahooScraper:

    def __init__(self):
        self.proxy = "Not Checked"

    def _setup_session(self, use_proxy=False, max_retries=3):
        for attempt in range(max_retries):
            try:
                session = requests.Session()

                if use_proxy:
                    session.proxies = {
                        "http": "socks5h://127.0.0.1:9050",
                        "https": "socks5h://127.0.0.1:9050"
                    }

                session.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/120.0.0.0 Safari/537.36"
                })

                session.get("https://fc.yahoo.com", timeout=30)
                response = session.get(
                    "https://query1.finance.yahoo.com/v1/test/getcrumb",
                    timeout=30
                )

                if response.status_code == 200:
                    crumb = response.text
                    return session, crumb
                else:
                    print(f"Failed to get crumb: {response.status_code}")

            except Exception as e:
                print(f"Setup error on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        raise SessionSetupError("Failed to setup session after all retries")

    def data_v10(self, ticker, session, crumb, full_access=False):
        try:
            if full_access:
                modules = [
                    "financialData", "incomeStatementHistory", "quarterlyIncomeStatementHistory",
                    "balanceSheetHistory", "quarterlyBalanceSheetHistory",
                    "cashflowStatementHistory", "quarterlyCashflowStatementHistory",
                    "assetProfile", "summaryProfile"  # ADDED summaryProfile
                ]
                modules_string = ",".join(modules)
                url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={modules_string}&crumb={crumb}"
            else:
                # Request both assetProfile and summaryProfile for description fallback
                url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile,financialData,summaryProfile&crumb={crumb}"

            response = session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            data = response.json()
            result = data.get('quoteSummary', {}).get('result')

            if not result:
                return None

            data_content = result[0]

            if full_access:
                # Inject the fallback description logic even for full access return
                # (Optional: depends on how you use full_access return elsewhere)
                if 'assetProfile' in data_content:
                    desc = data_content['assetProfile'].get('longBusinessSummary')
                    if not desc and 'summaryProfile' in data_content:
                        data_content['assetProfile']['longBusinessSummary'] = data_content['summaryProfile'].get(
                            'longBusinessSummary')
                return data_content

            shortpath_v10 = data_content.get('financialData', {})
            raw_profile = data_content.get('assetProfile', {})
            summary_profile = data_content.get('summaryProfile', {})

            # --- ROBUST DESCRIPTION EXTRACTION ---
            description = raw_profile.get('longBusinessSummary')
            if not description:
                description = summary_profile.get('longBusinessSummary')

            if not description:
                description = 'N/A'
            # -------------------------------------

            financial_data = {
                "Industry": raw_profile.get('industry', 'N/A'),
                "Sector": raw_profile.get('sector', 'N/A'),
                "Website": raw_profile.get('website', 'N/A'),
                "Description": description,
                "Target Mean Price": shortpath_v10.get('targetMeanPrice', {}).get('raw'),
                "Recommendation": shortpath_v10.get('recommendationKey'),
                "Number of Analyst Opinions": shortpath_v10.get('numberOfAnalystOpinions', {}).get('raw'),
                "Profit Margins": shortpath_v10.get('profitMargins', {}).get('raw'),
                "Gross Margins": shortpath_v10.get('grossMargins', {}).get('raw'),
                "Operating Margins": shortpath_v10.get('operatingMargins', {}).get('raw'),
                "EBITDA Margins": shortpath_v10.get('ebitdaMargins', {}).get('raw'),
                "Revenue Growth": shortpath_v10.get('revenueGrowth', {}).get('raw'),
                "Earnings Growth": shortpath_v10.get('earningsGrowth', {}).get('raw'),
                "Return on Equity": shortpath_v10.get('returnOnEquity', {}).get('raw'),
                "Return on Assets": shortpath_v10.get('returnOnAssets', {}).get('raw'),
                "Total Cash": shortpath_v10.get('totalCash', {}).get('raw'),
                "Total Debt": shortpath_v10.get('totalDebt', {}).get('raw'),
                "Debt to Equity": shortpath_v10.get('debtToEquity', {}).get('raw'),
                "Current Ratio": shortpath_v10.get('currentRatio', {}).get('raw'),
                "Free Cash Flow": shortpath_v10.get('freeCashflow', {}).get('raw'),
                "Revenue Per Share": shortpath_v10.get('revenuePerShare', {}).get('raw'),
                "Total Cash Per Share": shortpath_v10.get('totalCashPerShare', {}).get('raw')
            }
            return financial_data

        except Exception as e:
            print(f"Error v10: {e}")
            return None

    def data_v8(self, ticker, session, time_range="1d", interval="1d"):
        """
        IMPROVED VERSION: More lenient with international stocks
        """
        try:
            # Build URL based on time range
            if time_range == "max":
                period1 = 0
                period2 = int(time.time())
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true"
            else:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={time_range}&interval={interval}&events=history&includeAdjustedClose=true"

            # Make request
            response = session.get(url, timeout=15)
            if response.status_code != 200:
                # print(f"⚠️ HTTP {response.status_code} for {ticker}")
                return None

            data = response.json()

            # Check for errors in response
            if data.get("chart", {}).get("error"):
                # error_msg = data["chart"]["error"].get("description", "Unknown error")
                # print(f"⚠️ Yahoo API error for {ticker}: {error_msg}")
                return None

            if not data.get("chart", {}).get("result"):
                # print(f"⚠️ No chart results for {ticker}")
                return None

            shortpath = data["chart"]["result"][0]

            # MORE LENIENT APPROACH: Check if we have any valid data instead of failing immediately
            raw_timestamps = shortpath.get("timestamp")

            # If no timestamps at all, then we truly can't proceed
            if not raw_timestamps or len(raw_timestamps) == 0:
                # print(f"⚠️ No timestamps available for {ticker}")
                return None

            # Extract quotes safely
            quote_data = shortpath.get("indicators", {}).get("quote", [{}])[0]

            # Helper to safely get list or empty list
            def get_col(name):
                return quote_data.get(name, [])

            # Get all data columns
            raw_close = get_col("close")
            raw_open = get_col("open")
            raw_high = get_col("high")
            raw_low = get_col("low")
            raw_volume = get_col("volume")

            # Adjusted close might be in a different substructure
            adj_close_data = shortpath.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])

            # CRITICAL FIX: Filter out None/invalid data but don't fail if some data is missing
            clean_data = []
            for i in range(len(raw_timestamps)):
                # We need at least a timestamp and close price
                if i < len(raw_close) and raw_close[i] is not None and raw_close[i] > 0:
                    clean_data.append({
                        "timestamp": raw_timestamps[i],
                        "close": raw_close[i],
                        "open": raw_open[i] if i < len(raw_open) and raw_open[i] is not None else raw_close[i],
                        "high": raw_high[i] if i < len(raw_high) and raw_high[i] is not None else raw_close[i],
                        "low": raw_low[i] if i < len(raw_low) and raw_low[i] is not None else raw_close[i],
                        "volume": raw_volume[i] if i < len(raw_volume) and raw_volume[i] is not None else 0,
                        "adjclose": adj_close_data[i] if i < len(adj_close_data) and adj_close_data[i] is not None else
                        raw_close[i]
                    })

            # If we have no valid data after filtering, return None
            if len(clean_data) == 0:
                # print(f"⚠️ All data points invalid for {ticker}")
                return None

            # Build the return dictionary with clean data
            historical_data = {
                "TimeStamp": [d["timestamp"] for d in clean_data],
                "Close": [d["close"] for d in clean_data],
                "Open": [d["open"] for d in clean_data],
                "High": [d["high"] for d in clean_data],
                "Low": [d["low"] for d in clean_data],
                "Volume": [d["volume"] for d in clean_data],
                "AdjClose": [d["adjclose"] for d in clean_data]
            }

            # print(f"✓ Successfully fetched {len(clean_data)} valid data points for {ticker}")
            return historical_data

        except Exception as e:
            # print(f"Error v8 for {ticker}: {e}")
            return None

    def data_v7(self, ticker, session, crumb):
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}&crumb={crumb}"

            response = session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get('quoteResponse', {}).get('result'):
                return None

            shortpath_v7 = data['quoteResponse']['result'][0]

            snapshot_data = {
                "Symbol": shortpath_v7.get('symbol'),
                "Name": shortpath_v7.get('longName'),
                "Current Price": shortpath_v7.get('regularMarketPrice'),
                "Open": shortpath_v7.get('regularMarketOpen'),
                "Prev Close": shortpath_v7.get('regularMarketPreviousClose'),
                "Day High": shortpath_v7.get('regularMarketDayHigh'),
                "Day Low": shortpath_v7.get('regularMarketDayLow'),
                "Volume": shortpath_v7.get('regularMarketVolume'),
                "Avg Volume (3M)": shortpath_v7.get('averageDailyVolume3Month'),
                "Avg Volume (10D)": shortpath_v7.get('averageDailyVolume10Day'),
                "50 Day Avg": shortpath_v7.get('fiftyDayAverage'),
                "200 Day Avg": shortpath_v7.get('twoHundredDayAverage'),
                "52W High": shortpath_v7.get('fiftyTwoWeekHigh'),
                "52W Low": shortpath_v7.get('fiftyTwoWeekLow'),
                "Trailing PE": shortpath_v7.get('trailingPE'),
                "Forward PE": shortpath_v7.get('forwardPE'),
                "Market Cap": shortpath_v7.get('marketCap'),
                "Price to Book": shortpath_v7.get('priceToBook'),
                "EPS (TTM)": shortpath_v7.get('epsTrailingTwelveMonths')
            }

            return snapshot_data

        except Exception as e:
            print(f"Error v7: {e}")
            return None

    def check_proxy_ip(self, use_proxy=True):
        try:
            session = requests.Session()

            if use_proxy:
                session.proxies = {
                    "http": "socks5h://127.0.0.1:9050",
                    "https": "socks5h://127.0.0.1:9050"
                }

            response = session.get("https://api.ipify.org?format=json", timeout=15)
            if response.status_code == 200:
                return response.json().get("ip")

        except Exception as e:
            print(f"Error checking IP: {e}")
            return None

        return None

    def v8_formatter(self, ticker_data):
        try:
            if not ticker_data.get('v8'):
                raise Exception("Historical Data Not Present")

            data = pd.DataFrame(ticker_data.get('v8'))
            data["TimeStamp"] = pd.to_datetime(data["TimeStamp"], unit="s")
            data["TimeStamp"] = data["TimeStamp"].dt.date
            data.rename(columns={"TimeStamp": "Date"}, inplace=True)
            data.set_index("Date", inplace=True)

            cols = ["Close", "Open", "High", "Low", "AdjClose"]
            for col in cols:
                data[col] = data[col].astype("float32")

            return data

        except Exception as e:
            print(f"Exception Formatter v8_formatter: {e}")
            return None

    def scrape(self, ticker, ip_address=None, time_range="1d", interval="1d",
               use_proxy=False, v10=False, v8=False, v7=False,
               v10_full_access=False, max_retries=3):

        result = {}

        try:
            try:
                session, crumb = self._setup_session(use_proxy=use_proxy, max_retries=max_retries)
            except SessionSetupError as e:
                print(f"Session setup failed: {e}")
                return None

            if use_proxy and ip_address:
                current_ip = self.check_proxy_ip(use_proxy=True)
                if current_ip and current_ip != ip_address:
                    self.proxy = "Proxy Active"
                else:
                    self.proxy = "Proxy Inactive"

            if v10:
                result["v10"] = self.data_v10(ticker, session, crumb, full_access=v10_full_access)

            if v8:
                result["v8"] = self.data_v8(ticker, session, time_range=time_range, interval=interval)

            if v7:
                result["v7"] = self.data_v7(ticker, session, crumb)

            return result

        except Exception as e:
            print(f"Scrape error for {ticker}: {e}")
            return None


if __name__ == "__main__":
    start = time.perf_counter()

    ys = YahooScraper()
    ip = ys.check_proxy_ip(use_proxy=False)
    print(f"My IP: {ip}")

    # Test with international stock
    print("\n=== Testing TCS.NS (Indian Stock) ===")
    s = ys.scrape(
        "TCS.NS",
        ip_address=ip,
        time_range="1mo",
        use_proxy=True,
        v10=True,
        v8=True,
        v7=True,
        v10_full_access=False
    )

    end = time.perf_counter()

    print(f"\nProxy Status: {ys.proxy}")
    print(f"Time taken: {end - start:.2f} seconds")

    if s:
        print("\nv7 (Current Quote):", s.get("v7"))
        if s.get("v10"):
            print("\nDescription:", s.get("v10").get("Description"))