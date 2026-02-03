import requests
import time
import pandas as pd

# ===== CUSTOM EXCEPTIONS =====
class ScraperException(Exception):
    """Base exception for scraper errors"""
    pass


class SessionSetupError(ScraperException):
    """Raised when session setup fails"""
    pass


class DataFetchError(ScraperException):
    """Raised when data fetching fails"""
    pass


class InvalidTickerError(ScraperException):
    """Raised when ticker is invalid"""
    pass


# ===== MAIN SCRAPER CLASS =====
class YahooScraper:

    def _setup_session(self,use_proxy=False, max_retries=3):
        """Setup Yahoo Finance session with retry logic"""
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

                # Get cookies
                session.get("https://fc.yahoo.com", timeout=30)

                # Get crumb
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


    def data_v10(self,ticker,session,crumb,full_access = False):
        """Fetch v10 data (Fundamentals)"""
        try:
            if full_access:
                modules = [
                    "financialData", "incomeStatementHistory", "quarterlyIncomeStatementHistory",
                    "balanceSheetHistory", "quarterlyBalanceSheetHistory",
                    "cashflowStatementHistory", "quarterlyCashflowStatementHistory"
                ]
                modules_string = ",".join(modules)
                url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules={modules_string}&crumb={crumb}"
            else:
                url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile,financialData&crumb={crumb}"

            response = session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            data = response.json()
            result = data.get('quoteSummary', {}).get('result')

            if not result:
                return None

            data_content = result[0]
            if full_access:
                return data_content

            shortpath_v10 = data_content.get('financialData', {})
            raw_profile = data_content.get('assetProfile', {})

            financial_data = {
                "Industry": raw_profile.get('industry', 'N/A'),
                "Sector": raw_profile.get('sector', 'N/A'),
                "Website": raw_profile.get('website', 'N/A'),
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

    def data_v8(self,ticker,session, time_range = "1d", interval = "1d"):
        """Fetch v8 data (Historical Prices)"""
        try:
            if time_range == "max":
                period1 = 0
                period2 = int(time.time())
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true"
            else:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={time_range}&interval={interval}&events=history&includeAdjustedClose=true"

            response = session.get(url, timeout=15)
            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get("chart", {}).get("result"):
                return None

            shortpath = data["chart"]["result"][0]
            if "timestamp" not in shortpath:
                return None

            historical_data = {
                "TimeStamp": shortpath["timestamp"],
                "Close": shortpath["indicators"]["quote"][0]["close"],
                "Open": shortpath["indicators"]["quote"][0]["open"],
                "High": shortpath["indicators"]["quote"][0]["high"],
                "Low": shortpath["indicators"]["quote"][0]["low"],
                "Volume": shortpath["indicators"]["quote"][0]["volume"],
                "AdjClose": shortpath["indicators"]["adjclose"][0].get("adjclose", [])
            }

            return  historical_data
        except Exception as e:
            print(f"Error v8: {e}")
            return None

    def data_v7(self,ticker,session,crumb):
        """Fetch v7 data (Current Quote)"""
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

    def check_proxy_ip(self,use_proxy = True):
        """Check current IP"""
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
        except:
            return None

        return None


    def v8_formatter(self,ticker_data):
        try:
            if not ticker_data.get('v8'):
                raise Exception("Historical Data Not Present")
            else:
                data = pd.DataFrame(ticker_data.get('v8'))

                data["TimeStamp"] = pd.to_datetime(data["TimeStamp"], unit="s")
                data["TimeStamp"] = data["TimeStamp"].dt.date
                data.rename(columns={"TimeStamp": "Date"}, inplace=True)
                data.set_index("Date",inplace=True)

                cols = ["Close", "Open", "High", "Low", "AdjClose"]
                for i in cols:
                    data[i] = data[i].astype("float32")

                return data

        except Exception as e:
            print("Exception Formatter v8_formatter: ",e)


    def scrape(self, ticker,ip_address = None,time_range = "1d",interval = "1d",use_proxy = False,v10 = False,
               v8= False,v7= False,v10_full_access = False,max_retries: int = 3):
        """
        Main scraping method
        """
        result = {}
        try:
            # 1. Setup Session
            try:
                session, crumb = self._setup_session(use_proxy=use_proxy, max_retries=max_retries)
            except SessionSetupError as e:
                print(f"Session failed: {e}")
                return None

            #print(self.check_proxy_ip(use_proxy=True))

            # 2. Check Proxy
            if use_proxy and ip_address:
                current_ip = self.check_proxy_ip(use_proxy=True)
                if current_ip and current_ip != ip_address:
                    self.proxy = "Proxy Active"
                else:
                    self.proxy = "Proxy Inactive"

            # 3. Fetch Data Modules
            if v10:
                result["v10"] = self.data_v10(ticker, session, crumb, full_access=v10_full_access)

            if v8:
                result["v8"] = self.data_v8(ticker, session, time_range=time_range, interval=interval)

            if v7:
                result["v7"] = self.data_v7(ticker, session, crumb)

            return result

        except Exception as  e:
            print(f"Scrape master error for {ticker}: {e}")


if __name__ == "__main__":
    start = time.perf_counter()
    ys =YahooScraper()
    ip = ys.check_proxy_ip(use_proxy=False)
    print("My IP:",ip)
    s = ys.scrape("ETH-USD",ip_address=ip,time_range="1d",use_proxy=True,v10=True,v8=True,v7=True,v10_full_access=False)
    end = time.perf_counter()
    print("Tor Proxy Status:",ys.proxy)
    print("v10",s["v10"],"\n\n\n\n")
    print("v8",s["v8"],"\n\n\n\n")
    print("v7",s["v7"],"\n\n\n\n")
    print(ys.v8_formatter(s))