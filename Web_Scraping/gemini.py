from google import genai
from google.genai import types
import json


class Gemini:

    def __init__(self):
        # Initialize Gemini AI integration
        pass

    # Call Gemini API to retrieve market intelligence for a stock ticker
    def retrieve_data(self, ticker, api_key):

        # Initialize Gemini client with API key
        client = genai.Client(api_key=api_key)

        # Construct detailed prompt for Gemini to analyze the stock
        prompt = f"""
        Analyze the company with ticker symbol {ticker} and return a JSON object with FIVE sections:

        1. "partners": List the top 5 major business partners (Key Clients, Critical Suppliers, or Strategic Alliances).

        2. "peers": List the top 3 direct competitor/peer companies in the same market/sector.

        3. "sectoral_index": Provide the SECTOR-SPECIFIC stock market index from the SAME COUNTRY as {ticker}.
           - Examples: ^CNXIT (India IT), XLK (US Tech), XLV (US Healthcare).
           - If no specific index exists, leave as empty string "".

        4. "market_index": Provide the main general stock market index from the SAME COUNTRY as {ticker}.
           - Examples: ^GSPC (S&P 500), ^NSEI (Nifty 50), ^N225 (Nikkei 225).

        5. "market_regime": Classify the stock's price behavior into one of TWO categories:
           - "stable": Use for blue-chip companies, value stocks, low-beta entities, or established industry leaders with steady price action (e.g., JNJ, PG, KO, TCS.NS).
           - "volatile": Use for high-growth tech stocks, momentum stocks, speculative assets, or highly cyclical companies with large daily swings (e.g., TSLA, NVDA, coin-linked stocks).

        CRITICAL REQUIREMENTS:
        - ALL ticker symbols MUST be valid Yahoo Finance format (e.g., TCS.NS, AAPL, ^GSPC).
        - Use "Private" for non-public partners.
        - Verify the country of {ticker} and provide indices from that same country only.

        Return strictly a JSON object with this schema:
        {{
          "partners": [
            {{ 
                "name": "Company Name", 
                "role": "Role", 
                "ticker": "Ticker",
                "impact_reason": "Reason"
            }}
          ],
          "peers": [
            {{ "name": "Name", "ticker": "Ticker" }}
          ],
          "sectoral_index": "Ticker",
          "market_index": "Ticker",
          "market_regime": "stable" or "volatile"
        }}

        IMPORTANT: Return ONLY the JSON. No markdown, no code blocks, no explanations.
        """

        # Call Gemini API with the prompt
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        return response.text

    # Parse and validate the JSON response from Gemini
    def format_info(self, response):
        try:
            # Parse JSON string into Python dictionary
            data = json.loads(response)
            return data
        except json.JSONDecodeError as e:
            print(f"\n⚠️ Error parsing Gemini response: {e}")
            return None
        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}")
            return None

    # High-level method to get complete market intelligence including regime
    def get_info(self, ticker, api_key):
        try:
            # Fetch data from Gemini API
            raw_response = self.retrieve_data(ticker, api_key)

            # Parse and return formatted data
            return self.format_info(raw_response)

        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return None

    # Flask-specific method: Get only market regime (stable or volatile)
    def get_market_regime(self, ticker, api_key):
        try:
            # Get full info from Gemini
            info = self.get_info(ticker, api_key)

            if info and 'market_regime' in info:
                return info['market_regime']

            # Default to 'volatile' if unable to determine
            return 'volatile'

        except Exception as e:
            print(f"⚠️ Error getting market regime: {e}")
            return 'volatile'  # Default fallback

    # Flask-specific method: Get peers for comparison analysis
    def get_peers(self, ticker, api_key):
        try:
            # Get full info from Gemini
            info = self.get_info(ticker, api_key)

            if info and 'peers' in info:
                return info['peers']

            return []

        except Exception as e:
            print(f"⚠️ Error getting peers: {e}")
            return []

    # Flask-specific method: Get business partners/suppliers
    def get_partners(self, ticker, api_key):
        try:
            # Get full info from Gemini
            info = self.get_info(ticker, api_key)

            if info and 'partners' in info:
                return info['partners']

            return []

        except Exception as e:
            print(f"⚠️ Error getting partners: {e}")
            return []


# Test Gemini integration
if __name__ == "__main__":
    # You need to set your Gemini API key
    GEMINI_API_KEY = "your-gemini-api-key-here"

    # Initialize Gemini
    gemini = Gemini()

    # Test with a stock ticker
    ticker = "AAPL"

    print(f"Testing Gemini AI for {ticker}...")
    print("=" * 60)

    # Get full market intelligence
    info = gemini.get_info(ticker, GEMINI_API_KEY)

    if info:
        print(f"\nMarket Regime: {info.get('market_regime')}")
        print(f"\nPeers: {info.get('peers')}")
        print(f"\nPartners: {info.get('partners')}")
        print(f"\nSector Index: {info.get('sectoral_index')}")
        print(f"\nMarket Index: {info.get('market_index')}")
    else:
        print("Failed to retrieve data from Gemini")