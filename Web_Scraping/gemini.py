from google import genai
from google.genai import types
import json


class Gemini:

    def retrive_data(self, ticker, api_key):
        """
        Get related companies, peers, indices, and market regime for a ticker.

        Returns JSON with:
        - partners: List of key business partners/suppliers
        - peers: List of competitor companies
        - sectoral_index: Sector-specific index (Yahoo Finance format)
        - market_index: General market index (Yahoo Finance format)
        - market_regime: 'stable' or 'volatile' classification
        """
        client = genai.Client(api_key=api_key)

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

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        return response.text

    def format_info(self, response):
        """Parse and format the JSON response"""
        try:
            data = json.loads(response)
            return data
        except Exception as e:
            print(f"\n⚠️ Error parsing Gemini response: {e}")
            return None

    def get_info(self, ticker, api_key):
        """High-level method to get all information including market regime"""
        try:
            return self.format_info(self.retrive_data(ticker, api_key))
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return None