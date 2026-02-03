from google import genai
from google.genai import types
import json

class Gemini:

    def retrive_data(self, ticker, api_key):
        """
        Get related companies, peers, sectoral index, and market index for a ticker
        
        Returns JSON with:
        - partners: List of key business partners/suppliers
        - peers: List of competitor companies
        - sectoral_index: Sector-specific index from same country (if available)
        - market_index: General market index from same country (fallback)
        """
        client = genai.Client(api_key=api_key)

        prompt = f"""
                Analyze the company with ticker symbol {ticker} and return a JSON object with FOUR sections:

                1. "partners": List the top 5 major business partners (Key Clients, Critical Suppliers, or Strategic Alliances) 
                   whose relationship is material to {ticker}'s stock price or revenue stability.
                   
                2. "peers": List the top 3 direct competitor/peer companies in the same market/sector.
                
                3. "sectoral_index": Provide the SECTOR-SPECIFIC stock market index from the SAME COUNTRY as {ticker}.
                   - This should be a sector/industry-focused index, NOT the general market index
                   - Examples by country and sector:
                     
                     USA:
                     * Technology sector → QQQ (Nasdaq-100) or XLK (Technology Select Sector)
                     * Financial sector → XLF (Financial Select Sector)
                     * Healthcare sector → XLV (Health Care Select Sector)
                     * Energy sector → XLE (Energy Select Sector)
                     * Consumer sector → XLY (Consumer Discretionary) or XLP (Consumer Staples)
                     
                     India:
                     * IT sector → ^CNXIT (Nifty IT Index)
                     * Banking sector → ^NSEBANK (Nifty Bank Index)
                     * Pharma sector → ^CNXPHARMA (Nifty Pharma Index)
                     * Auto sector → ^CNXAUTO (Nifty Auto Index)
                     * FMCG sector → ^CNXFMCG (Nifty FMCG Index)
                     
                     Japan:
                     * Technology → 1321.T (Nikkei 225 ETF)
                     * Auto → Use peer average or ^N225
                     
                     UK:
                     * Financial → ^FTSE (FTSE 100) - subset
                     * Energy → Use sector ETFs
                   
                   - If no sector-specific index exists, leave this EMPTY (blank string "")
                
                4. "market_index": Provide the main general stock market index from the SAME COUNTRY as {ticker}.
                   - This is the fallback if no sectoral index exists
                   - Examples:
                     * USA → ^GSPC (S&P 500) or ^DJI (Dow Jones)
                     * India → ^NSEI (Nifty 50) or ^BSESN (BSE Sensex)
                     * Japan → ^N225 (Nikkei 225)
                     * UK → ^FTSE (FTSE 100)
                     * Germany → ^GDAXI (DAX)
                     * France → ^FCHI (CAC 40)
                     * China → 000001.SS (SSE Composite)
                     * Hong Kong → ^HSI (Hang Seng)
                     * Canada → ^GSPTSE (S&P/TSX Composite)
                     * Australia → ^AXJO (ASX 200)

                CRITICAL REQUIREMENTS:
                - ALL ticker symbols (partners, peers, sectoral_index, market_index) MUST be valid Yahoo Finance ticker symbols
                - Use the EXACT format Yahoo Finance uses:
                  * US stocks: AAPL, MSFT, GOOGL
                  * Indian stocks: TCS.NS, INFY.NS, RELIANCE.NS
                  * Japanese stocks: 7203.T, 6758.T, 9984.T
                  * UK stocks: BP.L, HSBA.L, VOD.L
                  * Hong Kong stocks: 0700.HK, 0005.HK, 0941.HK
                  * Indices: Start with ^ (^GSPC, ^NSEI, ^N225, ^FTSE)
                  
                - For sectoral_index: Only provide if a REAL sector index exists. If unsure, leave blank ""
                - For market_index: Always provide the main country index
                - For non-public companies in partners list, use ticker: "Private"
                - Verify the country of {ticker} first, then provide indices from that SAME country only

                Return strictly a JSON object with this schema:
                {{
                  "partners": [
                    {{ 
                        "name": "Company Name", 
                        "role": "Client/Supplier/Strategic Partner", 
                        "ticker": "EXACT Yahoo Finance Ticker or 'Private'",
                        "impact_reason": "Brief reason why this affects stock price"
                    }}
                  ],
                  "peers": [
                    {{ 
                        "name": "Peer Company Name", 
                        "ticker": "EXACT Yahoo Finance Ticker"
                    }}
                  ],
                  "sectoral_index": "EXACT sector index ticker from same country (or empty string if none)",
                  "market_index": "EXACT market index ticker from same country"
                }}

                IMPORTANT: Return ONLY the JSON. No markdown, no explanations, no code blocks.
                Double-check all ticker symbols are correct Yahoo Finance format.
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
        """High-level method to get all information"""
        try:
            return self.format_info(self.retrive_data(ticker, api_key))
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return None


if __name__ == "__main__":
    # Test the Gemini class
    from config_setup import GEMINI_API_KEY
    
    test_tickers = ["TCS.NS", "AAPL", "TM", "7203.T"]
    
    gs = Gemini()
    
    for ticker in test_tickers:
        print(f"\n{'='*70}")
        print(f"Testing: {ticker}")
        print('='*70)
        
        result = gs.get_info(ticker, GEMINI_API_KEY)
        
        if result:
            print(json.dumps(result, indent=2))
        else:
            print(f"Failed to get data for {ticker}")
