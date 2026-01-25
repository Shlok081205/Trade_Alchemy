from google import genai
from google.genai import types
import json

class Gemini:

    def retrive_data(self,ticker,api_key):
        client = genai.Client(api_key=api_key)

        prompt = f"""
                Analyze the company {ticker} and return a JSON object with two lists:

                1. "partners": List the top 5 major business partners (Key Clients, Critical Suppliers, or Strategic Alliances) 
                   whose relationship is material to {ticker}'s stock price or revenue stability.
                2. "peers": List the top 3 direct competitor/peer companies in the same market.

                Return strictly a JSON object with this schema:
                {{
                  "partners": [
                    {{ 
                        "name": "Company Name", 
                        "role": "Client/Supplier/Strategic Partner", 
                        "ticker": "Stock Ticker (if public, else 'Private')",
                        "impact_reason": "Brief reason why this affects the stock price"
                    }}
                  ],
                  "peers": [
                    {{ "name": "Peer Company Name", "ticker": "Stock Ticker" }}
                  ]
                }}

                Do not use markdown formatting. Return raw JSON only.
                """

        print(f"🤖 Asking Gemini about {ticker}...")

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        return response.text

    def format_info(self,response):
        try:
            partners = json.loads(response)
            return partners
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return None


    def get_info(self,ticker,api_key):
        try:
            return self.format_info(self.retrive_data(ticker,api_key))
        except Exception as e:
            print(e)


if __name__ =="__main__":
    from config_setup import GEMINI_API_KEY
    gs = Gemini()
    print(gs.get_info("TCS.NS",GEMINI_API_KEY))