from Web_Scraping import YahooScraper,Gemini
from Web_Scraping import g_key
import json


def main():
    ys = YahooScraper()
    gs = Gemini()

    ticker = input("Enter Ticker: ").strip().upper()
    ip = ys.check_proxy_ip(use_proxy=False)
    data_ys = ys.scrape(ticker,ip,v7=True,use_proxy=True)

    data_gs = gs.retrive_data(ticker, g_key)
    data_dict = json.loads(data_gs)

    names_partners = [ x['name'] for x in data_dict["partners"]]
    names_peers = [x['name'] for x in data_dict["peers"]]


    print(data_ys)
    print(names_partners)
    print(names_peers)


main()


