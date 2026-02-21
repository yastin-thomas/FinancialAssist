import os
import requests
from newspaper import Article
from urllib.parse import urlparse

# Curated list of 70+ Financial Education URLs
# Categories: Basics, Investing, Markets, Retirement, Taxes, Economics
urls = [
    # --- Financial Basics ---
    "https://www.investopedia.com/articles/basics/06/invest1000.asp",
    "https://www.investopedia.com/articles/basics/11/3-s-simple-investing.asp",
    "https://www.investopedia.com/terms/c/compoundinterest.asp",
    "https://www.investopedia.com/terms/i/inflation.asp",
    "https://www.investopedia.com/terms/d/diversification.asp",
    "https://www.investopedia.com/terms/a/assetallocation.asp",
    "https://www.investopedia.com/terms/l/liquidity.asp",
    "https://www.investopedia.com/terms/n/networth.asp",
    "https://www.investopedia.com/terms/b/budget.asp",
    "https://www.investopedia.com/terms/e/emergency_fund.asp",
    "https://www.investopedia.com/terms/c/creditscore.asp",
    "https://www.investopedia.com/terms/i/interestrate.asp",
    "https://www.investopedia.com/terms/a/apr.asp",
    "https://www.investopedia.com/terms/p/principal.asp",
    "https://www.investopedia.com/terms/r/risk.asp",

    # --- Investing Vehicles ---
    "https://www.investopedia.com/terms/s/stock.asp",
    "https://www.investopedia.com/terms/b/bond.asp",
    "https://www.investopedia.com/terms/m/mutualfund.asp",
    "https://www.investopedia.com/terms/e/etf.asp",
    "https://www.investopedia.com/terms/i/indexfund.asp",
    "https://www.investopedia.com/terms/r/reit.asp",
    "https://www.investopedia.com/terms/c/cryptocurrency.asp",
    "https://www.investopedia.com/terms/o/option.asp",
    "https://www.investopedia.com/terms/f/futures.asp",
    "https://www.investopedia.com/terms/c/commodity.asp",
    "https://www.investopedia.com/terms/t/treasury-yield.asp",
    "https://www.investopedia.com/terms/d/dividend.asp",
    "https://www.investopedia.com/terms/b/bluechipstock.asp",
    "https://www.investopedia.com/terms/p/penystock.asp",

    # --- Market Concepts ---
    "https://www.investopedia.com/terms/b/bullmarket.asp",
    "https://www.investopedia.com/terms/b/bearmarket.asp",
    "https://www.investopedia.com/terms/m/marketcapitalization.asp",
    "https://www.investopedia.com/terms/v/volatility.asp",
    "https://www.investopedia.com/terms/l/liquidity.asp",
    "https://www.investopedia.com/terms/i/ipo.asp",
    "https://www.investopedia.com/terms/s/sp500.asp",
    "https://www.investopedia.com/terms/d/dowjones.asp",
    "https://www.investopedia.com/terms/n/nasdaq.asp",
    "https://www.investopedia.com/terms/s/stock-split.asp",
    "https://www.investopedia.com/terms/s/shortselling.asp",
    "https://www.investopedia.com/terms/m/margin.asp",
    
    # --- Retirement & Accounts ---
    "https://www.investopedia.com/terms/1/401kplan.asp",
    "https://www.investopedia.com/terms/i/ira.asp",
    "https://www.investopedia.com/terms/r/rothira.asp",
    "https://www.investopedia.com/terms/t/traditionalira.asp",
    "https://www.investopedia.com/terms/r/requiredminimumdistribution.asp",
    "https://www.investopedia.com/terms/v/vesting.asp",
    "https://www.investopedia.com/terms/s/socialsecurity.asp",
    "https://www.investopedia.com/terms/p/pensionplan.asp",
    "https://www.investopedia.com/terms/a/annuity.asp",

    # --- Taxes ---
    "https://www.investopedia.com/terms/c/capitalgain.asp",
    "https://www.investopedia.com/terms/c/capital_gains_tax.asp",
    "https://www.investopedia.com/terms/t/taxbracket.asp",
    "https://www.investopedia.com/terms/t/taxdeduction.asp",
    "https://www.investopedia.com/terms/t/taxcredit.asp",
    "https://www.investopedia.com/terms/s/standarddeduction.asp",
    "https://www.investopedia.com/terms/w/w2form.asp",
    "https://www.investopedia.com/terms/1/1099.asp",
    
    # --- Economics ---
    "https://www.investopedia.com/terms/g/gdp.asp",
    "https://www.investopedia.com/terms/c/cpi.asp",
    "https://www.investopedia.com/terms/f/federalreservebank.asp",
    "https://www.investopedia.com/terms/m/monetarypolicy.asp",
    "https://www.investopedia.com/terms/f/fiscalpolicy.asp",
    "https://www.investopedia.com/terms/r/recession.asp",
    "https://www.investopedia.com/terms/d/depression.asp",
    "https://www.investopedia.com/terms/s/supply-and-demand.asp",
    "https://www.investopedia.com/terms/o/opportunitycost.asp",
]

def determine_category(url, article):
    """
    Deduce category from URL structure or keywords.
    """
    path = urlparse(url).path
    if "basics" in path or "terms" in path:
        # Refine based on keywords in title
        title_lower = article.title.lower()
        if any(x in title_lower for x in ["tax", "deduction", "credit", "irs"]):
            return "Taxes"
        if any(x in title_lower for x in ["retirement", "401k", "ira", "pension"]):
            return "Retirement"
        if any(x in title_lower for x in ["stock", "market", "bull", "bear", "ipo"]):
            return "Markets"
        if any(x in title_lower for x in ["economy", "gdp", "inflation", "recession"]):
            return "Economics"
        if any(x in title_lower for x in ["investing", "portfolio", "asset", "diversification"]):
            return "Investing"
            
    return "General_Finance"

def scrape_to_knowledge_base(url_list, output_dir="financial_docs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Deduplicate URLs while preserving order
    url_list = list(dict.fromkeys(url_list))

    print(f"Starting scrape of {len(url_list)} unique articles...")
    
    for i, url in enumerate(url_list):
        try:
            print(f"[{i+1}/{len(url_list)}] Downloading: {url}")
            article = Article(url)
            article.download()
            article.parse()
            # article.nlp() # Optional, requires 'nltk' download
            
            # Determine category
            category = determine_category(url, article)
            
            # Create category directory
            category_dir = os.path.join(output_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)

            # Clean filename
            safe_title = "".join([c for c in article.title if c.isalnum() or c==' ']).strip()
            filename = f"{safe_title.replace(' ', '_')}.txt"
            file_path = os.path.join(category_dir, filename)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Title: {article.title}\n")
                f.write(f"Source: {url}\n")
                f.write(f"Category: {category}\n")
                f.write("-" * 20 + "\n")
                f.write(article.text)
                
            print(f"  -> Saved to {category}/{filename}")
            
        except Exception as e:
            print(f"  -> Failed: {e}")

if __name__ == "__main__":
    scrape_to_knowledge_base(urls)
