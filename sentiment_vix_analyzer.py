import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import yfinance as yf
from datetime import datetime, timedelta

# --- Configuration ---
# Ticker to analyze news for (SPY represents the general market)
TICKER = 'SPY' 
# VIX Ticker for volatility comparison
VIX_TICKER = '^VIX' 

def download_nltk_resources():
    """Ensures necessary NLTK data is downloaded."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading VADER lexicon...")
        nltk.download('vader_lexicon')

def scrape_finviz_news(ticker):
    """
    Scrapes the news table from FinViz for a specific ticker.
    Returns a DataFrame with Date, Time, and Headline.
    """
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    req = Request(url=url, headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}) 
    
    print(f"Scraping news for {ticker} from FinViz...")
    try:
        response = urlopen(req)
        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id='news-table')
        
        parsed_news = []
        
        # Iterate through all rows in the news table
        for x in news_table.findAll('tr'):
            text = x.a.get_text() 
            date_scrape = x.td.text.split()
            
            # FinViz date format: "Dec-22-25 09:30AM" or just "09:30AM" if same day
            if len(date_scrape) == 1:
                time = date_scrape[0]
                # Date remains the same as previous iteration, so we don't update 'date' variable
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            
            parsed_news.append([date, time, text])
            
        columns = ['date', 'time', 'headline']
        df = pd.DataFrame(parsed_news, columns=columns)
        
        # Clean up dates
        # FinViz uses "Today" or "Dec-22-25"
        today_str = datetime.now().strftime("%b-%d-%y")
        df['date'] = df['date'].replace('Today', today_str)
        
        # Convert to datetime objects
        df['parsed_date'] = pd.to_datetime(df['date'], format='%b-%d-%y', errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"Error scraping data: {e}")
        return pd.DataFrame()

def analyze_sentiment(df):
    """
    Applies VADER Sentiment Analysis to the headlines.
    """
    print("Running Sentiment Analysis (VADER)...")
    vader = SentimentIntensityAnalyzer()
    
    # Calculate scores
    # Compound score: -1 (Most Negative) to +1 (Most Positive)
    df['compound'] = df['headline'].apply(lambda title: vader.polarity_scores(title)['compound'])
    
    # Classify as Hawkish (Negative/Fear) or Dovish (Positive/Calm)
    # Note: In general market terms, Positive News = Bullish = Lower Volatility
    # Negative News = Bearish = Higher Volatility
    df['sentiment_type'] = df['compound'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))
    
    return df

def get_market_data(start_date):
    """
    Fetches VIX (Volatility Index) data from yfinance.
    """
    print(f"Fetching VIX data from {start_date}...")
    vix = yf.download(VIX_TICKER, start=start_date, progress=False)['Close']
    vix = vix.reset_index()
    vix.columns = ['Date', 'VIX_Close']
    
    # Ensure Date format matches our news data
    # Convert to datetime, remove timezone if present, and then extract date
    vix['Date'] = pd.to_datetime(vix['Date']).dt.tz_localize(None).dt.date
    return vix

def plot_correlation(news_df, vix_df):
    """
    Aggregates sentiment by day and plots it against VIX.
    """
    # 1. Aggregate Sentiment by Date (Mean Compound Score)
    daily_sentiment = news_df.groupby(news_df['parsed_date'].dt.date)['compound'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Avg_Sentiment']
    
    # 2. Merge with VIX Data
    merged_df = pd.merge(daily_sentiment, vix_df, on='Date', how='inner')
    
    if merged_df.empty:
        print("Not enough overlapping data to plot.")
        return

    print(merged_df.head())

    # Ensure Date is string for plotting to avoid matplotlib timezone/date type errors
    merged_df['Date'] = merged_df['Date'].astype(str)

    # 3. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Sentiment (Bar Chart)
    sns.barplot(x='Date', y='Avg_Sentiment', data=merged_df, ax=ax1, alpha=0.5, color='blue', label='News Sentiment')
    ax1.set_ylabel('Avg Sentiment Score (-1 to 1)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(merged_df['Date'], rotation=45)

    # Plot VIX (Line Chart) on Secondary Axis
    ax2 = ax1.twinx()
    sns.lineplot(x='Date', y='VIX_Close', data=merged_df, ax=ax2, color='red', marker='o', label='VIX (Volatility)')
    ax2.set_ylabel('VIX Close Price', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(f'News Sentiment vs Market Volatility (VIX) - {TICKER}')
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    download_nltk_resources()
    
    # 1. Scrape News
    news_df = scrape_finviz_news(TICKER)
    
    if not news_df.empty:
        # 2. Analyze Sentiment
        news_df = analyze_sentiment(news_df)
        
        print("\nLatest Headlines & Sentiment:")
        print(news_df[['parsed_date', 'headline', 'compound']].head())
        
        # 3. Get Market Data (Look back as far as the oldest news article)
        start_date = news_df['parsed_date'].min().strftime('%Y-%m-%d')
        vix_df = get_market_data(start_date)
        
        # 4. Correlate and Visualize
        plot_correlation(news_df, vix_df)
    else:
        print("No news found.")
