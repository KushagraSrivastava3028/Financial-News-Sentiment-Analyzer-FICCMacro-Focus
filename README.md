# Financial News Sentiment Analyzer (FICC Macro Focus)

## Overview
The **Financial News Sentiment Analyzer** is a Python-based project that analyzes financial news sentiment and examines its relationship with market volatility.  
It scrapes market news headlines from **FinViz**, applies **VADER sentiment analysis**, and correlates the aggregated sentiment with the **VIX (Volatility Index)** to study fear and risk behavior in financial markets.

This project is inspired by **FICC Macro analysis**, where news-driven sentiment plays a crucial role in volatility and risk pricing.

---

## Objectives
- Scrape real-time financial news headlines
- Perform sentiment analysis on market news
- Classify sentiment as **Positive, Negative, or Neutral**
- Analyze correlation between sentiment and **VIX volatility**
- Visualize sentiment vs market fear trends

---

## Tech Stack
- **Python**
- **Pandas & NumPy** – Data manipulation
- **BeautifulSoup** – Web scraping (FinViz)
- **NLTK (VADER)** – Sentiment analysis
- **yFinance** – Market data (VIX)
- **Matplotlib & Seaborn** – Visualization

---

## Data Sources
- **FinViz** – Financial news headlines
- **Yahoo Finance** – VIX historical data

---

## Sentiment Logic
- **Compound Score Range:** `-1` (Very Negative) to `+1` (Very Positive)
- **Classification:**
  - `Positive` → Bullish / Lower volatility
  - `Negative` → Bearish / Higher volatility
  - `Neutral` → No strong sentiment

---

## Visualization
- **Bar Chart:** Daily average news sentiment
- **Line Chart:** Corresponding VIX closing values
- Dual-axis plot to observe sentiment–volatility dynamics

---

## How to Run
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Financial-News-Sentiment-Analyzer.git
cd Financial-News-Sentiment-Analyzer
```

### 2️⃣ Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn beautifulsoup4 nltk yfinance lxml

```

### 3️⃣ Run the Script
```bash
python sentiment_analyzer.py
```

## Note: 
The script automatically downloads the required NLTK VADER lexicon on the first run.
