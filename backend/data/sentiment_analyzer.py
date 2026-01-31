"""
Sentiment Pipeline V2
Robust sentiment analysis with multiple sources, proper time alignment, and entity linking.

Key Improvements:
1. Multiple data sources (Google News RSS, FinViz)
2. Ticker entity linking with aliases
3. UTC/ET timezone handling with market cutoff (4:00 PM ET)
4. Exponential decay weighted aggregation
5. Rolling window for smoother signals
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

# Timezone support
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from backend.config import DATA_DIR


# ============================================
# TICKER ENTITY LINKING
# ============================================

TICKER_ALIASES: Dict[str, List[str]] = {
    "AAPL": ["Apple", "AAPL", "$AAPL", "iPhone", "iPad", "MacBook", "Tim Cook", "Apple Inc"],
    "TSLA": ["Tesla", "TSLA", "$TSLA", "Elon Musk", "Model 3", "Model Y", "Cybertruck", "Tesla Inc"],
    "MSFT": ["Microsoft", "MSFT", "$MSFT", "Azure", "Windows", "Satya Nadella", "Xbox", "Microsoft Corp"],
    "GOOGL": ["Google", "GOOGL", "$GOOGL", "Alphabet", "YouTube", "Sundar Pichai", "Android"],
    "AMZN": ["Amazon", "AMZN", "$AMZN", "AWS", "Jeff Bezos", "Andy Jassy", "Prime"],
    "META": ["Meta", "META", "$META", "Facebook", "Instagram", "WhatsApp", "Mark Zuckerberg", "Metaverse"],
    "NVDA": ["NVIDIA", "NVDA", "$NVDA", "Jensen Huang", "GeForce", "RTX", "GPU"],
    "AMD": ["AMD", "$AMD", "Advanced Micro Devices", "Lisa Su", "Ryzen", "EPYC"],
    "INTC": ["Intel", "INTC", "$INTC", "Pat Gelsinger", "Core i"],
    "TSM": ["TSMC", "TSM", "$TSM", "Taiwan Semi", "Taiwan Semiconductor"],
    "VZ": ["Verizon", "VZ", "$VZ", "Verizon Communications"],
}


def matches_ticker(text: str, ticker: str) -> bool:
    """Check if text mentions the ticker or its aliases."""
    aliases = TICKER_ALIASES.get(ticker, [ticker])
    text_lower = text.lower()
    
    for alias in aliases:
        if alias.lower() in text_lower:
            return True
    
    # Also check for exact ticker match with $ prefix
    if f"${ticker}" in text.upper():
        return True
    
    return False


# ============================================
# SENTIMENT SOURCE BASE CLASS
# ============================================

class SentimentSource(ABC):
    """Abstract base class for sentiment data sources."""
    
    @abstractmethod
    def fetch_news(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch news for a ticker.
        
        Returns:
            DataFrame with columns: ['timestamp', 'headline', 'source']
            timestamp should be timezone-aware (UTC)
        """
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the source for logging."""
        pass


# ============================================
# GOOGLE NEWS RSS SOURCE
# ============================================

class GoogleNewsSource(SentimentSource):
    """Fetch news from Google News RSS (no API key needed)."""
    
    BASE_URL = "https://news.google.com/rss/search"
    
    @property
    def source_name(self) -> str:
        return "GoogleNews"
    
    def fetch_news(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch news from Google News RSS."""
        # Use ticker and company name for better coverage
        aliases = TICKER_ALIASES.get(ticker, [ticker])
        query = f"{ticker} stock OR {aliases[0] if aliases else ticker}"
        
        url = f"{self.BASE_URL}?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse RSS XML
            root = ET.fromstring(response.content)
            
            news_items = []
            cutoff_date = datetime.now(ZoneInfo("UTC")) - timedelta(days=days_back)
            
            for item in root.findall(".//item"):
                title = item.find("title")
                pub_date = item.find("pubDate")
                
                if title is None or pub_date is None:
                    continue
                
                headline = title.text
                
                # Parse RSS date format: "Wed, 29 Jan 2025 14:30:00 GMT"
                try:
                    timestamp = datetime.strptime(
                        pub_date.text, "%a, %d %b %Y %H:%M:%S %Z"
                    ).replace(tzinfo=ZoneInfo("UTC"))
                except ValueError:
                    continue
                
                # Filter by date
                if timestamp < cutoff_date:
                    continue
                
                # Filter by ticker relevance
                if not matches_ticker(headline, ticker):
                    continue
                
                news_items.append({
                    "timestamp": timestamp,
                    "headline": headline,
                    "source": self.source_name
                })
            
            return pd.DataFrame(news_items)
            
        except Exception as e:
            print(f"[{self.source_name}] Error fetching news for {ticker}: {e}")
            return pd.DataFrame()


# ============================================
# FINVIZ SOURCE (IMPROVED)
# ============================================

class FinVizSource(SentimentSource):
    """Improved FinViz news scraper with timezone handling."""
    
    @property
    def source_name(self) -> str:
        return "FinViz"
    
    def fetch_news(self, ticker: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch news from FinViz with proper timezone handling."""
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            
            news_table = soup.find(id="news-table")
            if not news_table:
                return pd.DataFrame()
            
            news_items = []
            current_date = None
            cutoff_date = datetime.now(ZoneInfo("US/Eastern")).date() - timedelta(days=days_back)
            
            for row in news_table.findAll("tr"):
                text_cell = row.find("td", align="left")
                if not text_cell:
                    continue
                
                link_tag = text_cell.find("a")
                if not link_tag:
                    continue
                
                headline = link_tag.get_text()
                
                date_cell = row.find("td", width="130")
                if not date_cell:
                    continue
                
                date_text = date_cell.get_text().strip()
                
                # Parse FinViz timestamp
                timestamp = self._parse_finviz_timestamp(date_text, current_date)
                if timestamp is None:
                    continue
                
                # Update current_date for rows with only time
                current_date = timestamp.date()
                
                # Filter by date
                if timestamp.date() < cutoff_date:
                    continue
                
                news_items.append({
                    "timestamp": timestamp,
                    "headline": headline,
                    "source": self.source_name
                })
            
            return pd.DataFrame(news_items)
            
        except Exception as e:
            print(f"[{self.source_name}] Error fetching news for {ticker}: {e}")
            return pd.DataFrame()
    
    def _parse_finviz_timestamp(
        self, date_text: str, current_date: Optional[datetime.date]
    ) -> Optional[datetime]:
        """Parse FinViz timestamp with timezone awareness."""
        date_text = date_text.strip()
        et_tz = ZoneInfo("US/Eastern")
        
        # Format 1: "Jan-26-25 04:30PM" (full date + time)
        if " " in date_text and "-" in date_text:
            parts = date_text.split(" ")
            if len(parts) >= 2:
                try:
                    date_part = parts[0]
                    time_part = parts[1]
                    dt = datetime.strptime(f"{date_part} {time_part}", "%b-%d-%y %I:%M%p")
                    return dt.replace(tzinfo=et_tz)
                except ValueError:
                    pass
        
        # Format 2: "04:30PM" (time only, use current_date)
        elif ":" in date_text and current_date:
            try:
                time_dt = datetime.strptime(date_text.upper(), "%I:%M%p")
                combined = datetime.combine(current_date, time_dt.time())
                return combined.replace(tzinfo=et_tz)
            except ValueError:
                pass
        
        return None


# ============================================
# SENTIMENT ANALYZER V2
# ============================================

class SentimentAnalyzerV2:
    """
    Enhanced sentiment analyzer with:
    - Multiple data sources
    - Time alignment (market cutoff at 4:00 PM ET)
    - Exponential decay aggregation
    - Rolling window smoothing
    """
    
    MARKET_CLOSE_HOUR = 16  # 4:00 PM ET
    
    def __init__(self, decay_halflife: int = 3, rolling_window: int = 3):
        """
        Args:
            decay_halflife: Days for sentiment decay half-life.
            rolling_window: Days for rolling average smoothing.
        """
        self.decay_halflife = decay_halflife
        self.rolling_window = rolling_window
        
        self.save_dir = DATA_DIR / "sentiment"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize VADER
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            print("Downloading VADER lexicon...")
            nltk.download("vader_lexicon", quiet=True)
        
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize sources
        self.sources: List[SentimentSource] = [
            GoogleNewsSource(),
            FinVizSource(),
        ]
    
    def fetch_all_news(self, ticker: str, days_back: int = 14) -> pd.DataFrame:
        """Fetch news from all sources and combine."""
        all_news = []
        
        for source in self.sources:
            print(f"  Fetching from {source.source_name}...")
            df = source.fetch_news(ticker, days_back)
            if not df.empty:
                all_news.append(df)
                print(f"    -> {len(df)} headlines")
        
        if not all_news:
            return pd.DataFrame()
        
        combined = pd.concat(all_news, ignore_index=True)
        
        # Remove duplicates (same headline within 1 hour)
        combined = combined.sort_values("timestamp")
        combined = combined.drop_duplicates(subset=["headline"], keep="first")
        
        return combined
    
    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VADER compound sentiment for each headline."""
        if df.empty:
            return df
        
        df = df.copy()
        df["sentiment"] = df["headline"].apply(
            lambda x: self.vader.polarity_scores(x)["compound"]
        )
        return df
    
    def align_to_trading_day(self, timestamp: datetime) -> datetime.date:
        """
        Align news timestamp to the correct trading day.
        
        Rule: News published AFTER 4:00 PM ET affects NEXT trading day.
        News published before 4:00 PM ET affects CURRENT trading day.
        """
        # Convert to ET
        et_tz = ZoneInfo("US/Eastern")
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
        
        et_time = timestamp.astimezone(et_tz)
        
        # Check if after market close
        if et_time.hour >= self.MARKET_CLOSE_HOUR:
            # Affects next trading day
            return (et_time + timedelta(days=1)).date()
        else:
            # Affects current day
            return et_time.date()
    
    def aggregate_daily_sentiment(
        self, df: pd.DataFrame, apply_decay: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate sentiment by trading day with optional exponential decay.
        
        Args:
            df: DataFrame with ['timestamp', 'headline', 'sentiment']
            apply_decay: Whether to apply time decay weighting.
            
        Returns:
            DataFrame with daily sentiment scores.
        """
        if df.empty:
            return pd.DataFrame(columns=["Date", "Sentiment_Score", "News_Count"])
        
        df = df.copy()
        
        # Align to trading day
        df["trading_day"] = df["timestamp"].apply(self.align_to_trading_day)
        
        # Calculate decay weights (more recent = higher weight)
        if apply_decay:
            max_date = df["trading_day"].max()
            df["days_ago"] = df["trading_day"].apply(lambda d: (max_date - d).days)
            df["weight"] = np.exp(-np.log(2) * df["days_ago"] / self.decay_halflife)
        else:
            df["weight"] = 1.0
        
        # Weighted mean per day
        grouped = df.groupby("trading_day").apply(
            lambda x: pd.Series({
                "Sentiment_Score": np.average(x["sentiment"], weights=x["weight"]),
                "News_Count": len(x),
                "Sources": ",".join(x["source"].unique())
            })
        ).reset_index()
        
        grouped.columns = ["Date", "Sentiment_Score", "News_Count", "Sources"]
        grouped["Date"] = pd.to_datetime(grouped["Date"])
        grouped = grouped.set_index("Date").sort_index()
        
        return grouped
    
    def apply_rolling_smooth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling average for smoother sentiment signal."""
        if df.empty or len(df) < 2:
            return df
        
        df = df.copy()
        df["Sentiment_Score_Raw"] = df["Sentiment_Score"]
        df["Sentiment_Score"] = df["Sentiment_Score"].rolling(
            window=self.rolling_window, min_periods=1
        ).mean()
        
        return df
    
    def get_daily_sentiment(
        self, 
        ticker: str, 
        days_back: int = 14,
        apply_decay: bool = True,
        apply_rolling: bool = True,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Main pipeline: Fetch -> Analyze -> Align -> Aggregate.
        
        Args:
            ticker: Stock ticker.
            days_back: Days of history to fetch.
            apply_decay: Apply exponential decay weighting.
            apply_rolling: Apply rolling average smoothing.
            save: Save results to CSV.
            
        Returns:
            DataFrame with daily sentiment aligned to trading days.
        """
        print(f"Fetching sentiment for {ticker}...")
        
        # Fetch from all sources
        news_df = self.fetch_all_news(ticker, days_back)
        
        if news_df.empty:
            print(f"  No news found for {ticker}")
            return pd.DataFrame()
        
        # Analyze sentiment
        news_df = self.analyze_sentiment(news_df)
        
        # Aggregate by trading day
        daily_df = self.aggregate_daily_sentiment(news_df, apply_decay)
        
        # Apply rolling smoothing
        if apply_rolling and not daily_df.empty:
            daily_df = self.apply_rolling_smooth(daily_df)
        
        # Save
        if save and not daily_df.empty:
            # Save daily aggregated
            save_path = self.save_dir / f"{ticker}_daily_v2.csv"
            daily_df.to_csv(save_path)
            print(f"  Saved to {save_path}")
            
            # Save raw news for debugging
            raw_path = self.save_dir / f"{ticker}_raw_v2.csv"
            news_df.to_csv(raw_path, index=False)
        
        print(f"  Total: {len(news_df)} headlines -> {len(daily_df)} trading days")
        
        return daily_df
    
    def merge_with_price_data(
        self, 
        price_df: pd.DataFrame, 
        sentiment_df: pd.DataFrame,
        fill_method: str = "neutral"
    ) -> pd.DataFrame:
        """
        Merge sentiment with price data, handling missing days.
        
        Args:
            price_df: Price DataFrame with DatetimeIndex.
            sentiment_df: Sentiment DataFrame with DatetimeIndex.
            fill_method: How to fill missing sentiment ('neutral'=0, 'ffill', 'drop').
            
        Returns:
            Price DataFrame with Sentiment_Score column.
        """
        price_df = price_df.copy()
        
        # Ensure indices are compatible
        if not isinstance(sentiment_df.index, pd.DatetimeIndex):
            sentiment_df.index = pd.to_datetime(sentiment_df.index)
        
        # Join
        price_df = price_df.join(sentiment_df[["Sentiment_Score"]], how="left")
        
        # Handle missing values
        if fill_method == "neutral":
            price_df["Sentiment_Score"] = price_df["Sentiment_Score"].fillna(0)
        elif fill_method == "ffill":
            price_df["Sentiment_Score"] = price_df["Sentiment_Score"].ffill()
        elif fill_method == "drop":
            price_df = price_df.dropna(subset=["Sentiment_Score"])
        
        return price_df


# ============================================
# BACKWARD COMPATIBILITY
# ============================================

class SentimentAnalyzer(SentimentAnalyzerV2):
    """Alias for backward compatibility."""
    pass


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    analyzer = SentimentAnalyzerV2(decay_halflife=3, rolling_window=3)
    
    test_ticker = "AAPL"
    print(f"\n{'='*60}")
    print(f"Testing Sentiment Pipeline V2 for {test_ticker}")
    print(f"{'='*60}\n")
    
    df = analyzer.get_daily_sentiment(test_ticker, days_back=14)
    
    if not df.empty:
        print(f"\nDaily Sentiment for {test_ticker}:")
        print(df.tail(10))
    else:
        print("No sentiment data retrieved.")
