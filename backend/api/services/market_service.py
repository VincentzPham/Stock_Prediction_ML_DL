"""
Market Service.

Handles trading calendar and market-related utilities.
"""

from typing import List
import pandas as pd


class MarketService:
    """
    Service class for market-related operations.
    
    Provides utilities for trading calendars, market hours, and related calculations.
    """
    
    # Tickers that trade 24/7 (cryptocurrencies)
    CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]
    
    @classmethod
    def get_trading_days(
        cls,
        start_date: pd.Timestamp,
        num_days: int,
        ticker: str = "AAPL"
    ) -> List[pd.Timestamp]:
        """
        Get a list of actual trading days using market calendars.
        
        Excludes weekends AND market holidays (MLK Day, Presidents Day, etc.)
        For cryptocurrencies, returns all days (24/7 trading).

        Args:
            start_date: Starting date (exclusive - will start from next trading day).
            num_days: Number of trading days to return.
            ticker: Ticker symbol to determine exchange (BTC-USD uses 24/7).

        Returns:
            List of trading day timestamps.
        """
        # Crypto trades 24/7, use all days
        if ticker in cls.CRYPTO_TICKERS:
            return cls._get_crypto_days(start_date, num_days)
        
        # For stocks, use NYSE calendar
        return cls._get_stock_trading_days(start_date, num_days)
    
    @classmethod
    def _get_crypto_days(
        cls,
        start_date: pd.Timestamp,
        num_days: int
    ) -> List[pd.Timestamp]:
        """
        Get consecutive days for crypto (trades 24/7).
        
        Args:
            start_date: Starting date (exclusive).
            num_days: Number of days to return.
            
        Returns:
            List of consecutive timestamps.
        """
        days = []
        current_date = start_date
        
        while len(days) < num_days:
            current_date += pd.Timedelta(days=1)
            days.append(current_date)
        
        return days
    
    @classmethod
    def _get_stock_trading_days(
        cls,
        start_date: pd.Timestamp,
        num_days: int
    ) -> List[pd.Timestamp]:
        """
        Get trading days for stocks using NYSE calendar.
        
        Args:
            start_date: Starting date (exclusive).
            num_days: Number of trading days to return.
            
        Returns:
            List of trading day timestamps.
        """
        try:
            import pandas_market_calendars as mcal
            
            calendar = mcal.get_calendar("NYSE")
            
            # Get schedule for next 90 days (buffer for holidays)
            end_date = start_date + pd.Timedelta(days=num_days * 2 + 30)
            schedule = calendar.schedule(
                start_date=start_date + pd.Timedelta(days=1),
                end_date=end_date
            )
            
            if len(schedule) >= num_days:
                trading_days = [pd.Timestamp(d) for d in schedule.index[:num_days]]
                return trading_days
                
        except Exception:
            pass
        
        # Fallback to simple business days (weekdays only)
        return cls._get_business_days_fallback(start_date, num_days)
    
    @classmethod
    def _get_business_days_fallback(
        cls,
        start_date: pd.Timestamp,
        num_days: int
    ) -> List[pd.Timestamp]:
        """
        Fallback method to get business days (weekdays only).
        
        Args:
            start_date: Starting date (exclusive).
            num_days: Number of business days to return.
            
        Returns:
            List of weekday timestamps.
        """
        business_days = []
        current_date = start_date
        
        while len(business_days) < num_days:
            current_date += pd.Timedelta(days=1)
            # Monday=0, Friday=4
            if current_date.weekday() < 5:
                business_days.append(current_date)
        
        return business_days
