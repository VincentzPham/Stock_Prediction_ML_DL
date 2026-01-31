"""
Backend Caching Module.

Provides in-memory caching for expensive operations.
Uses TTLCache from cachetools for time-based expiration.
"""

from cachetools import TTLCache
from typing import Any, Callable
from functools import wraps

# Cache TTL in seconds (5 minutes)
CACHE_TTL = 300

# Maximum cache size
CACHE_MAXSIZE = 200

# Global cache instances
_leaderboard_cache: TTLCache = TTLCache(maxsize=1, ttl=CACHE_TTL)
_market_overview_cache: TTLCache = TTLCache(maxsize=1, ttl=CACHE_TTL)
_ticker_comparison_cache: TTLCache = TTLCache(maxsize=20, ttl=CACHE_TTL)
_metrics_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)


def cached_leaderboard(func: Callable) -> Callable:
    """
    Decorator for caching leaderboard data.
    
    Args:
        func: Function to wrap.
        
    Returns:
        Wrapped function with caching.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        cache_key = "leaderboard"
        if cache_key in _leaderboard_cache:
            return _leaderboard_cache[cache_key]
        
        result = func(*args, **kwargs)
        _leaderboard_cache[cache_key] = result
        return result
    
    return wrapper


def cached_market_overview(func: Callable) -> Callable:
    """
    Decorator for caching market overview data.
    
    Args:
        func: Function to wrap.
        
    Returns:
        Wrapped function with caching.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        cache_key = "market_overview"
        if cache_key in _market_overview_cache:
            return _market_overview_cache[cache_key]
        
        result = func(*args, **kwargs)
        _market_overview_cache[cache_key] = result
        return result
    
    return wrapper


def cached_ticker_comparison(func: Callable) -> Callable:
    """
    Decorator for caching ticker comparison data.
    
    Args:
        func: Function to wrap.
        
    Returns:
        Wrapped function with caching.
    """
    @wraps(func)
    def wrapper(cls_or_self, ticker: str, *args, **kwargs) -> Any:
        cache_key = ticker
        if cache_key in _ticker_comparison_cache:
            return _ticker_comparison_cache[cache_key]
        
        result = func(cls_or_self, ticker, *args, **kwargs)
        _ticker_comparison_cache[cache_key] = result
        return result
    
    return wrapper


def clear_all_caches() -> None:
    """Clear all backend caches."""
    _leaderboard_cache.clear()
    _market_overview_cache.clear()
    _ticker_comparison_cache.clear()
    _metrics_cache.clear()


def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache stats.
    """
    return {
        "leaderboard": {
            "size": len(_leaderboard_cache),
            "maxsize": _leaderboard_cache.maxsize,
        },
        "market_overview": {
            "size": len(_market_overview_cache),
            "maxsize": _market_overview_cache.maxsize,
        },
        "ticker_comparison": {
            "size": len(_ticker_comparison_cache),
            "maxsize": _ticker_comparison_cache.maxsize,
        },
        "metrics": {
            "size": len(_metrics_cache),
            "maxsize": _metrics_cache.maxsize,
        },
    }
