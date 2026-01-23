"""
Data Downloader Module
Tải dữ liệu từ yfinance.
"""

import yfinance as yf
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Optional, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, TICKERS, IPO_DATES


class DataDownloader:
    """
    Class tải dữ liệu stock từ yfinance.
    """
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or DATA_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def download(
        self, 
        ticker: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Tải dữ liệu cho một ticker.
        
        Args:
            ticker: Mã chứng khoán
            start_date: Ngày bắt đầu (YYYY-MM-DD). Nếu None, dùng IPO date.
            end_date: Ngày kết thúc. Nếu None, dùng today.
            save: Có lưu file CSV không.
            
        Returns:
            DataFrame với dữ liệu stock.
        """
        if start_date is None:
            start_date = IPO_DATES.get(ticker, '1970-01-01')
            
        if end_date is None:
            end_date = date.today().strftime('%Y-%m-%d')
        
        print(f"Downloading {ticker} from {start_date} to {end_date}...")
        
        df = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            auto_adjust=True,
            progress=False
        )
        
        print(f"  -> {len(df)} rows downloaded")
        
        if save:
            filepath = self.save_dir / f"{ticker}.csv"
            df.to_csv(filepath)
            print(f"  -> Saved to {filepath}")
        
        return df
    
    def download_all(
        self, 
        tickers: Optional[List[str]] = None,
        save: bool = True
    ) -> dict:
        """
        Tải dữ liệu cho tất cả tickers.
        
        Args:
            tickers: Danh sách mã. Nếu None, dùng default TICKERS.
            save: Có lưu file CSV không.
            
        Returns:
            Dict {ticker: DataFrame}
        """
        if tickers is None:
            tickers = TICKERS
            
        results = {}
        
        for ticker in tickers:
            try:
                df = self.download(ticker, save=save)
                results[ticker] = df
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                results[ticker] = None
                
        return results
    
    def update_data(self, ticker: str) -> pd.DataFrame:
        """
        Cập nhật dữ liệu mới nhất cho ticker.
        Đọc file hiện có và chỉ tải thêm dữ liệu mới.
        """
        filepath = self.save_dir / f"{ticker}.csv"
        
        if filepath.exists():
            # Đọc file hiện có
            existing_df = pd.read_csv(filepath, skiprows=2)
            existing_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            
            # Lấy ngày cuối cùng
            last_date = existing_df['Date'].max().strftime('%Y-%m-%d')
            
            # Tải dữ liệu mới
            new_df = self.download(ticker, start_date=last_date, save=False)
            
            if len(new_df) > 1:  # Có dữ liệu mới
                # Merge
                new_df = new_df.reset_index()
                new_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                
                combined = pd.concat([existing_df, new_df]).drop_duplicates(subset='Date')
                combined.sort_values('Date', inplace=True)
                
                # Lưu lại
                combined.set_index('Date').to_csv(filepath)
                print(f"Updated {ticker}: {len(combined)} total rows")
                
                return combined
        
        # Nếu chưa có file, tải mới
        return self.download(ticker, save=True)


# CLI interface
if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.download_all()
