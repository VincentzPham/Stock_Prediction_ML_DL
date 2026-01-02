import yfinance as yf
import pandas as pd
from datetime import date
import os

# Lấy đường dẫn thư mục chứa script này
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Danh sách ticker hardcode từ file của bạn
tickers = ['AAPL', 'AMZN', 'AVGO', 'BTC-USD', 'GOOG', 'META', 'MSFT', 'NVDA', 'SAP', 'TSLA', 'TSM']

# Hàm tìm ngày start (mở rộng dict với các ngày từ search)
def find_start_date(ticker):
    start_dates = {
        'AAPL': '1980-12-12',  # IPO date
        'AMZN': '1997-05-15',  # IPO date
        'AVGO': '2009-08-06',  # IPO date (xác nhận từ 2009-08)
        'BTC-USD': '2014-09-17',  # Start date on yfinance (không phải 2010 như gốc)
        'GOOG': '2004-08-19',  # IPO date
        'META': '2012-05-18',  # IPO date
        'MSFT': '1986-03-13',  # IPO date
        'NVDA': '1999-01-22',  # IPO date
        'SAP': '1988-11-04',   # IPO date (gốc ở Đức; trên NYSE khoảng 1998, nhưng yfinance có từ 1996)
        'TSLA': '2010-06-29',  # IPO date
        'TSM': '1994-09-05'    # IPO date (ADR on NYSE từ 1997, nhưng search cho 1994)
    }
    if ticker in start_dates:
        return start_dates[ticker]
    else:
        print(f"Không tìm thấy start date cho {ticker}, dùng default 1970-01-01")
        return '1970-01-01'

# Hàm tải và lưu dữ liệu cho tất cả ticker
def download_and_save_all():
    end_date = date.today().strftime('%Y-%m-%d')
    for ticker in tickers:
        start_date = find_start_date(ticker)
        try:
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)  # Thêm auto_adjust để tránh warning
            print(f"{ticker} - Số hàng: {len(df)}")
            print(df.head())
            print(df.tail())
            
            # Lưu file vào cùng thư mục với script (Data/)
            filename = os.path.join(SCRIPT_DIR, f"{ticker}.csv")
            df.to_csv(filename)
            print(f"Dữ liệu đã lưu vào {filename}")
        except Exception as e:
            print(f"Lỗi khi tải {ticker}: {e}")

def main():
    # Chạy thủ công nếu cần
    download_and_save_all()
    
if __name__ == "__main__":
    main()