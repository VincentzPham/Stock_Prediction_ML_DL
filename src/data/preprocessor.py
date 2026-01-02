"""
Data Preprocessor Module
Xử lý và chuẩn bị dữ liệu cho các mô hình.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, TRAIN_TEST_SPLIT, TIME_STEP, 
    TARGET_COLUMN, FEATURE_COLUMNS
)


class DataPreprocessor:
    """
    Class xử lý dữ liệu cho stock prediction.
    Hỗ trợ cả time series models và ML models.
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.df: Optional[pd.DataFrame] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self._is_fitted = False
        
    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Đọc dữ liệu từ CSV file.
        
        Args:
            file_path: Đường dẫn file CSV. Nếu None, sử dụng default path.
            
        Returns:
            DataFrame đã được xử lý.
        """
        if file_path is None:
            file_path = DATA_DIR / f"{self.ticker}.csv"
            
        # Đọc CSV, skip 2 hàng đầu (header của yfinance)
        df = pd.read_csv(file_path, skiprows=2)
        
        # Đặt tên cột
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        
        # Chuyển Date thành datetime và set làm index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Sắp xếp theo thời gian
        df.sort_index(inplace=True)
        
        # Loại bỏ NaN
        df.dropna(inplace=True)
        
        self.df = df
        return df
    
    def add_features(self) -> pd.DataFrame:
        """
        Thêm các features engineering.
        
        Returns:
            DataFrame với các features mới.
        """
        if self.df is None:
            raise ValueError("Chưa load data. Gọi load_data() trước.")
            
        df = self.df.copy()
        
        # Basic features
        df['Price_Diff'] = df['High'] - df['Low']
        df['Avg_Price'] = (df['High'] + df['Low']) / 2
        df['Price_Change'] = df['Close'] - df['Open']
        
        # Volume feature
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        
        # Moving Averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=21).std()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Loại bỏ NaN sau khi thêm features
        df.dropna(inplace=True)
        
        self.df = df
        return df
    
    def get_train_test_split(
        self, 
        split_ratio: float = TRAIN_TEST_SPLIT
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chia dữ liệu thành train và test set (giữ thứ tự thời gian).
        
        Args:
            split_ratio: Tỷ lệ train set.
            
        Returns:
            Tuple (train_df, test_df)
        """
        if self.df is None:
            raise ValueError("Chưa load data. Gọi load_data() trước.")
            
        split_idx = int(len(self.df) * split_ratio)
        train_df = self.df.iloc[:split_idx].copy()
        test_df = self.df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def prepare_lstm_data(
        self, 
        time_step: int = TIME_STEP,
        horizon: int = 1,
        target_col: str = TARGET_COLUMN
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Chuẩn bị dữ liệu cho LSTM/RNN models.
        
        Args:
            time_step: Số bước thời gian lookback.
            horizon: Số ngày dự đoán tiếp theo (1 = next day).
            target_col: Tên cột target.
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test, scaler)
        """
        if self.df is None:
            raise ValueError("Chưa load data. Gọi load_data() trước.")
        
        # Lấy target column
        data = self.df[target_col].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.target_scaler.fit_transform(data)
        self._is_fitted = True
        
        # Split train/test
        train_size = int(len(scaled_data) * TRAIN_TEST_SPLIT)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # Tạo sequences với horizon
        X_train, y_train = self._create_sequences(train_data, time_step, horizon)
        X_test, y_test = self._create_sequences(test_data, time_step, horizon)
        
        # Reshape cho LSTM (samples, time_steps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return X_train, X_test, y_train, y_test, self.target_scaler
    
    def prepare_ml_data(
        self, 
        feature_cols: Optional[List[str]] = None,
        horizon: int = 1,
        target_col: str = TARGET_COLUMN
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Chuẩn bị dữ liệu cho ML models (Random Forest, Decision Tree, etc).
        
        Args:
            feature_cols: Danh sách features. Nếu None, sử dụng default.
            horizon: Số ngày dự đoán tiếp theo (1 = next day).
            target_col: Tên cột target.
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        if self.df is None:
            raise ValueError("Chưa load data. Gọi load_data() trước.")
        
        # Default features cho ML
        if feature_cols is None:
            feature_cols = ['Open', 'High', 'Low', 'Volume', 'Price_Diff', 'Avg_Price', 'Volume_Ratio']
            # Chỉ lấy các cột có trong df
            feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        # Tạo target shift theo horizon
        # y tại index t là giá Close tại t + horizon
        df_ml = self.df.copy()
        df_ml['Target'] = df_ml[target_col].shift(-horizon)
        
        # Drop NaN do shift (các dòng cuối không có target tương lai)
        df_ml.dropna(inplace=True)
        
        X = df_ml[feature_cols].values
        y = df_ml['Target'].values
        
        # Split
        split_idx = int(len(X) * TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test

    def _create_sequences(
        self, 
        data: np.ndarray, 
        time_step: int,
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function tạo sequences cho LSTM.
        
        Args:
            data: Scaled data array.
            time_step: Window size.
            horizon: Forecast horizon.
            
        Returns:
            Tuple (X, y)
        """
        X, y = [], []
        # Loop từ time_step đến len(data) - horizon
        # Tại index i:
        # X = data[i-time_step : i] (window quá khứ)
        # y = data[i + horizon - 1] (giá trị tương lai tại horizon)
        
        for i in range(time_step, len(data) - horizon + 1):
            X.append(data[i-time_step:i, 0])
            y.append(data[i + horizon - 1, 0])
            
        return np.array(X), np.array(y)
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def prepare_timeseries_data(
        self, 
        target_col: str = TARGET_COLUMN
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Chuẩn bị dữ liệu cho Time Series models (ARIMA, SARIMA, etc).
        
        Args:
            target_col: Tên cột target.
            
        Returns:
            Tuple (train_series, test_series)
        """
        if self.df is None:
            raise ValueError("Chưa load data. Gọi load_data() trước.")
        
        series = self.df[target_col]
        
        split_idx = int(len(series) * TRAIN_TEST_SPLIT)
        train_series = series.iloc[:split_idx]
        test_series = series.iloc[split_idx:]
        
        return train_series, test_series
    
    def prepare_prophet_data(
        self, 
        target_col: str = TARGET_COLUMN
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chuẩn bị dữ liệu cho Prophet model.
        Prophet yêu cầu columns: 'ds' (date), 'y' (value)
        
        Returns:
            Tuple (train_df, test_df) với columns ['ds', 'y']
        """
        if self.df is None:
            raise ValueError("Chưa load data. Gọi load_data() trước.")
        
        # Prophet format
        prophet_df = pd.DataFrame({
            'ds': self.df.index,
            'y': self.df[target_col].values
        })
        
        split_idx = int(len(prophet_df) * TRAIN_TEST_SPLIT)
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]
        
        return train_df, test_df
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Chuyển dữ liệu đã scale về giá trị gốc.
        """
        if not self._is_fitted:
            raise ValueError("Scaler chưa được fit. Gọi prepare_lstm_data() trước.")
        return self.target_scaler.inverse_transform(data.reshape(-1, 1))
    
    def get_dates(self, dataset: str = 'test') -> pd.DatetimeIndex:
        """
        Lấy dates cho train hoặc test set.
        """
        if self.df is None:
            raise ValueError("Chưa load data.")
            
        split_idx = int(len(self.df) * TRAIN_TEST_SPLIT)
        
        if dataset == 'train':
            return self.df.index[:split_idx]
        else:
            return self.df.index[split_idx:]
    
    @staticmethod
    def _create_sequences(data: np.ndarray, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo sequences cho time series data."""
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i - time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def get_info(self) -> dict:
        """Trả về thông tin về dataset."""
        if self.df is None:
            return {"status": "No data loaded"}
        
        return {
            "ticker": self.ticker,
            "shape": self.df.shape,
            "date_range": f"{self.df.index.min()} to {self.df.index.max()}",
            "columns": list(self.df.columns),
            "train_size": int(len(self.df) * TRAIN_TEST_SPLIT),
            "test_size": len(self.df) - int(len(self.df) * TRAIN_TEST_SPLIT)
        }
