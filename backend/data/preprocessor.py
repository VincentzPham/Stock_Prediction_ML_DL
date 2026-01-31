"""
Data Preprocessor Module
Xử lý và chuẩn bị dữ liệu cho các mô hình.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import joblib

from backend.config import (
    DATA_DIR,
    TRAIN_TEST_SPLIT,
    TIME_STEP,
    TARGET_COLUMN,
    PREDICTION_HORIZONS,
    MODELS_DIR,
)
from backend.data.sentiment_analyzer import SentimentAnalyzer


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

    def load_data(
        self, file_path: Optional[Path] = None, years_back: int = 15
    ) -> pd.DataFrame:
        """
        Đọc dữ liệu từ CSV file.

        Args:
            file_path: Đường dẫn file CSV. Nếu None, sử dụng default path.
            years_back: Số năm dữ liệu cần lấy (từ hiện tại trở về trước).
                       Mặc định 15 năm. Đặt 0 để lấy tất cả dữ liệu.

        Returns:
            DataFrame đã được xử lý.
        """
        if file_path is None:
            file_path = DATA_DIR / f"{self.ticker}.csv"

        # Đọc CSV, skip 2 hàng đầu (header của yfinance)
        df = pd.read_csv(file_path, skiprows=2)

        # Đặt tên cột
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

        # Chuyển Date thành datetime và set làm index
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # Sắp xếp theo thời gian
        df.sort_index(inplace=True)

        # Filter theo years_back (chỉ lấy dữ liệu gần đây)
        if years_back > 0:
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years_back)
            original_len = len(df)
            df = df[df.index >= cutoff_date]
            print(f"Filtered data: {original_len} -> {len(df)} rows (last {years_back} years)")

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
        df["Price_Diff"] = df["High"] - df["Low"]
        df["Avg_Price"] = (df["High"] + df["Low"]) / 2
        df["Price_Change"] = df["Close"] - df["Open"]

        # Volume feature
        df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(window=20).mean()

        # Returns
        df["Returns"] = df["Close"].pct_change()

        # Moving Averages
        df["MA_7"] = df["Close"].rolling(window=7).mean()
        df["MA_21"] = df["Close"].rolling(window=21).mean()
        df["MA_50"] = df["Close"].rolling(window=50).mean()

        # Volatility
        df["Volatility"] = df["Returns"].rolling(window=21).std()

        # RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # ============================================
        # SENTIMENT INTEGRATION (V2)
        # ============================================
        try:
            print(f"Adding sentiment data for {self.ticker}...")
            from data.sentiment_analyzer import SentimentAnalyzerV2
            
            analyzer = SentimentAnalyzerV2(decay_halflife=3, rolling_window=3)
            sentiment_df = analyzer.get_daily_sentiment(self.ticker, days_back=30)

            if not sentiment_df.empty:
                # Merge with main dataframe using left join
                df = df.join(sentiment_df[["Sentiment_Score"]], how="left")

                # Fill missing sentiment with NEUTRAL (0) to avoid leakage
                # Do NOT use ffill() as it can leak future information
                df["Sentiment_Score"] = df["Sentiment_Score"].fillna(0)
                print("  -> Sentiment data merged (neutral fill for missing).")
            else:
                print("  -> No sentiment data found. Filling with 0.")
                df["Sentiment_Score"] = 0.0

        except Exception as e:
            print(f"Warning: Could not add sentiment features: {e}")
            df["Sentiment_Score"] = 0.0

        # Loại bỏ NaN sau khi thêm features
        df.dropna(inplace=True)

        self.df = df
        return df

    def get_train_test_split(
        self, split_ratio: float = TRAIN_TEST_SPLIT
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
        target_col: str = TARGET_COLUMN,
        fit_scaler: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Chuẩn bị dữ liệu cho LSTM/RNN models.

        Args:
            time_step: Số bước thời gian lookback.
            horizon: Số ngày dự đoán tiếp theo (1 = next day).
            target_col: Tên cột target.
            fit_scaler: If True, fit scaler on train data. If False, use existing scaler.

        Returns:
            Tuple (X_train, X_test, y_train, y_test, scaler)
        """
        if self.df is None:
            raise ValueError("Chưa load data. Gọi load_data() trước.")

        # Lấy target column
        data = self.df[target_col].values.reshape(-1, 1)

        # Split TRƯỚC khi scale để tránh leakage
        train_size = int(len(data) * TRAIN_TEST_SPLIT)
        train_data_raw = data[:train_size]
        test_data_raw = data[train_size:]

        # Scale: Fit ONLY on train, transform cả hai
        if fit_scaler:
            self.target_scaler.fit(train_data_raw)
            self._is_fitted = True

        if not self._is_fitted:
            raise ValueError("Scaler chưa được fit. Gọi với fit_scaler=True trước.")

        train_data = self.target_scaler.transform(train_data_raw)
        test_data = self.target_scaler.transform(test_data_raw)

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
        target_col: str = TARGET_COLUMN,
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
            feature_cols = [
                "Open",
                "High",
                "Low",
                "Volume",
                "Price_Diff",
                "Avg_Price",
                "Volume_Ratio",
            ]
            # Chỉ lấy các cột có trong df
            feature_cols = [col for col in feature_cols if col in self.df.columns]

        # Tạo target shift theo horizon
        # y tại index t là giá Close tại t + horizon
        df_ml = self.df.copy()
        df_ml["Target"] = df_ml[target_col].shift(-horizon)

        # Drop NaN do shift (các dòng cuối không có target tương lai)
        df_ml.dropna(inplace=True)

        X = df_ml[feature_cols].values
        y = df_ml["Target"].values

        # Split
        split_idx = int(len(X) * TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    def _create_sequences(
        self, data: np.ndarray, time_step: int, horizon: int = 1
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
            X.append(data[i - time_step : i, 0])
            y.append(data[i + horizon - 1, 0])

        return np.array(X), np.array(y)

    def prepare_timeseries_data(
        self, target_col: str = TARGET_COLUMN
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
        self, target_col: str = TARGET_COLUMN
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
        prophet_df = pd.DataFrame(
            {"ds": self.df.index, "y": self.df[target_col].values}
        )

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

    def get_dates(self, dataset: str = "test") -> pd.DatetimeIndex:
        """
        Lấy dates cho train hoặc test set.
        """
        if self.df is None:
            raise ValueError("Chưa load data.")

        split_idx = int(len(self.df) * TRAIN_TEST_SPLIT)

        if dataset == "train":
            return self.df.index[:split_idx]
        else:
            return self.df.index[split_idx:]

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
            "test_size": len(self.df) - int(len(self.df) * TRAIN_TEST_SPLIT),
        }

    # ============================================
    # RETURNS-BASED METHODS (Scale-Invariant)
    # ============================================

    def prepare_returns_multi_horizon_data(
        self,
        time_step: int = TIME_STEP,
        horizons: List[int] = None,
        target_col: str = TARGET_COLUMN,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Prepare data using RETURNS instead of absolute prices.
        
        This solves the scaler range problem - returns are bounded and
        don't suffer from out-of-range issues when prices exceed training max.
        
        Args:
            time_step: Lookback window size.
            horizons: List of forecast horizons [1, 3, 7, 14, 30, 60].
            target_col: Target column name.
            
        Returns:
            X_train, X_test, y_train, y_test: Arrays for training
            last_prices: Array of last known prices for each sample
            metadata: Dict with scaling info
        """
        if self.df is None:
            raise ValueError("Chưa load data.")
        
        if horizons is None:
            horizons = PREDICTION_HORIZONS
        
        prices = self.df[target_col].values
        
        # Calculate log returns (more stable than simple returns)
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Split before any scaling
        train_size = int(len(log_returns) * TRAIN_TEST_SPLIT)
        train_returns = log_returns[:train_size]
        test_returns = log_returns[train_size:]
        
        # Store prices (shifted by 1 because returns start from index 1)
        train_prices = prices[1:train_size + 1]
        test_prices = prices[train_size + 1:]
        
        # Scale returns using StandardScaler (returns are centered around 0)
        from sklearn.preprocessing import StandardScaler
        self.returns_scaler = StandardScaler()
        train_returns_scaled = self.returns_scaler.fit_transform(train_returns.reshape(-1, 1)).flatten()
        test_returns_scaled = self.returns_scaler.transform(test_returns.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train, y_train, prices_train = self._create_returns_sequences(
            train_returns_scaled, train_prices, time_step, horizons, log_returns[:train_size]
        )
        X_test, y_test, prices_test = self._create_returns_sequences(
            test_returns_scaled, test_prices, time_step, horizons, log_returns[train_size:]
        )
        
        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        metadata = {
            "horizons": horizons,
            "returns_scaler": self.returns_scaler,
            "train_price_range": (train_prices.min(), train_prices.max()),
            "test_price_range": (test_prices.min(), test_prices.max()),
        }
        
        # Return last_prices dict for price conversion
        last_prices = {
            "train": prices_train,
            "test": prices_test
        }
        
        return X_train, X_test, y_train, y_test, self.returns_scaler, last_prices

    def _create_returns_sequences(
        self, 
        returns_scaled: np.ndarray, 
        prices: np.ndarray,
        time_step: int, 
        horizons: List[int],
        raw_returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for returns-based prediction.
        
        Target: Cumulative log returns for each horizon
        """
        max_horizon = max(horizons)
        X, y, last_prices = [], [], []
        
        for i in range(time_step, len(returns_scaled) - max_horizon + 1):
            # Input: scaled returns for lookback window
            X.append(returns_scaled[i - time_step : i])
            
            # Store the last known price (at index i-1 in original price array)
            # This is the price BEFORE the forecast period starts
            last_prices.append(prices[i - 1])
            
            # Target: cumulative log returns for each horizon
            targets = []
            for h in horizons:
                # Sum of log returns from i to i+h-1
                cum_return = np.sum(raw_returns[i : i + h])
                targets.append(cum_return)
            y.append(targets)
        
        return np.array(X), np.array(y), np.array(last_prices)

    def get_last_sequence_for_returns(
        self, time_step: int = TIME_STEP, target_col: str = TARGET_COLUMN
    ) -> Tuple[np.ndarray, float]:
        """
        Get the last sequence for making returns-based predictions.
        
        Returns:
            X: Input sequence for prediction
            last_price: The last known price (to convert returns to prices)
        """
        if self.df is None:
            raise ValueError("Chưa load data.")
        
        if not hasattr(self, 'returns_scaler'):
            raise ValueError("Returns scaler not fitted. Train first.")
        
        prices = self.df[target_col].values
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Get last sequence of returns
        last_returns = log_returns[-time_step:]
        last_returns_scaled = self.returns_scaler.transform(last_returns.reshape(-1, 1)).flatten()
        
        # Shape for LSTM: (1, time_step, 1)
        X = last_returns_scaled.reshape(1, time_step, 1)
        
        # Last known price
        last_price = prices[-1]
        
        return X, last_price

    def returns_to_prices(
        self, cum_log_returns: np.ndarray, last_price: float
    ) -> np.ndarray:
        """
        Convert cumulative log returns to actual prices.
        
        Args:
            cum_log_returns: Predicted cumulative log returns for each horizon
            last_price: The last known price
            
        Returns:
            Predicted prices for each horizon
        """
        # Price_future = Price_current * exp(cum_log_return)
        return last_price * np.exp(cum_log_returns)

    # ============================================
    # MULTI-HORIZON METHODS
    # ============================================

    def prepare_multi_horizon_data(
        self,
        time_step: int = TIME_STEP,
        horizons: List[int] = None,
        target_col: str = TARGET_COLUMN,
        fit_scaler: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Prepare data for multi-horizon prediction (Direct Multi-Output approach).
        
        Args:
            time_step: Lookback window size.
            horizons: List of forecast horizons (default: PREDICTION_HORIZONS).
            target_col: Target column name.
            fit_scaler: If True, fit scaler on train data.
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test, scaler)
            - y_train/y_test shape: (n_samples, len(horizons))
        """
        if self.df is None:
            raise ValueError("Chưa load data. Gọi load_data() trước.")
        
        if horizons is None:
            horizons = PREDICTION_HORIZONS
        
        max_horizon = max(horizons)
        
        # Get target data
        data = self.df[target_col].values.reshape(-1, 1)
        
        # Split TRƯỚC khi scale để tránh leakage
        train_size = int(len(data) * TRAIN_TEST_SPLIT)
        train_data_raw = data[:train_size]
        test_data_raw = data[train_size:]
        
        # Scale: Fit ONLY on train
        if fit_scaler:
            self.target_scaler.fit(train_data_raw)
            self._is_fitted = True
        
        if not self._is_fitted:
            raise ValueError("Scaler chưa được fit.")
        
        train_data = self.target_scaler.transform(train_data_raw)
        test_data = self.target_scaler.transform(test_data_raw)
        
        # Create multi-horizon sequences
        X_train, y_train = self._create_multi_horizon_sequences(
            train_data, time_step, horizons
        )
        X_test, y_test = self._create_multi_horizon_sequences(
            test_data, time_step, horizons
        )
        
        # Reshape X for LSTM (samples, time_steps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return X_train, X_test, y_train, y_test, self.target_scaler

    def _create_multi_horizon_sequences(
        self, data: np.ndarray, time_step: int, horizons: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with multiple horizon targets.
        
        Returns:
            X: shape (n_samples, time_step)
            y: shape (n_samples, len(horizons))
        """
        max_horizon = max(horizons)
        X, y = [], []
        
        for i in range(time_step, len(data) - max_horizon + 1):
            X.append(data[i - time_step : i, 0])
            # Collect targets for all horizons
            targets = [data[i + h - 1, 0] for h in horizons]
            y.append(targets)
        
        return np.array(X), np.array(y)

    # ============================================
    # SCALER PERSISTENCE
    # ============================================

    def save_scaler(self, model_name: str) -> Path:
        """
        Save fitted scaler for a specific model.
        
        Args:
            model_name: Name of the model using this scaler.
            
        Returns:
            Path to saved scaler file.
        """
        if not self._is_fitted:
            raise ValueError("Scaler chưa được fit.")
        
        scaler_dir = MODELS_DIR / self.ticker / model_name
        scaler_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = scaler_dir / "scaler.pkl"
        
        joblib.dump(self.target_scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        return scaler_path

    def load_scaler(self, model_name: str) -> MinMaxScaler:
        """
        Load a previously saved scaler.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Loaded MinMaxScaler instance.
        """
        scaler_path = MODELS_DIR / self.ticker / model_name / "scaler.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.target_scaler = joblib.load(scaler_path)
        self._is_fitted = True
        print(f"Scaler loaded from {scaler_path}")
        return self.target_scaler

    def get_last_sequence(
        self, 
        time_step: int = TIME_STEP, 
        target_col: str = TARGET_COLUMN
    ) -> Tuple[np.ndarray, float]:
        """
        Get the last sequence for prediction (scaled).
        
        Returns:
            Tuple (X_input, last_actual_price)
            - X_input shape: (1, time_step, 1) ready for model.predict()
        """
        if self.df is None:
            raise ValueError("Chưa load data.")
        if not self._is_fitted:
            raise ValueError("Scaler chưa được fit.")
        
        data = self.df[target_col].values.reshape(-1, 1)
        last_actual = float(data[-1][0])
        
        scaled_data = self.target_scaler.transform(data)
        last_sequence = scaled_data[-time_step:]
        X_input = last_sequence.reshape(1, time_step, 1)
        
        return X_input, last_actual

    # ============================================
    # RETURNS-BASED HELPERS
    # ============================================

    def save_returns_scaler(self, model_name: str) -> Path:
        """
        Save returns scaler for a specific model.
        """
        if self.returns_scaler is None:
            raise ValueError("Returns scaler chưa được fit.")
        
        scaler_dir = MODELS_DIR / self.ticker / model_name
        scaler_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = scaler_dir / "returns_scaler.pkl"
        
        joblib.dump(self.returns_scaler, scaler_path)
        print(f"Returns scaler saved to {scaler_path}")
        return scaler_path

    def load_returns_scaler(self, model_name: str) -> MinMaxScaler:
        """
        Load a previously saved returns scaler.
        """
        scaler_path = MODELS_DIR / self.ticker / model_name / "returns_scaler.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Returns scaler not found: {scaler_path}")
        
        self.returns_scaler = joblib.load(scaler_path)
        print(f"Returns scaler loaded from {scaler_path}")
        return self.returns_scaler

    def returns_to_prices(
        self, 
        cumulative_returns: np.ndarray, 
        base_prices: np.ndarray
    ) -> np.ndarray:
        """
        Convert cumulative log returns back to absolute prices.
        
        Args:
            cumulative_returns: Predicted cumulative log returns
            base_prices: Base prices (last known prices before prediction)
            
        Returns:
            Predicted prices
        """
        # Handle scalar base_prices
        if np.isscalar(base_prices):
            base_prices = np.full_like(cumulative_returns, base_prices)
        
        # Handle 1D base_prices with different length
        if len(base_prices) == 1:
            base_prices = np.full_like(cumulative_returns, base_prices[0])
        
        # Price = base_price * exp(cumulative_return)
        prices = base_prices * np.exp(cumulative_returns)
        return prices

    def get_last_sequence_for_returns(
        self, 
        time_step: int = TIME_STEP, 
        target_col: str = TARGET_COLUMN
    ) -> Tuple[np.ndarray, float]:
        """
        Get the last sequence for returns-based prediction.
        
        Returns:
            Tuple (X_input, last_price)
            - X_input: Scaled log returns sequence, shape (1, time_step, 1)
            - last_price: Last actual price for converting returns to prices
        """
        if self.df is None:
            raise ValueError("Chưa load data.")
        if self.returns_scaler is None:
            raise ValueError("Returns scaler chưa được load. Call load_returns_scaler() first.")
        
        prices = self.df[target_col].values
        last_price = float(prices[-1])
        
        # Calculate log returns
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Scale returns
        returns_scaled = self.returns_scaler.transform(log_returns.reshape(-1, 1))
        
        # Get last sequence
        last_sequence = returns_scaled[-time_step:]
        X_input = last_sequence.reshape(1, time_step, 1)
        
        return X_input, last_price
