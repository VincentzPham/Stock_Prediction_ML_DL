"""
Unit tests for backend.data.preprocessor module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestDataPreprocessorInit:
    """Test DataPreprocessor initialization."""

    def test_init_creates_instance(self):
        """Test that DataPreprocessor can be instantiated."""
        from backend.data.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor(ticker="AAPL")
        assert preprocessor.ticker == "AAPL"
        assert preprocessor.df is None
        assert preprocessor._is_fitted is False

    def test_init_creates_scalers(self):
        """Test that scalers are created on init."""
        from backend.data.preprocessor import DataPreprocessor
        from sklearn.preprocessing import MinMaxScaler
        
        preprocessor = DataPreprocessor(ticker="MSFT")
        assert preprocessor.scaler is not None
        assert preprocessor.target_scaler is not None
        assert isinstance(preprocessor.scaler, MinMaxScaler)


class TestDataPreprocessorLoadData:
    """Test DataPreprocessor.load_data method."""

    def test_load_data_from_csv(self, tmp_path):
        """Test loading data from a CSV file."""
        from backend.data.preprocessor import DataPreprocessor
        
        # Create a mock CSV file with recent dates
        csv_content = """Ticker,AAPL
Date,Close,High,Low,Open,Volume
2025-01-01,100.0,105.0,95.0,98.0,1000000
2025-01-02,102.0,107.0,99.0,100.0,1200000
2025-01-03,104.0,108.0,101.0,102.0,1100000
"""
        csv_file = tmp_path / "TEST.csv"
        csv_file.write_text(csv_content)
        
        preprocessor = DataPreprocessor(ticker="TEST")
        df = preprocessor.load_data(file_path=csv_file, years_back=0)
        
        assert df is not None
        assert len(df) >= 1  # At least some data loaded
        assert "Close" in df.columns
        assert preprocessor.df is not None

    def test_load_data_sets_df_attribute(self, tmp_path):
        """Test that load_data sets the df attribute."""
        from backend.data.preprocessor import DataPreprocessor
        
        csv_content = """Ticker,AAPL
Date,Close,High,Low,Open,Volume
2024-01-01,100.0,105.0,95.0,98.0,1000000
"""
        csv_file = tmp_path / "TEST.csv"
        csv_file.write_text(csv_content)
        
        preprocessor = DataPreprocessor(ticker="TEST")
        assert preprocessor.df is None
        
        preprocessor.load_data(file_path=csv_file, years_back=0)
        assert preprocessor.df is not None


class TestDataPreprocessorTrainTestSplit:
    """Test DataPreprocessor.get_train_test_split method."""

    def test_split_raises_without_data(self):
        """Test that get_train_test_split raises error without data."""
        from backend.data.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor(ticker="TEST")
        with pytest.raises(ValueError, match="Chưa load data"):
            preprocessor.get_train_test_split()

    def test_split_preserves_data_order(self):
        """Test that split preserves chronological order."""
        from backend.data.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor(ticker="TEST")
        
        # Create mock dataframe
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        preprocessor.df = pd.DataFrame({
            "Close": range(100),
            "High": range(100),
            "Low": range(100),
            "Open": range(100),
            "Volume": range(100)
        }, index=dates)
        
        train_df, test_df = preprocessor.get_train_test_split(split_ratio=0.8)
        
        assert len(train_df) == 80
        assert len(test_df) == 20
        assert train_df.index[-1] < test_df.index[0]


class TestDataPreprocessorAddFeatures:
    """Test DataPreprocessor.add_features method."""

    def test_add_features_raises_without_data(self):
        """Test that add_features raises error without data."""
        from backend.data.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor(ticker="TEST")
        with pytest.raises(ValueError, match="Chưa load data"):
            preprocessor.add_features()

    @patch("backend.data.sentiment_analyzer.SentimentAnalyzerV2")
    def test_add_features_creates_technical_indicators(self, mock_sentiment):
        """Test that add_features creates technical indicators."""
        from backend.data.preprocessor import DataPreprocessor
        
        # Mock sentiment analyzer to return empty dataframe
        mock_instance = MagicMock()
        mock_instance.get_daily_sentiment.return_value = pd.DataFrame()
        mock_sentiment.return_value = mock_instance
        
        preprocessor = DataPreprocessor(ticker="TEST")
        
        # Create mock dataframe with enough data for rolling calculations
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        preprocessor.df = pd.DataFrame({
            "Close": np.random.uniform(100, 200, 100),
            "High": np.random.uniform(100, 200, 100),
            "Low": np.random.uniform(100, 200, 100),
            "Open": np.random.uniform(100, 200, 100),
            "Volume": np.random.uniform(1000000, 2000000, 100)
        }, index=dates)
        
        df = preprocessor.add_features()
        
        # Check technical indicators were added
        assert "Price_Diff" in df.columns
        assert "Returns" in df.columns
        assert "MA_7" in df.columns
        assert "RSI" in df.columns
        assert "Sentiment_Score" in df.columns


class TestDataPreprocessorHelperFunctions:
    """Test helper functions in preprocessor module."""

    def test_create_sequences(self):
        """Test sequence creation for LSTM."""
        from backend.data.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor(ticker="TEST")
        
        # Create simple data
        data = np.array([[i] for i in range(10)])
        X, y = preprocessor._create_sequences(data, time_step=3, horizon=1)
        
        # With 10 points, time_step=3, horizon=1:
        # We should get sequences like [0,1,2] -> 3, [1,2,3] -> 4, etc.
        assert len(X) == len(y)
        assert X.shape[1] == 3  # time_step

    def test_inverse_transform_predictions(self):
        """Test inverse transform of predictions."""
        from backend.data.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor(ticker="TEST")
        
        # Fit the scaler
        original_data = np.array([[100], [200], [150]])
        preprocessor.target_scaler.fit(original_data)
        
        # Scale and inverse
        scaled = preprocessor.target_scaler.transform(original_data)
        inversed = preprocessor.target_scaler.inverse_transform(scaled)
        
        np.testing.assert_array_almost_equal(original_data, inversed)
