"""
Unit tests for backend.config module.
"""

from pathlib import Path


class TestConfigPaths:
    """Test configuration paths."""

    def test_root_dir_exists(self):
        """Test that ROOT_DIR is properly defined."""
        from backend.config import ROOT_DIR
        assert ROOT_DIR is not None
        assert isinstance(ROOT_DIR, Path)

    def test_data_dir_defined(self):
        """Test DATA_DIR is properly defined."""
        from backend.config import DATA_DIR
        assert DATA_DIR is not None
        assert isinstance(DATA_DIR, Path)
        assert DATA_DIR.name == "Data"

    def test_models_dir_defined(self):
        """Test MODELS_DIR is properly defined."""
        from backend.config import MODELS_DIR
        assert MODELS_DIR is not None
        assert isinstance(MODELS_DIR, Path)
        assert MODELS_DIR.name == "Models"

    def test_results_dir_defined(self):
        """Test RESULTS_DIR is properly defined."""
        from backend.config import RESULTS_DIR
        assert RESULTS_DIR is not None
        assert isinstance(RESULTS_DIR, Path)
        assert RESULTS_DIR.name == "Result"


class TestConfigTickers:
    """Test ticker configuration."""

    def test_tickers_not_empty(self):
        """Test that TICKERS list is not empty."""
        from backend.config import TICKERS
        assert len(TICKERS) > 0

    def test_tickers_contains_expected_symbols(self):
        """Test TICKERS contains expected stock symbols."""
        from backend.config import TICKERS
        expected = ["AAPL", "MSFT", "GOOG", "AMZN"]
        for symbol in expected:
            assert symbol in TICKERS

    def test_crypto_tickers_defined(self):
        """Test CRYPTO_TICKERS is properly defined."""
        from backend.config import CRYPTO_TICKERS
        assert isinstance(CRYPTO_TICKERS, list)
        assert "BTC-USD" in CRYPTO_TICKERS

    def test_ipo_dates_contains_all_tickers(self):
        """Test IPO_DATES has entries for all tickers."""
        from backend.config import TICKERS, IPO_DATES
        for ticker in TICKERS:
            assert ticker in IPO_DATES


class TestConfigModelNames:
    """Test model configuration."""

    def test_model_names_not_empty(self):
        """Test MODEL_NAMES is not empty."""
        from backend.config import MODEL_NAMES
        assert len(MODEL_NAMES) > 0

    def test_model_names_contains_lstm(self):
        """Test MODEL_NAMES contains LSTM."""
        from backend.config import MODEL_NAMES
        assert "LSTM" in MODEL_NAMES

    def test_model_types_defined(self):
        """Test model type lists are defined."""
        from backend.config import DEEP_LEARNING_MODELS, TIME_SERIES_MODELS, ML_MODELS
        assert len(DEEP_LEARNING_MODELS) > 0
        assert len(TIME_SERIES_MODELS) > 0
        assert len(ML_MODELS) > 0


class TestConfigDataProcessing:
    """Test data processing configuration."""

    def test_train_test_split_valid(self):
        """Test TRAIN_TEST_SPLIT is a valid ratio."""
        from backend.config import TRAIN_TEST_SPLIT
        assert 0 < TRAIN_TEST_SPLIT < 1
        assert TRAIN_TEST_SPLIT == 0.8

    def test_time_step_positive(self):
        """Test TIME_STEP is positive."""
        from backend.config import TIME_STEP
        assert TIME_STEP > 0
        assert TIME_STEP == 60

    def test_feature_columns_defined(self):
        """Test FEATURE_COLUMNS is defined."""
        from backend.config import FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) > 0
        assert "Close" in FEATURE_COLUMNS

    def test_target_column_defined(self):
        """Test TARGET_COLUMN is defined."""
        from backend.config import TARGET_COLUMN
        assert TARGET_COLUMN == "Close"

    def test_prediction_horizons_defined(self):
        """Test PREDICTION_HORIZONS is defined and valid."""
        from backend.config import PREDICTION_HORIZONS
        assert len(PREDICTION_HORIZONS) > 0
        assert all(h > 0 for h in PREDICTION_HORIZONS)


class TestConfigWalkForward:
    """Test walk-forward validation configuration."""

    def test_walk_forward_config_defined(self):
        """Test WALK_FORWARD_CONFIG is properly defined."""
        from backend.config import WALK_FORWARD_CONFIG
        assert isinstance(WALK_FORWARD_CONFIG, dict)
        required_keys = ["mode", "retrain_every", "test_window", "embargo", "final_holdout"]
        for key in required_keys:
            assert key in WALK_FORWARD_CONFIG

    def test_walk_forward_mode_valid(self):
        """Test walk-forward mode is valid."""
        from backend.config import WALK_FORWARD_CONFIG
        assert WALK_FORWARD_CONFIG["mode"] in ["expanding", "rolling"]
