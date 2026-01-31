"""
Integration tests for the Stock Prediction project.
"""

import pytest


class TestProjectStructure:
    """Test overall project structure."""

    def test_backend_package_importable(self):
        """Test that backend package can be imported."""
        import backend
        assert backend is not None

    def test_backend_config_importable(self):
        """Test that backend.config can be imported."""
        from backend import config
        assert config is not None

    def test_backend_data_importable(self):
        """Test that backend.data can be imported."""
        from backend import data
        assert data is not None

    def test_backend_models_importable(self):
        """Test that backend.models can be imported."""
        from backend import models
        assert models is not None

    def test_preprocessor_importable(self):
        """Test that DataPreprocessor can be imported."""
        from backend.data.preprocessor import DataPreprocessor
        assert DataPreprocessor is not None

    def test_base_model_importable(self):
        """Test that BaseModel can be imported."""
        from backend.models.base import BaseModel
        assert BaseModel is not None


class TestDataDirectoryStructure:
    """Test data directory configuration."""

    def test_data_dir_configured(self):
        """Test DATA_DIR is configured correctly."""
        from backend.config import DATA_DIR
        assert DATA_DIR is not None
        assert DATA_DIR.name == "Data"

    def test_models_dir_configured(self):
        """Test MODELS_DIR is configured correctly."""
        from backend.config import MODELS_DIR
        assert MODELS_DIR is not None
        assert MODELS_DIR.name == "Models"


class TestModuleConsistency:
    """Test module consistency and configuration."""

    def test_all_model_types_covered(self):
        """Test all MODEL_NAMES are in a model type list."""
        from backend.config import (
            MODEL_NAMES,
            DEEP_LEARNING_MODELS,
            TIME_SERIES_MODELS,
            ML_MODELS
        )
        
        all_model_types = set(DEEP_LEARNING_MODELS + TIME_SERIES_MODELS + ML_MODELS)
        
        for model_name in MODEL_NAMES:
            assert model_name in all_model_types, f"{model_name} not in any model type list"

    def test_prediction_horizons_sorted(self):
        """Test PREDICTION_HORIZONS is sorted."""
        from backend.config import PREDICTION_HORIZONS
        
        assert PREDICTION_HORIZONS == sorted(PREDICTION_HORIZONS)

    def test_ipo_dates_valid_format(self):
        """Test IPO_DATES have valid date format."""
        from backend.config import IPO_DATES
        import datetime
        
        for ticker, date_str in IPO_DATES.items():
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                pytest.fail(f"Invalid date format for {ticker}: {date_str}")


class TestEndToEndSmoke:
    """Smoke tests for end-to-end functionality."""

    def test_preprocessor_can_be_created_for_all_tickers(self):
        """Test DataPreprocessor can be created for all tickers."""
        from backend.config import TICKERS
        from backend.data.preprocessor import DataPreprocessor
        
        for ticker in TICKERS:
            preprocessor = DataPreprocessor(ticker=ticker)
            assert preprocessor.ticker == ticker
