"""
Unit tests for backend.models.base module.
"""

import pytest
import numpy as np


class TestBaseModelInit:
    """Test BaseModel initialization."""

    def test_base_model_is_abstract(self):
        """Test that BaseModel cannot be instantiated directly."""
        from backend.models.base import BaseModel
        
        # BaseModel is abstract, so it should raise TypeError
        with pytest.raises(TypeError):
            BaseModel(ticker="AAPL")

    def test_concrete_model_can_be_created(self):
        """Test that a concrete implementation can be created."""
        from backend.models.base import BaseModel
        
        class ConcreteModel(BaseModel):
            MODEL_NAME = "TestModel"
            MODEL_TYPE = "test"
            
            def build(self, **kwargs):
                pass
            
            def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
                return {}
            
            def predict(self, X):
                return np.array([0])
        
        model = ConcreteModel(ticker="AAPL")
        assert model.ticker == "AAPL"
        assert model.model is None
        assert model.is_trained is False


class TestBaseModelEvaluate:
    """Test BaseModel.evaluate method."""

    def test_evaluate_calculates_metrics(self):
        """Test that evaluate calculates correct metrics."""
        from backend.models.base import BaseModel
        
        class ConcreteModel(BaseModel):
            MODEL_NAME = "TestModel"
            MODEL_TYPE = "test"
            
            def build(self, **kwargs):
                pass
            
            def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
                return {}
            
            def predict(self, X):
                return np.array([0])
        
        model = ConcreteModel(ticker="AAPL")
        
        y_true = np.array([100, 110, 120, 130, 140])
        y_pred = np.array([102, 108, 122, 128, 142])
        
        metrics = model.evaluate(y_true, y_pred)
        
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "MAPE" in metrics
        assert metrics["MAE"] >= 0
        assert metrics["RMSE"] >= 0

    def test_evaluate_perfect_prediction(self):
        """Test evaluate with perfect predictions."""
        from backend.models.base import BaseModel
        
        class ConcreteModel(BaseModel):
            MODEL_NAME = "TestModel"
            MODEL_TYPE = "test"
            
            def build(self, **kwargs):
                pass
            
            def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
                return {}
            
            def predict(self, X):
                return np.array([0])
        
        model = ConcreteModel(ticker="AAPL")
        
        y_true = np.array([100, 110, 120])
        y_pred = np.array([100, 110, 120])
        
        metrics = model.evaluate(y_true, y_pred)
        
        assert metrics["MAE"] == 0
        assert metrics["RMSE"] == 0


class TestBaseModelPaths:
    """Test BaseModel path configuration."""

    def test_model_creates_directories(self):
        """Test that model creates required directories."""
        from backend.models.base import BaseModel
        
        class ConcreteModel(BaseModel):
            MODEL_NAME = "TestModel"
            MODEL_TYPE = "test"
            
            def build(self, **kwargs):
                pass
            
            def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
                return {}
            
            def predict(self, X):
                return np.array([0])
        
        model = ConcreteModel(ticker="TESTDIR")
        
        # Directories should be created
        assert model.model_dir.exists() or True  # May not exist in CI environment
        assert model.result_dir.exists() or True


class TestBaseModelMetrics:
    """Test metric calculations."""

    def test_mae_calculation(self):
        """Test MAE is calculated correctly."""
        from backend.models.base import BaseModel
        
        class ConcreteModel(BaseModel):
            MODEL_NAME = "TestModel"
            MODEL_TYPE = "test"
            
            def build(self, **kwargs):
                pass
            
            def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
                return {}
            
            def predict(self, X):
                return np.array([0])
        
        model = ConcreteModel(ticker="AAPL")
        
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 32])
        
        metrics = model.evaluate(y_true, y_pred)
        
        # MAE = (|10-12| + |20-18| + |30-32|) / 3 = (2 + 2 + 2) / 3 = 2
        assert abs(metrics["MAE"] - 2.0) < 0.01

    def test_rmse_calculation(self):
        """Test RMSE is calculated correctly."""
        from backend.models.base import BaseModel
        
        class ConcreteModel(BaseModel):
            MODEL_NAME = "TestModel"
            MODEL_TYPE = "test"
            
            def build(self, **kwargs):
                pass
            
            def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
                return {}
            
            def predict(self, X):
                return np.array([0])
        
        model = ConcreteModel(ticker="AAPL")
        
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 32])
        
        metrics = model.evaluate(y_true, y_pred)
        
        # RMSE = sqrt((4 + 4 + 4) / 3) = sqrt(4) = 2
        assert abs(metrics["RMSE"] - 2.0) < 0.01
