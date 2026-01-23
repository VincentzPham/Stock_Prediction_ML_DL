"""
Hyperparameter Tuner Module
Uses Optuna to optimize model hyperparameters.
"""

import optuna
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from config import SEARCH_SPACES, DEEP_LEARNING_MODELS, ML_MODELS, TIME_STEP
from config import SEARCH_SPACES, DEEP_LEARNING_MODELS, ML_MODELS, TIME_STEP

class HyperparameterTuner:
    """Tuner class using Optuna."""
    
    def __init__(self, ticker: str, n_trials: int = 20):
        self.ticker = ticker
        self.n_trials = n_trials
        
    def optimize(self, model_name: str, preprocessor, model_registry, **kwargs) -> Dict[str, Any]:
        """
        Run optimization for a specific model.
        Returns the best parameters.
        """
        if model_name not in SEARCH_SPACES:
            print(f"No search space defined for {model_name}. Using default config.")
            return {}
            
        print(f"\n[TUNING] Tuning {model_name} for {self.ticker} ({self.n_trials} trials)...")
        
        # Prepare data once
        if model_name in DEEP_LEARNING_MODELS:
            X_train, X_val, y_train, y_val, _ = preprocessor.prepare_lstm_data(time_step=TIME_STEP)
            # Use small validation set for tuning speed
            validation_data = (X_val, y_val)
        elif model_name in ML_MODELS:
            X_train, X_val, y_train, y_val = preprocessor.prepare_ml_data()
            validation_data = (X_val, y_val)
        else:
            return {} # Skip TS models
            
        def objective(trial):
            # hyperparams
            params = self._suggest_params(trial, model_name)
            
            # Build & Train
            ModelClass = model_registry[model_name]
            model = ModelClass(self.ticker)
            
            # Update config with suggested params (temporary override)
            if hasattr(model, 'config'):
                model.config.update(params)
                
            try:
                if model_name in DEEP_LEARNING_MODELS:
                    model.build(
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        **params
                    )
                    history = model.train(
                        X_train, y_train,
                        X_val=X_val, y_val=y_val,
                        epochs=20,  # Limits epochs for faster tuning
                        batch_size=32,
                        patience=3, # Fail fast
                        verbose=0
                    )
                    # Minimize Validation Loss (MSE)
                    val_loss = history['val_loss'][-1]
                    return val_loss
                    
                elif model_name in ML_MODELS:
                    model.build(**params)
                    model.train(X_train, y_train)
                    # Evaluate on validation
                    preds = model.predict(X_val)
                    mse = np.mean((y_val - preds) ** 2)
                    return mse
                    
            except Exception as e:
                # Prune failed trials
                print(f"Trial failed: {e}")
                raise optuna.exceptions.TrialPruned()
                
            return float('inf')

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        
        print(f"[BEST] Best params: {study.best_params}")
        return study.best_params

    def _suggest_params(self, trial, model_name) -> Dict[str, Any]:
        """Suggest parameters based on search space."""
        space = SEARCH_SPACES[model_name]
        params = {}
        
        # LSTM/RNN/Deep Learning
        if 'units_min' in space:
            n_layers = trial.suggest_int('n_layers', space.get('layers_min', 1), space.get('layers_max', 2))
            # Create list of units for each layer (e.g., [64, 32])
            units = []
            for i in range(n_layers):
                units.append(trial.suggest_int(f'units_l{i}', space['units_min'], space['units_max']))
            params['units'] = units
            
        if 'dropout_min' in space:
            params['dropout'] = trial.suggest_float('dropout', space['dropout_min'], space['dropout_max'])
            
        if 'lr_min' in space:
            params['learning_rate'] = trial.suggest_float('learning_rate', space['lr_min'], space['lr_max'], log=True)
            
        # Machine Learning
        if 'n_estimators_min' in space:
            params['n_estimators'] = trial.suggest_int('n_estimators', space['n_estimators_min'], space['n_estimators_max'])
            
        if 'max_depth_min' in space:
             params['max_depth'] = trial.suggest_int('max_depth', space['max_depth_min'], space['max_depth_max'])
             
        if 'min_samples_split_min' in space:
            params['min_samples_split'] = trial.suggest_int('min_samples_split', space['min_samples_split_min'], space['min_samples_split_max'])
            
        return params
