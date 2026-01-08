"""
Regression models for difficulty score prediction
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Dict, Any
from ..utils.logger import logger


class DifficultyScoreRegressor:
    """Ensemble regressor for problem difficulty score"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize regressor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.model = None
        self._build_model()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'gradient_boosting': {
                'n_estimators': 250,  # Reduced from 300
                'learning_rate': 0.05,  # Increased from 0.05
                'max_depth': 6,
                'min_samples_split': 10,
                'random_state': 42,
                'verbose': 0,
                'subsample': 0.8    
            },
            'random_forest': {
                'n_estimators': 200,  # Reduced from 200
                'max_depth': 15,  # Reduced from 12
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,  # Use all cores
                'verbose': 0  # Show progress
            },
            'ensemble_weights': [0.6, 0.4]  # No SVR weight
    }
    
    def _build_model(self):
        """Build ensemble model"""
        # Gradient Boosting
        gb = GradientBoostingRegressor(**self.config['gradient_boosting'])
        
        # Random Forest
        rf = RandomForestRegressor(**self.config['random_forest'])
        
    
        
        # Voting Regressor
        self.model = VotingRegressor(
            estimators=[('gb', gb), ('rf', rf)],
            weights=self.config['ensemble_weights']
        )
        
        logger.info("Built ensemble regressor (GB + RF)")
    
    def fit(self, X_train, y_train):
        """
        Train the regressor
        
        Args:
            X_train: Training features
            y_train: Training scores
        """
        logger.info(f"Training regressor on {len(X_train)} samples")
        logger.info(f"Feature dimensions: {X_train.shape}")  # ← ADD THIS
        logger.info("Starting model.fit()...")
        self.model.fit(X_train, y_train)
        logger.info("Fit complete, calculating metrics...")
        # Get training metrics
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        logger.info(f"Training RMSE: {train_rmse:.4f}")
        logger.info(f"Training MAE: {train_mae:.4f}")
        logger.info(f"Training R²: {train_r2:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict difficulty scores
        
        Args:
            X: Features
        
        Returns:
            Predicted scores
        """
        predictions = self.model.predict(X)
        # Clip predictions to valid range [0, 10]
        predictions = np.clip(predictions, 0, 10)
        return predictions
    
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """
        Evaluate the regressor
        
        Args:
            X_test: Test features
            y_test: Test scores
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating regressor...")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.1))) * 100
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred,
            'actuals': y_test
        }
        
        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        logger.info(f"Test R²: {r2:.4f}")
        logger.info(f"Test MAPE: {mape:.2f}%")
        
        return results
    
    def get_feature_importance(self, feature_names=None) -> np.ndarray:
        """
        Get feature importance from Random Forest
        
        Args:
            feature_names: List of feature names
        
        Returns:
            Feature importance array
        """
        # Get RF model from ensemble
        rf_model = self.model.named_estimators_['rf']
        importance = rf_model.feature_importances_
        
        if feature_names:
            # Sort by importance
            indices = np.argsort(importance)[::-1]
            logger.info("\nTop 20 important features for regression:")
            for i in range(min(20, len(indices))):
                idx = indices[i]
                logger.info(f"{i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
        
        return importance
    
    def save(self, filepath: str):
        """Save model to file"""
        joblib.dump(self.model, filepath)
        logger.info(f"Saved regressor to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = joblib.load(filepath)
        logger.info(f"Loaded regressor from {filepath}")
        return self


class SingleModelRegressor:
    """Single model regressor wrapper"""
    
    def __init__(self, model_type='gradient_boosting', **kwargs):
        """
        Initialize single model regressor
        
        Args:
            model_type: Type of model ('gradient_boosting', 'random_forest', 'svr')
            **kwargs: Model parameters
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs):
        """Create model based on type"""
        if self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 300),
                learning_rate=kwargs.get('learning_rate', 0.05),
                max_depth=kwargs.get('max_depth', 5),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 12),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'svr':
            return SVR(
                C=kwargs.get('C', 10.0),
                epsilon=kwargs.get('epsilon', 0.1),
                kernel=kwargs.get('kernel', 'rbf')
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Predict scores"""
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, 10)
