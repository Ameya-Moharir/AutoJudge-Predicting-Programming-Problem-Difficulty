"""
Classification models for difficulty prediction
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import joblib
from typing import Dict, Any
from ..utils.logger import logger


class DifficultyClassifier:
    """Ensemble classifier for problem difficulty"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize classifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.classes_ = ['easy', 'medium', 'hard']
        self._build_model()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs':-1
            },
            'xgboost': {
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 7,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'mlogloss'
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
                'solver': 'lbfgs'
            },
            'ensemble_weights': [0.4, 0.35, 0.25],
            'use_smote': True,
            'smote_k_neighbors': 5
        }
    
    def _build_model(self):
        """Build ensemble model"""
        # Random Forest
        rf = RandomForestClassifier(**self.config['random_forest'])
        
        # XGBoost
        xgb = XGBClassifier(**self.config['xgboost'])
        
        # Logistic Regression
        lr = LogisticRegression(**self.config['logistic_regression'])
        
        # Voting Classifier
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)],
            voting='soft',
            weights=self.config['ensemble_weights']
        )
        
        logger.info("Built ensemble classifier")
    
    def fit(self, X_train, y_train):
        """
        Train the classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info(f"Training classifier on {len(X_train)} samples")
        
        self.model.fit(X_train, y_train)
        
        # Get training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        logger.info(f"Training accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict difficulty class
        
        Args:
            X: Features
        
        Returns:
            Predicted classes
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features
        
        Returns:
            Class probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        """
        Evaluate the classifier
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating classifier...")
        
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
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
            logger.info("\nTop 20 important features:")
            for i in range(min(20, len(indices))):
                idx = indices[i]
                logger.info(f"{i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
        
        return importance
    
    def save(self, filepath: str):
        """Save model to file"""
        joblib.dump(self.model, filepath)
        logger.info(f"Saved classifier to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = joblib.load(filepath)
        logger.info(f"Loaded classifier from {filepath}")
        return self


class SingleModelClassifier:
    """Single model classifier wrapper"""
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        Initialize single model classifier
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'logistic')
            **kwargs: Model parameters
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs):
        """Create model based on type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 15),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 150),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 7),
                random_state=kwargs.get('random_state', 42)
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                C=kwargs.get('C', 1.0),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Predict classes"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
