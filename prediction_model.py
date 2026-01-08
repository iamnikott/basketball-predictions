"""
Basketball Match Prediction Model

This module contains the PredictionModel class that uses Gradient Boosting
to predict basketball match outcomes and provide probability scores.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Union


class PredictionModel:
    """
    A machine learning model for predicting basketball match outcomes.
    
    Uses GradientBoostingClassifier to train on basketball-specific features
    and predict match outcomes with probability scores.
    
    Attributes:
        model (GradientBoostingClassifier): The underlying gradient boosting classifier
        scaler (StandardScaler): Feature scaler for normalization
        is_trained (bool): Whether the model has been trained
        feature_names (List[str]): Names of features used in training
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42,
        subsample: float = 0.8
    ):
        """
        Initialize the PredictionModel.
        
        Args:
            n_estimators (int): Number of boosting stages. Default: 100
            learning_rate (float): Learning rate shrinks contribution of each tree. Default: 0.1
            max_depth (int): Maximum depth of individual trees. Default: 5
            random_state (int): Random seed for reproducibility. Default: 42
            subsample (float): Fraction of samples for fitting individual base learners. Default: 0.8
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            subsample=subsample,
            verbose=0
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.classes_ = None
    
    def train(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        outcomes: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        validate: bool = True
    ) -> Dict[str, float]:
        """
        Train the prediction model on basketball features and outcomes.
        
        Args:
            features (Union[pd.DataFrame, np.ndarray]): Training features
                Shape: (n_samples, n_features)
            outcomes (Union[pd.Series, np.ndarray]): Match outcomes (binary: 0 or 1)
                Shape: (n_samples,)
            test_size (float): Proportion of data to use for testing. Default: 0.2
            validate (bool): Whether to perform train-test split validation. Default: True
        
        Returns:
            Dict[str, float]: Training metrics including accuracy and other performance metrics
                - 'train_accuracy': Accuracy on training set
                - 'test_accuracy': Accuracy on test set (if validate=True)
        
        Raises:
            ValueError: If features and outcomes have mismatched lengths
        """
        # Convert to numpy arrays if needed
        if isinstance(features, pd.DataFrame):
            self.feature_names = list(features.columns)
            X = features.values
        else:
            X = np.array(features)
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        y = np.array(outcomes)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Features and outcomes must have same number of samples. "
                f"Got {X.shape[0]} features and {y.shape[0]} outcomes."
            )
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        metrics = {}
        
        if validate:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=self.model.random_state
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Calculate metrics
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            metrics['train_accuracy'] = train_accuracy
            metrics['test_accuracy'] = test_accuracy
        else:
            # Train on entire dataset
            self.model.fit(X_scaled, y)
            train_accuracy = self.model.score(X_scaled, y)
            metrics['train_accuracy'] = train_accuracy
        
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        return metrics
    
    def predict(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions on new basketball match data.
        
        Args:
            features (Union[pd.DataFrame, np.ndarray]): Features for prediction
                Shape: (n_samples, n_features)
        
        Returns:
            np.ndarray: Predicted outcomes (0 or 1)
                Shape: (n_samples,)
        
        Raises:
            RuntimeError: If model has not been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions. Call train() first.")
        
        # Convert to numpy array if needed
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = np.array(features)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Predict probability scores for match outcomes.
        
        Args:
            features (Union[pd.DataFrame, np.ndarray]): Features for prediction
                Shape: (n_samples, n_features)
        
        Returns:
            Dict[str, np.ndarray]: Probability scores for each outcome class
                - 'probabilities': Array of shape (n_samples, 2) with probabilities for each class
                - 'class_labels': Array with class labels [0, 1]
                - 'home_win_prob': Probability that home team wins (class 1)
                - 'away_win_prob': Probability that away team wins (class 0)
        
        Raises:
            RuntimeError: If model has not been trained yet
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before making predictions. Call train() first."
            )
        
        # Convert to numpy array if needed
        if isinstance(features, pd.DataFrame):
            X = features.values
        else:
            X = np.array(features)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X_scaled)
        
        return {
            'probabilities': probabilities,
            'class_labels': self.model.classes_,
            'away_win_prob': probabilities[:, 0],  # Probability of class 0
            'home_win_prob': probabilities[:, 1]   # Probability of class 1
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Dict[str, float]: Feature names mapped to their importance scores
        
        Raises:
            RuntimeError: If model has not been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before getting feature importance.")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is None:
            feature_dict = {f"feature_{i}": imp for i, imp in enumerate(importances)}
        else:
            feature_dict = {name: imp for name, imp in zip(self.feature_names, importances)}
        
        # Sort by importance descending
        return dict(sorted(feature_dict.items(), key=lambda x: x[1], reverse=True))
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model.
        
        Returns:
            Dict: Model information including parameters and training status
        """
        return {
            'model_type': 'GradientBoostingClassifier',
            'is_trained': self.is_trained,
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'classes': list(self.classes_) if self.classes_ is not None else None
        }
    
    def batch_predict_with_probabilities(
        self,
        features: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Make batch predictions with probability scores for multiple matches.
        
        Args:
            features (Union[pd.DataFrame, np.ndarray]): Features for prediction
                Shape: (n_samples, n_features)
        
        Returns:
            pd.DataFrame: DataFrame with predictions and probability scores
                Columns: 'prediction', 'away_win_prob', 'home_win_prob', 'confidence'
        
        Raises:
            RuntimeError: If model has not been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions.")
        
        predictions = self.predict(features)
        proba_dict = self.predict_proba(features)
        
        # Calculate confidence as the maximum probability
        max_probs = np.max(proba_dict['probabilities'], axis=1)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'away_win_prob': proba_dict['away_win_prob'],
            'home_win_prob': proba_dict['home_win_prob'],
            'confidence': max_probs
        })
        
        return results
