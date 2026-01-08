import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


class BasketballPredictor:
    """
    A machine learning-based basketball match prediction system.
    
    This class handles loading match data, training predictive models,
    making predictions for team matchups, and providing comprehensive
    statistics and team performance analysis.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the BasketballPredictor.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = None
        self.feature_columns = None
        self.team_stats = {}
        self.model_metrics = {}
        
    def load_data(self, filepath):
        """
        Load match data from a CSV file.
        
        Expected columns: home_team, away_team, home_score, away_score,
        and any relevant statistics (points, rebounds, assists, etc.)
        
        Args:
            filepath (str): Path to the CSV file containing match data
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            self.data = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully: {len(self.data)} matches")
            print(f"  Columns: {', '.join(self.data.columns.tolist())}")
            return True
        except FileNotFoundError:
            print(f"✗ Error: File '{filepath}' not found")
            return False
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self, target_column='home_win'):
        """
        Preprocess the data for model training.
        
        Creates target variable (1 if home team wins, 0 otherwise) and
        identifies feature columns for training.
        
        Args:
            target_column (str): Name of the target column
            
        Returns:
            bool: True if preprocessing successful, False otherwise
        """
        try:
            if self.data is None:
                print("✗ Error: No data loaded. Call load_data() first.")
                return False
            
            # Create target variable if it doesn't exist
            if target_column not in self.data.columns:
                if 'home_score' in self.data.columns and 'away_score' in self.data.columns:
                    self.data[target_column] = (
                        self.data['home_score'] > self.data['away_score']
                    ).astype(int)
                else:
                    print("✗ Error: Cannot create target variable. Missing score columns.")
                    return False
            
            # Identify feature columns (exclude non-numeric and target columns)
            exclude_cols = {
                'home_team', 'away_team', target_column, 
                'home_score', 'away_score', 'date', 'game_id'
            }
            self.feature_columns = [
                col for col in self.data.columns 
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.data[col])
            ]
            
            if not self.feature_columns:
                print("✗ Error: No numeric features found for training")
                return False
            
            print(f"✓ Data preprocessed successfully")
            print(f"  Features ({len(self.feature_columns)}): {', '.join(self.feature_columns[:5])}{'...' if len(self.feature_columns) > 5 else ''}")
            return True
            
        except Exception as e:
            print(f"✗ Error during preprocessing: {str(e)}")
            return False
    
    def train_model(self, test_size=0.2, n_estimators=100):
        """
        Train a Random Forest model for match prediction.
        
        Args:
            test_size (float): Proportion of data to use for testing (0.0-1.0)
            n_estimators (int): Number of trees in the random forest
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            if self.data is None or self.feature_columns is None:
                print("✗ Error: Data not loaded or preprocessed. Call load_data() and preprocess_data() first.")
                return False
            
            # Prepare features and target
            X = self.data[self.feature_columns].fillna(0)
            y = self.data['home_win']
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                max_depth=15,
                min_samples_split=5
            )
            self.model.fit(self.X_train, self.y_train)
            
            # Calculate metrics
            train_pred = self.model.predict(self.X_train)
            test_pred = self.model.predict(self.X_test)
            
            self.model_metrics = {
                'train_accuracy': accuracy_score(self.y_train, train_pred),
                'test_accuracy': accuracy_score(self.y_test, test_pred),
                'precision': precision_score(self.y_test, test_pred, zero_division=0),
                'recall': recall_score(self.y_test, test_pred, zero_division=0),
                'f1_score': f1_score(self.y_test, test_pred, zero_division=0)
            }
            
            print(f"✓ Model trained successfully")
            print(f"  Training samples: {len(self.X_train)} | Test samples: {len(self.X_test)}")
            return True
            
        except Exception as e:
            print(f"✗ Error during training: {str(e)}")
            return False
    
    def predict_matchup(self, home_team, away_team):
        """
        Make a prediction for a specific team matchup.
        
        Args:
            home_team (str): Name of the home team
            away_team (str): Name of the away team
            
        Returns:
            dict: Prediction details including win probability and confidence
        """
        try:
            if self.model is None:
                print("✗ Error: Model not trained. Call train_model() first.")
                return None
            
            # Get team statistics from training data
            home_stats = self.data[self.data['home_team'] == home_team][self.feature_columns].mean()
            away_stats = self.data[self.data['away_team'] == away_team][self.feature_columns].mean()
            
            if home_stats.empty or away_stats.empty:
                print(f"✗ Error: Team data not found. Available teams: {self.data['home_team'].unique().tolist()[:5]}...")
                return None
            
            # Create feature vector
            matchup_features = home_stats.values.reshape(1, -1)
            matchup_features = self.scaler.transform(matchup_features)
            
            # Make prediction
            prediction = self.model.predict(matchup_features)[0]
            probability = self.model.predict_proba(matchup_features)[0]
            
            result = {
                'home_team': home_team,
                'away_team': away_team,
                'prediction': 'Home Win' if prediction == 1 else 'Away Win',
                'home_win_probability': probability[1],
                'away_win_probability': probability[0],
                'confidence': max(probability) * 100
            }
            
            return result
            
        except Exception as e:
            print(f"✗ Error making prediction: {str(e)}")
            return None
    
    def get_team_performance(self, team_name):
        """
        Get detailed performance statistics for a specific team.
        
        Args:
            team_name (str): Name of the team
            
        Returns:
            dict: Team performance statistics (wins, losses, average stats)
        """
        try:
            if self.data is None:
                print("✗ Error: No data loaded")
                return None
            
            # Home games
            home_games = self.data[self.data['home_team'] == team_name]
            home_wins = (home_games['home_score'] > home_games['away_score']).sum()
            home_losses = len(home_games) - home_wins
            
            # Away games
            away_games = self.data[self.data['away_team'] == team_name]
            away_wins = (away_games['away_score'] > away_games['home_score']).sum()
            away_losses = len(away_games) - away_wins
            
            # Total stats
            total_wins = home_wins + away_wins
            total_losses = home_losses + away_losses
            total_games = total_wins + total_losses
            win_percentage = (total_wins / total_games * 100) if total_games > 0 else 0
            
            # Average statistics
            all_games = pd.concat([home_games, away_games])
            avg_stats = {}
            for col in self.feature_columns:
                if col in all_games.columns:
                    avg_stats[col] = all_games[col].mean()
            
            performance = {
                'team_name': team_name,
                'home_record': f"{home_wins}-{home_losses}",
                'away_record': f"{away_wins}-{away_losses}",
                'overall_record': f"{total_wins}-{total_losses}",
                'win_percentage': round(win_percentage, 2),
                'total_games': total_games,
                'average_stats': avg_stats
            }
            
            return performance
            
        except Exception as e:
            print(f"✗ Error getting team performance: {str(e)}")
            return None
    
    def display_model_statistics(self):
        """Display comprehensive model performance statistics."""
        if not self.model_metrics:
            print("✗ Error: Model not trained yet")
            return
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Training Accuracy:   {self.model_metrics['train_accuracy']:.4f} ({self.model_metrics['train_accuracy']*100:.2f}%)")
        print(f"Testing Accuracy:    {self.model_metrics['test_accuracy']:.4f} ({self.model_metrics['test_accuracy']*100:.2f}%)")
        print(f"Precision:           {self.model_metrics['precision']:.4f}")
        print(f"Recall:              {self.model_metrics['recall']:.4f}")
        print(f"F1-Score:            {self.model_metrics['f1_score']:.4f}")
        print("="*60 + "\n")
    
    def display_prediction(self, prediction):
        """Display a prediction result in a formatted way."""
        if prediction is None:
            return
        
        print("\n" + "="*60)
        print("MATCH PREDICTION")
        print("="*60)
        print(f"Home Team:           {prediction['home_team']}")
        print(f"Away Team:           {prediction['away_team']}")
        print(f"Predicted Winner:    {prediction['prediction']}")
        print(f"Home Win Probability: {prediction['home_win_probability']*100:.2f}%")
        print(f"Away Win Probability: {prediction['away_win_probability']*100:.2f}%")
        print(f"Confidence:          {prediction['confidence']:.2f}%")
        print("="*60 + "\n")
    
    def display_team_analysis(self, team_name):
        """Display detailed team performance analysis."""
        performance = self.get_team_performance(team_name)
        
        if performance is None:
            return
        
        print("\n" + "="*60)
        print(f"TEAM PERFORMANCE ANALYSIS: {team_name}")
        print("="*60)
        print(f"Overall Record:      {performance['overall_record']}")
        print(f"Home Record:         {performance['home_record']}")
        print(f"Away Record:         {performance['away_record']}")
        print(f"Win Percentage:      {performance['win_percentage']:.2f}%")
        print(f"Total Games:         {performance['total_games']}")
        
        if performance['average_stats']:
            print("\nAverage Statistics:")
            for stat, value in performance['average_stats'].items():
                print(f"  {stat:.<30} {value:>10.2f}")
        
        print("="*60 + "\n")
    
    def get_feature_importance(self, top_n=10):
        """
        Get the most important features for predictions.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            dict: Dictionary of feature names and their importance scores
        """
        if self.model is None:
            print("✗ Error: Model not trained yet")
            return {}
        
        importance_dict = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        sorted_importance = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n])
        
        return sorted_importance
    
    def display_feature_importance(self, top_n=10):
        """Display the most important features for the model."""
        importance = self.get_feature_importance(top_n)
        
        if not importance:
            return
        
        print("\n" + "="*60)
        print(f"TOP {top_n} FEATURE IMPORTANCE")
        print("="*60)
        for i, (feature, score) in enumerate(importance.items(), 1):
            bar_length = int(score * 50)
            bar = "█" * bar_length
            print(f"{i:2d}. {feature:.<30} {score:>8.4f} {bar}")
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("BASKETBALL PREDICTIONS SYSTEM")
    print("="*60 + "\n")
    
    # Initialize predictor
    predictor = BasketballPredictor(random_state=42)
    
    # Load data (replace 'matches.csv' with your actual data file)
    # if predictor.load_data('matches.csv'):
    #     predictor.preprocess_data()
    #     predictor.train_model(test_size=0.2, n_estimators=100)
    #     predictor.display_model_statistics()
    #     predictor.display_feature_importance(top_n=10)
    #     
    #     # Make predictions
    #     prediction = predictor.predict_matchup('Team A', 'Team B')
    #     if prediction:
    #         predictor.display_prediction(prediction)
    #     
    #     # Get team analysis
    #     predictor.display_team_analysis('Team A')
    
    print("Ready to analyze basketball matches!")
    print("\nUsage Example:")
    print("  1. predictor.load_data('matches.csv')")
    print("  2. predictor.preprocess_data()")
    print("  3. predictor.train_model()")
    print("  4. prediction = predictor.predict_matchup('Team A', 'Team B')")
    print("  5. predictor.display_team_analysis('Team A')")
