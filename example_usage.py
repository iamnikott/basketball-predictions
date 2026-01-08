"""
Comprehensive usage examples for the Basketball Predictions library.

This module demonstrates how to use the basketball-predictions package
for making predictions, analyzing team statistics, and evaluating model performance.
"""

import logging
from datetime import datetime, timedelta
from basketball_predictions import (
    BasketballPredictor,
    DataLoader,
    ModelEvaluator,
    TeamAnalyzer,
    PredictionResult,
)

# Configure logging for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Basic Model Initialization and Configuration
# ============================================================================

def example_basic_initialization():
    """Demonstrate basic model initialization with default settings."""
    logger.info("Example 1: Basic Model Initialization")
    
    # Initialize predictor with default configuration
    predictor = BasketballPredictor()
    
    # Display model information
    print(f"Model Type: {predictor.model_type}")
    print(f"Version: {predictor.version}")
    print(f"Features Used: {predictor.feature_count}")
    
    return predictor


# ============================================================================
# EXAMPLE 2: Custom Configuration and Model Parameters
# ============================================================================

def example_custom_configuration():
    """Demonstrate custom configuration for specific use cases."""
    logger.info("Example 2: Custom Configuration")
    
    # Create predictor with custom parameters
    config = {
        'model_type': 'ensemble',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'feature_engineering': True,
        'use_historical_trends': True,
        'min_games_required': 5,
    }
    
    predictor = BasketballPredictor(**config)
    
    print(f"Custom Predictor Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return predictor


# ============================================================================
# EXAMPLE 3: Loading and Processing Data
# ============================================================================

def example_data_loading_and_processing():
    """Demonstrate data loading and preprocessing."""
    logger.info("Example 3: Data Loading and Processing")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data from various sources
    season_data = data_loader.load_season_data(season=2024)
    print(f"Loaded {len(season_data)} games for 2024 season")
    
    # Load team statistics
    team_stats = data_loader.load_team_statistics()
    print(f"Loaded statistics for {len(team_stats)} teams")
    
    # Load historical performance data
    historical_data = data_loader.load_historical_data(years=5)
    print(f"Loaded {len(historical_data)} historical records")
    
    # Process and clean data
    processed_data = data_loader.preprocess(season_data)
    print(f"After preprocessing: {len(processed_data)} valid game records")
    
    return season_data, team_stats, historical_data


# ============================================================================
# EXAMPLE 4: Team Analysis and Statistics
# ============================================================================

def example_team_analysis():
    """Demonstrate team analysis and statistics extraction."""
    logger.info("Example 4: Team Analysis and Statistics")
    
    analyzer = TeamAnalyzer()
    
    # Analyze specific team
    team_name = "Los Angeles Lakers"
    team_analysis = analyzer.analyze_team(team_name, season=2024)
    
    print(f"\nTeam Analysis for {team_name}:")
    print(f"  Win-Loss Record: {team_analysis['wins']}-{team_analysis['losses']}")
    print(f"  Points Per Game: {team_analysis['ppg']:.1f}")
    print(f"  Points Allowed Per Game: {team_analysis['papg']:.1f}")
    print(f"  Field Goal Percentage: {team_analysis['fg_percentage']:.1f}%")
    print(f"  Three Point Percentage: {team_analysis['three_p_percentage']:.1f}%")
    print(f"  Rebounds Per Game: {team_analysis['rpg']:.1f}")
    print(f"  Assists Per Game: {team_analysis['apg']:.1f}")
    
    # Compare teams
    team2_name = "Boston Celtics"
    comparison = analyzer.compare_teams(team_name, team2_name)
    print(f"\nComparison: {team_name} vs {team2_name}")
    print(f"  Offensive Advantage: {comparison['offensive_advantage']:.2f}%")
    print(f"  Defensive Advantage: {comparison['defensive_advantage']:.2f}%")
    
    # Get team trends
    trends = analyzer.get_team_trends(team_name, games=10)
    print(f"\nLast 10 Games Trend for {team_name}:")
    print(f"  Average Points Scored: {trends['avg_points']:.1f}")
    print(f"  Average Points Allowed: {trends['avg_points_allowed']:.1f}")
    print(f"  Win Percentage: {trends['win_percentage']:.1f}%")
    
    return team_analysis, comparison, trends


# ============================================================================
# EXAMPLE 5: Making Predictions on Individual Games
# ============================================================================

def example_single_game_prediction():
    """Demonstrate making predictions for a single game."""
    logger.info("Example 5: Single Game Prediction")
    
    predictor = BasketballPredictor()
    
    # Prediction for a specific matchup
    game_info = {
        'home_team': 'Los Angeles Lakers',
        'away_team': 'Golden State Warriors',
        'date': datetime.now().date(),
        'season': 2024,
    }
    
    # Make prediction
    prediction = predictor.predict_game(**game_info)
    
    print(f"\nPrediction: {game_info['home_team']} vs {game_info['away_team']}")
    print(f"  Predicted Winner: {prediction.predicted_winner}")
    print(f"  Home Team Win Probability: {prediction.home_win_probability:.2%}")
    print(f"  Away Team Win Probability: {prediction.away_win_probability:.2%}")
    print(f"  Predicted Point Spread: {prediction.predicted_spread:.1f}")
    print(f"  Predicted Total Score: {prediction.predicted_total:.1f}")
    print(f"  Confidence Level: {prediction.confidence:.2%}")
    print(f"  Key Factors: {', '.join(prediction.key_factors)}")
    
    return prediction


# ============================================================================
# EXAMPLE 6: Batch Predictions
# ============================================================================

def example_batch_predictions():
    """Demonstrate making predictions for multiple games."""
    logger.info("Example 6: Batch Predictions")
    
    predictor = BasketballPredictor()
    
    # Define multiple games
    games = [
        {
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Golden State Warriors',
            'date': datetime.now().date(),
        },
        {
            'home_team': 'Boston Celtics',
            'away_team': 'Miami Heat',
            'date': datetime.now().date(),
        },
        {
            'home_team': 'Denver Nuggets',
            'away_team': 'Phoenix Suns',
            'date': (datetime.now() + timedelta(days=1)).date(),
        },
    ]
    
    # Make batch predictions
    predictions = predictor.predict_batch(games)
    
    print(f"\nBatch Predictions ({len(predictions)} games):")
    for i, pred in enumerate(predictions, 1):
        print(f"\n  Game {i}: {pred.home_team} vs {pred.away_team}")
        print(f"    Winner: {pred.predicted_winner}")
        print(f"    Spread: {pred.predicted_spread:.1f}")
        print(f"    Confidence: {pred.confidence:.2%}")
    
    return predictions


# ============================================================================
# EXAMPLE 7: Season Predictions and Projections
# ============================================================================

def example_season_predictions():
    """Demonstrate season-long predictions and projections."""
    logger.info("Example 7: Season Predictions")
    
    predictor = BasketballPredictor()
    
    # Get season standings predictions
    standings = predictor.predict_season_standings(season=2024)
    
    print(f"\nPredicted Season Standings (Top 10):")
    for i, team_projection in enumerate(standings[:10], 1):
        print(f"  {i}. {team_projection['team']}: {team_projection['wins']}-{team_projection['losses']}")
        print(f"     Make Playoffs: {team_projection['playoff_probability']:.1%}")
        print(f"     Championship: {team_projection['championship_probability']:.2%}")
    
    return standings


# ============================================================================
# EXAMPLE 8: Model Evaluation and Validation
# ============================================================================

def example_model_evaluation():
    """Demonstrate model evaluation and performance metrics."""
    logger.info("Example 8: Model Evaluation")
    
    evaluator = ModelEvaluator()
    
    # Evaluate model on recent games
    evaluation_results = evaluator.evaluate_recent_games(
        model=BasketballPredictor(),
        games=100,
        season=2024
    )
    
    print(f"\nModel Evaluation Metrics:")
    print(f"  Accuracy: {evaluation_results['accuracy']:.2%}")
    print(f"  Precision: {evaluation_results['precision']:.2%}")
    print(f"  Recall: {evaluation_results['recall']:.2%}")
    print(f"  F1 Score: {evaluation_results['f1_score']:.4f}")
    print(f"  AUC-ROC: {evaluation_results['auc_roc']:.4f}")
    print(f"  Mean Absolute Error (Spread): {evaluation_results['mae_spread']:.2f}")
    
    # Cross-validation results
    cv_results = evaluator.cross_validate(
        model=BasketballPredictor(),
        folds=5,
        season=2024
    )
    
    print(f"\n5-Fold Cross-Validation Results:")
    print(f"  Mean Accuracy: {cv_results['mean_accuracy']:.2%}")
    print(f"  Std Accuracy: {cv_results['std_accuracy']:.2%}")
    
    return evaluation_results, cv_results


# ============================================================================
# EXAMPLE 9: Feature Importance and Model Interpretation
# ============================================================================

def example_feature_importance():
    """Demonstrate feature importance analysis."""
    logger.info("Example 9: Feature Importance")
    
    predictor = BasketballPredictor()
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    
    print(f"\nTop 15 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance[:15], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Get SHAP values for explainability
    game_info = {
        'home_team': 'Los Angeles Lakers',
        'away_team': 'Golden State Warriors',
    }
    
    shap_values = predictor.explain_prediction(game_info)
    
    print(f"\nTop Contributing Factors to Prediction:")
    for factor, contribution in shap_values['top_factors'].items():
        print(f"  {factor}: {contribution:+.4f}")
    
    return feature_importance, shap_values


# ============================================================================
# EXAMPLE 10: Error Handling and Edge Cases
# ============================================================================

def example_error_handling():
    """Demonstrate proper error handling."""
    logger.info("Example 10: Error Handling")
    
    predictor = BasketballPredictor()
    
    # Handle invalid game
    try:
        invalid_game = {
            'home_team': 'Invalid Team',
            'away_team': 'Another Invalid Team',
        }
        prediction = predictor.predict_game(**invalid_game)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Handle missing data
    try:
        incomplete_game = {
            'home_team': 'Los Angeles Lakers',
            # Missing away_team
        }
        prediction = predictor.predict_game(**incomplete_game)
    except KeyError as e:
        print(f"Caught expected error: Missing required field {e}")
    
    # Handle data loading errors
    try:
        data_loader = DataLoader()
        future_season_data = data_loader.load_season_data(season=2099)
    except ValueError as e:
        print(f"Caught expected error: {e}")


# ============================================================================
# EXAMPLE 11: Advanced Configuration with Weighted Ensembles
# ============================================================================

def example_weighted_ensemble():
    """Demonstrate weighted ensemble predictions."""
    logger.info("Example 11: Weighted Ensemble Predictions")
    
    # Create multiple models with different configurations
    models = [
        BasketballPredictor(model_type='gradient_boosting', weight=0.4),
        BasketballPredictor(model_type='random_forest', weight=0.3),
        BasketballPredictor(model_type='neural_network', weight=0.3),
    ]
    
    game_info = {
        'home_team': 'Los Angeles Lakers',
        'away_team': 'Golden State Warriors',
    }
    
    # Get predictions from all models
    print(f"\nEnsemble Predictions:")
    print(f"Individual Model Predictions:")
    predictions = []
    for model in models:
        pred = model.predict_game(**game_info)
        predictions.append(pred)
        print(f"  {model.model_type}: {pred.predicted_winner} ({pred.home_win_probability:.2%})")
    
    # Combine predictions using weights
    combined_prediction = predictor._combine_predictions(predictions, models)
    print(f"\nWeighted Ensemble Prediction:")
    print(f"  Winner: {combined_prediction.predicted_winner}")
    print(f"  Probability: {combined_prediction.home_win_probability:.2%}")
    print(f"  Confidence: {combined_prediction.confidence:.2%}")


# ============================================================================
# EXAMPLE 12: Real-time Predictions with Streaming Data
# ============================================================================

def example_streaming_predictions():
    """Demonstrate real-time predictions with live data."""
    logger.info("Example 12: Streaming Predictions")
    
    predictor = BasketballPredictor(use_streaming_data=True)
    
    # Simulate streaming game updates
    game_id = "2024_001"
    updates = [
        {'quarter': 1, 'time_remaining': 600, 'home_score': 25, 'away_score': 22},
        {'quarter': 2, 'time_remaining': 1200, 'home_score': 45, 'away_score': 42},
        {'quarter': 3, 'time_remaining': 1800, 'home_score': 65, 'away_score': 62},
        {'quarter': 4, 'time_remaining': 600, 'home_score': 85, 'away_score': 82},
    ]
    
    print(f"\nReal-time Win Probability Updates:")
    for update in updates:
        pred = predictor.predict_live_game(game_id, update)
        print(f"  Q{update['quarter']} - {update['time_remaining']//60}:00 remaining")
        print(f"    Score: {update['home_score']}-{update['away_score']}")
        print(f"    Home Win Probability: {pred.home_win_probability:.2%}")


# ============================================================================
# EXAMPLE 13: Custom Metrics and KPI Tracking
# ============================================================================

def example_custom_metrics():
    """Demonstrate custom metrics and KPI tracking."""
    logger.info("Example 13: Custom Metrics and KPI Tracking")
    
    predictor = BasketballPredictor()
    evaluator = ModelEvaluator()
    
    # Track predictions over time
    tracking_period = 30  # days
    kpis = evaluator.track_kpis(
        model=predictor,
        days=tracking_period,
        metrics=[
            'accuracy',
            'winning_percentage',
            'average_confidence',
            'roi',
        ]
    )
    
    print(f"\nKPI Report (Last {tracking_period} days):")
    print(f"  Prediction Accuracy: {kpis['accuracy']:.2%}")
    print(f"  Winning Percentage: {kpis['winning_percentage']:.2%}")
    print(f"  Average Confidence: {kpis['average_confidence']:.2%}")
    print(f"  Return on Investment: {kpis['roi']:.2%}")


# ============================================================================
# EXAMPLE 14: Exporting Results and Reports
# ============================================================================

def example_export_results():
    """Demonstrate exporting results and generating reports."""
    logger.info("Example 14: Exporting Results")
    
    predictor = BasketballPredictor()
    
    # Make predictions
    games = [
        {
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Golden State Warriors',
            'date': datetime.now().date(),
        },
        {
            'home_team': 'Boston Celtics',
            'away_team': 'Miami Heat',
            'date': datetime.now().date(),
        },
    ]
    
    predictions = predictor.predict_batch(games)
    
    # Export to CSV
    csv_path = predictor.export_predictions(predictions, format='csv')
    print(f"Exported predictions to CSV: {csv_path}")
    
    # Export to JSON
    json_path = predictor.export_predictions(predictions, format='json')
    print(f"Exported predictions to JSON: {json_path}")
    
    # Generate PDF report
    report_path = predictor.generate_report(predictions, format='pdf')
    print(f"Generated PDF report: {report_path}")


# ============================================================================
# EXAMPLE 15: Complete End-to-End Pipeline
# ============================================================================

def example_complete_pipeline():
    """Demonstrate complete end-to-end prediction pipeline."""
    logger.info("Example 15: Complete End-to-End Pipeline")
    
    print("\n" + "="*70)
    print("COMPLETE END-TO-END PIPELINE")
    print("="*70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    data_loader = DataLoader()
    season_data = data_loader.load_season_data(season=2024)
    print(f"  ✓ Loaded {len(season_data)} game records")
    
    # Step 2: Analyze teams
    print("\n[Step 2] Analyzing teams...")
    analyzer = TeamAnalyzer()
    team_analysis = analyzer.analyze_team('Los Angeles Lakers', season=2024)
    print(f"  ✓ Analyzed team statistics")
    
    # Step 3: Initialize model
    print("\n[Step 3] Initializing prediction model...")
    predictor = BasketballPredictor(
        model_type='ensemble',
        use_historical_trends=True,
        feature_engineering=True,
    )
    print(f"  ✓ Model initialized with {predictor.feature_count} features")
    
    # Step 4: Make predictions
    print("\n[Step 4] Making game predictions...")
    games = [
        {'home_team': 'Los Angeles Lakers', 'away_team': 'Golden State Warriors'},
        {'home_team': 'Boston Celtics', 'away_team': 'Miami Heat'},
    ]
    predictions = predictor.predict_batch(games)
    print(f"  ✓ Generated {len(predictions)} predictions")
    
    # Step 5: Evaluate model
    print("\n[Step 5] Evaluating model performance...")
    evaluator = ModelEvaluator()
    evaluation = evaluator.evaluate_recent_games(
        model=predictor,
        games=100,
        season=2024
    )
    print(f"  ✓ Model Accuracy: {evaluation['accuracy']:.2%}")
    
    # Step 6: Export results
    print("\n[Step 6] Exporting results...")
    csv_path = predictor.export_predictions(predictions, format='csv')
    print(f"  ✓ Results exported to {csv_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("BASKETBALL PREDICTIONS - COMPREHENSIVE USAGE EXAMPLES")
    print("="*70)
    
    try:
        # Run examples
        example_basic_initialization()
        print("\n" + "-"*70)
        
        example_custom_configuration()
        print("\n" + "-"*70)
        
        example_data_loading_and_processing()
        print("\n" + "-"*70)
        
        example_team_analysis()
        print("\n" + "-"*70)
        
        example_single_game_prediction()
        print("\n" + "-"*70)
        
        example_batch_predictions()
        print("\n" + "-"*70)
        
        example_season_predictions()
        print("\n" + "-"*70)
        
        example_model_evaluation()
        print("\n" + "-"*70)
        
        example_feature_importance()
        print("\n" + "-"*70)
        
        example_error_handling()
        print("\n" + "-"*70)
        
        example_weighted_ensemble()
        print("\n" + "-"*70)
        
        example_streaming_predictions()
        print("\n" + "-"*70)
        
        example_custom_metrics()
        print("\n" + "-"*70)
        
        example_export_results()
        print("\n" + "-"*70)
        
        example_complete_pipeline()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == '__main__':
    main()
