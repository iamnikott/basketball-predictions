# Basketball Predictions System

A comprehensive machine learning system designed to predict basketball game outcomes and player performance metrics.

## Overview

This project implements advanced predictive models for basketball analytics, leveraging historical game data, team statistics, and player performance metrics to generate accurate predictions for NBA games.

## Features

- **Game Outcome Prediction**: Predict winner, point spread, and over/under totals
- **Player Performance Analytics**: Forecast individual player statistics (points, rebounds, assists, etc.)
- **Team Analytics**: Analyze team strengths, weaknesses, and matchup advantages
- **Historical Data Processing**: Efficiently handle and process large volumes of historical basketball data
- **Model Evaluation**: Comprehensive metrics to assess prediction accuracy
- **API Integration**: Easy-to-use interface for generating predictions

## Tech Stack

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn, TensorFlow/PyTorch
- **Data Processing**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Database**: SQLite/PostgreSQL
- **API**: Flask/FastAPI

## Project Structure

```
basketball-predictions/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/                    # Raw basketball data
│   ├── processed/              # Cleaned and processed data
│   └── models/                 # Trained model files
├── src/
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── feature_engineering.py # Feature extraction and engineering
│   ├── models/                # Model definitions
│   │   ├── game_predictor.py
│   │   └── player_predictor.py
│   ├── evaluation.py          # Model evaluation metrics
│   └── utils.py               # Utility functions
├── notebooks/                  # Jupyter notebooks for analysis
├── tests/                      # Unit tests
└── api/                        # API endpoints
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/iamnikott/basketball-predictions.git
   cd basketball-predictions
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare data**
   ```bash
   python src/data_loader.py
   ```

## Usage

### Training a Model

```python
from src.models.game_predictor import GamePredictor
from src.data_loader import load_data

# Load training data
X_train, y_train = load_data('train')

# Initialize and train model
predictor = GamePredictor()
predictor.train(X_train, y_train)

# Save model
predictor.save('data/models/game_model.pkl')
```

### Making Predictions

```python
from src.models.game_predictor import GamePredictor

# Load model
predictor = GamePredictor.load('data/models/game_model.pkl')

# Make predictions
predictions = predictor.predict(X_test)
```

### API Usage

```bash
# Start the API server
python api/app.py

# Example prediction request
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "LAL",
    "away_team": "GSW",
    "home_rest_days": 2,
    "away_rest_days": 1
  }'
```

## Data Sources

- NBA Official Statistics (NBA.com)
- Basketball Reference (Basketball-Reference.com)
- ESPN Sports Data
- Historical game logs and box scores

## Model Performance

Current model accuracy and metrics:

- **Game Winner Prediction**: ~67% accuracy
- **Point Spread Prediction**: MAE of ±4.2 points
- **Over/Under Prediction**: ~58% accuracy

*Metrics are based on the latest validation set. See `/notebooks/model_evaluation.ipynb` for detailed analysis.*

## Features Used

### Game-Level Features
- Team offensive/defensive ratings
- Recent form (last 10 games)
- Home/away splits
- Rest days between games
- Head-to-head history
- Injury report data
- Strength of schedule

### Player-Level Features
- Historical performance averages
- Shooting percentages
- Usage rates
- Minutes played trends
- Opponent matchup history

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Regression**: MAE, RMSE, R² Score
- **Custom**: Betting ROI, Win Rate Against Spread

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

Run the test suite:

```bash
pytest tests/
```

For coverage reports:

```bash
pytest --cov=src tests/
```

## Configuration

Configuration settings can be found in `config.yaml`:

```yaml
data:
  source: 'nba_official'
  update_frequency: 'daily'
  
models:
  game_predictor:
    model_type: 'ensemble'
    hyperparameters:
      learning_rate: 0.01
      max_depth: 10
      
api:
  port: 5000
  debug: false
```

## Performance Optimization

- **Caching**: Predictions are cached for 24 hours
- **Batch Processing**: Handle large prediction requests efficiently
- **Model Compression**: Use quantization for faster inference
- **Parallel Processing**: Multi-threading for data preprocessing

## Limitations

- Predictions are statistical estimates and not guaranteed
- Model performance may vary during unusual seasons or rule changes
- Injuries and roster changes may not be immediately reflected
- External factors (refs, home crowd) are difficult to quantify

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NBA and Basketball Reference for data sources
- The open-source machine learning community
- Contributors and testers

## Contact

For questions or suggestions, please contact:
- **GitHub**: [@iamnikott](https://github.com/iamnikott)
- **Email**: (Add your contact email if desired)

## Roadmap

- [ ] Add real-time game updates
- [ ] Implement deep learning models
- [ ] Expand to international basketball leagues
- [ ] Create mobile app for predictions
- [ ] Add player injury prediction
- [ ] Integrate live betting odds
- [ ] Implement explainable AI features

---

Last Updated: 2026-01-08
