import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """
    A comprehensive data loader for basketball match history from 2021-2025.
    
    Handles loading, generating, and processing basketball match data with
    features for machine learning predictions.
    """
    
    def __init__(self, start_year: int = 2021, end_year: int = 2025, seed: int = 42):
        """
        Initialize the DataLoader.
        
        Args:
            start_year (int): Start year for historical data (default: 2021)
            end_year (int): End year for historical data (default: 2025)
            seed (int): Random seed for reproducibility (default: 42)
        """
        self.start_year = start_year
        self.end_year = end_year
        self.seed = seed
        np.random.seed(seed)
        
        self.matches_df = None
        self.teams = None
        self.team_stats = None
        
        # Sample NBA teams
        self.team_names = [
            'Lakers', 'Celtics', 'Warriors', 'Heat', 'Suns',
            'Mavericks', 'Nuggets', 'Grizzlies', 'Bucks', 'Nets',
            'Trail Blazers', 'Kings', 'Spurs', 'Rockets', 'Hornets',
            'Pacers', 'Pistons', 'Raptors', 'Cavaliers', 'Clippers',
            'Pelicans', 'Timberwolves', 'Thunder', 'Knicks', 'Hawks',
            '76ers', 'Bulls', 'Wizards', 'Jazz', 'Cavaliers'
        ]
        
    def generate_sample_data(self, matches_per_team_per_year: int = 50) -> pd.DataFrame:
        """
        Generate sample basketball match history data.
        
        Args:
            matches_per_team_per_year (int): Number of matches per team per year
            
        Returns:
            pd.DataFrame: DataFrame containing match history with columns:
                - date, home_team, away_team, home_score, away_score,
                  home_fg%, away_fg%, home_3p%, away_3p%, home_rebounds,
                  away_rebounds, home_assists, away_assists, home_turnovers,
                  away_turnovers, winner
        """
        matches_list = []
        
        for year in range(self.start_year, self.end_year + 1):
            year_start = datetime(year, 10, 1)  # NBA season typically starts in October
            year_end = datetime(year + 1, 6, 30)  # Regular season ends around June
            
            # Generate matches throughout the season
            num_days = (year_end - year_start).days
            
            for _ in range(len(self.team_names) * matches_per_team_per_year):
                # Random date within the season
                random_days = np.random.randint(0, num_days)
                match_date = year_start + timedelta(days=random_days)
                
                # Random teams
                home_team = np.random.choice(self.team_names)
                away_team = np.random.choice(
                    [t for t in self.team_names if t != home_team]
                )
                
                # Generate realistic basketball statistics
                home_fg_pct = np.random.normal(0.46, 0.05)  # Field goal percentage
                away_fg_pct = np.random.normal(0.46, 0.05)
                
                home_3p_pct = np.random.normal(0.36, 0.06)  # 3-point percentage
                away_3p_pct = np.random.normal(0.36, 0.06)
                
                # Clamp percentages to valid range [0, 1]
                home_fg_pct = np.clip(home_fg_pct, 0.3, 0.6)
                away_fg_pct = np.clip(away_fg_pct, 0.3, 0.6)
                home_3p_pct = np.clip(home_3p_pct, 0.2, 0.5)
                away_3p_pct = np.clip(away_3p_pct, 0.2, 0.5)
                
                # Simulate scores based on shooting percentages
                home_fga = np.random.poisson(82)  # Field goal attempts
                away_fga = np.random.poisson(82)
                
                home_3pa = np.random.poisson(30)  # 3-point attempts
                away_3pa = np.random.poisson(30)
                
                # Calculate points
                home_2p = int((home_fga - home_3pa) * home_fg_pct)
                away_2p = int((away_fga - away_3pa) * away_fg_pct)
                home_3p = int(home_3pa * home_3p_pct)
                away_3p = int(away_3pa * away_3p_pct)
                
                home_score = home_2p * 2 + home_3p * 3 + np.random.randint(-5, 10)
                away_score = away_2p * 2 + away_3p * 3 + np.random.randint(-5, 10)
                
                # Ensure scores are reasonable (typically 90-130 points)
                home_score = max(80, min(150, home_score))
                away_score = max(80, min(150, away_score))
                
                # Other statistics
                home_rebounds = np.random.normal(48, 5)
                away_rebounds = np.random.normal(48, 5)
                
                home_assists = np.random.normal(26, 4)
                away_assists = np.random.normal(26, 4)
                
                home_turnovers = np.random.normal(14, 3)
                away_turnovers = np.random.normal(14, 3)
                
                # Determine winner
                winner = 'home' if home_score > away_score else 'away'
                
                match = {
                    'date': match_date,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': int(home_score),
                    'away_score': int(away_score),
                    'home_fg_pct': round(home_fg_pct, 3),
                    'away_fg_pct': round(away_fg_pct, 3),
                    'home_3p_pct': round(home_3p_pct, 3),
                    'away_3p_pct': round(away_3p_pct, 3),
                    'home_rebounds': round(home_rebounds, 1),
                    'away_rebounds': round(away_rebounds, 1),
                    'home_assists': round(home_assists, 1),
                    'away_assists': round(away_assists, 1),
                    'home_turnovers': round(home_turnovers, 1),
                    'away_turnovers': round(away_turnovers, 1),
                    'winner': winner
                }
                
                matches_list.append(match)
        
        self.matches_df = pd.DataFrame(matches_list)
        self.matches_df['date'] = pd.to_datetime(self.matches_df['date'])
        self.matches_df = self.matches_df.sort_values('date').reset_index(drop=True)
        
        return self.matches_df
    
    def calculate_team_statistics(self, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling team statistics from match history.
        
        Args:
            window (int): Rolling window size for calculating statistics (default: 20)
            
        Returns:
            pd.DataFrame: DataFrame with team statistics including:
                - team, wins, losses, win_rate, avg_points_for, avg_points_against,
                  avg_fg_pct, avg_3p_pct, avg_rebounds, avg_assists, avg_turnovers
        """
        if self.matches_df is None:
            raise ValueError("Match data not loaded. Call generate_sample_data() first.")
        
        team_stats_list = []
        
        for team in self.matches_df['home_team'].unique():
            # Filter matches where team played
            home_matches = self.matches_df[self.matches_df['home_team'] == team]
            away_matches = self.matches_df[self.matches_df['away_team'] == team]
            
            # Combine statistics
            home_wins = (home_matches['home_score'] > home_matches['away_score']).sum()
            away_wins = (away_matches['away_score'] > away_matches['home_score']).sum()
            total_wins = home_wins + away_wins
            
            total_matches = len(home_matches) + len(away_matches)
            total_losses = total_matches - total_wins
            
            if total_matches == 0:
                continue
            
            win_rate = total_wins / total_matches
            
            # Points statistics
            home_pf = home_matches['home_score'].sum()
            away_pf = away_matches['away_score'].sum()
            home_pa = home_matches['away_score'].sum()
            away_pa = away_matches['home_score'].sum()
            
            avg_points_for = (home_pf + away_pf) / total_matches
            avg_points_against = (home_pa + away_pa) / total_matches
            
            # Shooting statistics
            avg_fg_pct = (
                home_matches['home_fg_pct'].sum() + 
                away_matches['away_fg_pct'].sum()
            ) / total_matches
            
            avg_3p_pct = (
                home_matches['home_3p_pct'].sum() + 
                away_matches['away_3p_pct'].sum()
            ) / total_matches
            
            # Rebounds
            avg_rebounds = (
                home_matches['home_rebounds'].sum() + 
                away_matches['away_rebounds'].sum()
            ) / total_matches
            
            # Assists
            avg_assists = (
                home_matches['home_assists'].sum() + 
                away_matches['away_assists'].sum()
            ) / total_matches
            
            # Turnovers
            avg_turnovers = (
                home_matches['home_turnovers'].sum() + 
                away_matches['away_turnovers'].sum()
            ) / total_matches
            
            team_stats_list.append({
                'team': team,
                'wins': total_wins,
                'losses': total_losses,
                'total_games': total_matches,
                'win_rate': round(win_rate, 3),
                'avg_points_for': round(avg_points_for, 1),
                'avg_points_against': round(avg_points_against, 1),
                'point_differential': round(avg_points_for - avg_points_against, 1),
                'avg_fg_pct': round(avg_fg_pct, 3),
                'avg_3p_pct': round(avg_3p_pct, 3),
                'avg_rebounds': round(avg_rebounds, 1),
                'avg_assists': round(avg_assists, 1),
                'avg_turnovers': round(avg_turnovers, 1)
            })
        
        self.team_stats = pd.DataFrame(team_stats_list)
        self.team_stats = self.team_stats.sort_values('win_rate', ascending=False).reset_index(drop=True)
        
        return self.team_stats
    
    def prepare_features(self, 
                        target_variable: str = 'winner',
                        include_team_stats: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning models.
        
        Args:
            target_variable (str): Target variable name ('winner' or custom)
            include_team_stats (bool): Whether to include team statistics features
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (Features DataFrame, Target Series)
        """
        if self.matches_df is None:
            raise ValueError("Match data not loaded. Call generate_sample_data() first.")
        
        if self.team_stats is None:
            self.calculate_team_statistics()
        
        df = self.matches_df.copy()
        
        # Extract basic features from match data
        features_df = pd.DataFrame()
        features_df['home_fg_pct'] = df['home_fg_pct']
        features_df['away_fg_pct'] = df['away_fg_pct']
        features_df['home_3p_pct'] = df['home_3p_pct']
        features_df['away_3p_pct'] = df['away_3p_pct']
        features_df['home_rebounds'] = df['home_rebounds']
        features_df['away_rebounds'] = df['away_rebounds']
        features_df['home_assists'] = df['home_assists']
        features_df['away_assists'] = df['away_assists']
        features_df['home_turnovers'] = df['home_turnovers']
        features_df['away_turnovers'] = df['away_turnovers']
        
        # Add team statistics if requested
        if include_team_stats:
            # Create mappings for team statistics
            team_stats_dict = self.team_stats.set_index('team').to_dict()
            
            for stat in ['avg_points_for', 'avg_points_against', 'win_rate', 'avg_rebounds', 'avg_assists']:
                features_df[f'home_team_{stat}'] = df['home_team'].map(
                    team_stats_dict[stat]
                )
                features_df[f'away_team_{stat}'] = df['away_team'].map(
                    team_stats_dict[stat]
                )
        
        # Add temporal features
        features_df['day_of_week'] = df['date'].dt.dayofweek
        features_df['month'] = df['date'].dt.month
        features_df['day_of_year'] = df['date'].dt.dayofyear
        
        # Add derived features
        features_df['fg_pct_diff'] = features_df['home_fg_pct'] - features_df['away_fg_pct']
        features_df['3p_pct_diff'] = features_df['home_3p_pct'] - features_df['away_3p_pct']
        features_df['rebounds_diff'] = features_df['home_rebounds'] - features_df['away_rebounds']
        features_df['assists_diff'] = features_df['home_assists'] - features_df['away_assists']
        features_df['turnovers_diff'] = features_df['away_turnovers'] - features_df['home_turnovers']
        
        # Target variable
        target = (df[target_variable] == 'home').astype(int)
        
        return features_df, target
    
    def get_match_history(self, 
                         team: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve match history with optional filtering.
        
        Args:
            team (Optional[str]): Filter by team name (home or away)
            start_date (Optional[str]): Start date in 'YYYY-MM-DD' format
            end_date (Optional[str]): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Filtered match history
        """
        if self.matches_df is None:
            raise ValueError("Match data not loaded. Call generate_sample_data() first.")
        
        result = self.matches_df.copy()
        
        if team:
            result = result[
                (result['home_team'] == team) | (result['away_team'] == team)
            ]
        
        if start_date:
            result = result[result['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            result = result[result['date'] <= pd.to_datetime(end_date)]
        
        return result.reset_index(drop=True)
    
    def get_statistics_summary(self) -> Dict:
        """
        Get a summary of loaded data statistics.
        
        Returns:
            Dict: Dictionary containing data summary statistics
        """
        if self.matches_df is None:
            return {'status': 'No data loaded'}
        
        summary = {
            'total_matches': len(self.matches_df),
            'date_range': f"{self.matches_df['date'].min().date()} to {self.matches_df['date'].max().date()}",
            'unique_teams': self.matches_df['home_team'].nunique(),
            'teams': sorted(self.matches_df['home_team'].unique().tolist()),
            'avg_home_score': round(self.matches_df['home_score'].mean(), 1),
            'avg_away_score': round(self.matches_df['away_score'].mean(), 1),
            'home_win_rate': round(
                (self.matches_df['home_score'] > self.matches_df['away_score']).sum() / 
                len(self.matches_df), 
                3
            ),
            'avg_scoring_margin': round(
                abs(self.matches_df['home_score'] - self.matches_df['away_score']).mean(),
                1
            )
        }
        
        return summary


def main():
    """Example usage of DataLoader class."""
    # Initialize loader
    loader = DataLoader(start_year=2021, end_year=2025, seed=42)
    
    # Generate sample data
    print("Generating sample data...")
    matches = loader.generate_sample_data(matches_per_team_per_year=50)
    print(f"Generated {len(matches)} matches\n")
    
    # Display first few matches
    print("Sample matches:")
    print(matches.head())
    print()
    
    # Calculate team statistics
    print("Calculating team statistics...")
    team_stats = loader.calculate_team_statistics()
    print("\nTop 5 teams by win rate:")
    print(team_stats[['team', 'wins', 'losses', 'win_rate', 'avg_points_for', 'avg_points_against']].head())
    print()
    
    # Prepare features for ML
    print("Preparing features for machine learning...")
    X, y = loader.prepare_features(include_team_stats=True)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("\nFeature columns:")
    print(X.columns.tolist())
    print()
    
    # Get match history for specific team
    print("Match history for Lakers:")
    lakers_history = loader.get_match_history(team='Lakers')
    print(f"Total matches: {len(lakers_history)}")
    print(lakers_history[['date', 'home_team', 'away_team', 'home_score', 'away_score', 'winner']].head())
    print()
    
    # Get statistics summary
    print("Data statistics summary:")
    summary = loader.get_statistics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
