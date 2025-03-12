from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load processed data
print("Loading processed midfielder data...")
try:
    df = pd.read_csv('midfielder_processed.csv')
    print(f"Loaded {len(df)} rows of midfielder data")
except FileNotFoundError:
    print("Error: midfielder_processed.csv not found. Please run the feature engineering script first.")
    exit(1)

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Define core features that should be available from basic FPL data
core_features = [
    'minutes_trend', 'rolling_points', 'value', 'avg_points_home', 'avg_points_away',
    'attacking_returns', 'opponent_strength_factor', 'was_home', 'recent_goals', 
    'recent_assists', 'bonus_trend'
]

# Define enhanced features that might be available
enhanced_features = core_features + [
    'goals_per_90', 'assists_per_90', 'goal_involvement_per_90',
    'shots_per_90', 'key_passes_per_90', 'minutes_consistency',
    'goal_involvement_pct', 'recent_goal_inv_pct',
    'shot_conversion', 'recent_creativity', 'recent_threat',
    'xG_per_90', 'xA_per_90', 'goals_minus_xG', 'assists_minus_xA',
    'set_piece_taker', 'penalty_taker', 'dribbles_per_90',
    'team_recent_goals', 'next_3_attacking_potential'
]

# Check which features are available in the dataframe
available_features = [f for f in enhanced_features if f in df.columns]
missing_features = [f for f in enhanced_features if f not in df.columns]

if missing_features:
    print(f"\nSome enhanced features are missing from the data: {missing_features}")
    print("Proceeding with available features.")

# Ensure all core features are available
missing_core = [f for f in core_features if f not in df.columns]
if missing_core:
    print(f"Warning: Core features are missing: {missing_core}")
    print("The model may not perform optimally without these features.")
    # Continue with available features
    features = available_features
else:
    print("All core features are available.")
    features = available_features

print(f"\nUsing {len(features)} features for model training:")
print(features)

# Prepare the data
X = df[features]
y = df['total_points']

# Create time-series split for validation
tscv = TimeSeriesSplit(n_splits=5)

# Create a pipeline with an imputer and the model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4))
])

# Train and evaluate the model
mae_scores = []

print("\nPerforming time-series cross-validation...")
fold = 1
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mae_scores.append(mae)
    
    print(f"Fold {fold} MAE: {mae:.2f}")
    fold += 1

print(f"\nAverage MAE: {np.mean(mae_scores):.2f}")

# Train the final model on all data
print("\nTraining final model on all data...")
final_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4))
])
final_pipeline.fit(X, y)

# Feature importance
final_model = final_pipeline.named_steps['model']
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Create feature importance visualization
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Midfielder Points Prediction')
plt.tight_layout()
plt.savefig('midfielder_feature_importance.png')
print("Feature importance plot saved to 'midfielder_feature_importance.png'")

# Evaluate model on test data
test_size = int(len(X) * 0.2)
X_test = X.iloc[-test_size:]
y_test = y.iloc[-test_size:]
test_predictions = final_pipeline.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
print(f"\nTest MAE: {test_mae:.2f}")

# Create a scatter plot of predicted vs actual points
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Points')
plt.ylabel('Predicted Points')
plt.title('Midfielder Model: Predicted vs Actual Points')
plt.tight_layout()
plt.savefig('midfielder_prediction_accuracy.png')
print("Prediction accuracy plot saved to 'midfielder_prediction_accuracy.png'")

# Save the model
print("\nSaving model to 'midfielder_model.pkl'...")
with open('midfielder_model.pkl', 'wb') as f:
    pickle.dump(final_pipeline, f)

# Save feature list used for training (helpful for prediction)
with open('midfielder_model_features.txt', 'w') as f:
    f.write('\n'.join(features))
print(f"Feature list saved to 'midfielder_model_features.txt'")

print("\nModel training completed successfully!")