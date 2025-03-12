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

# Load processed data
print("Loading processed data...")
df = pd.read_csv('defender_processed.csv')
print(f"Loaded {len(df)} rows of data")

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values[missing_values > 0])

# Define original basic features
basic_features = [
    'minutes_trend', 'rolling_points', 'value', 'avg_points_home', 'avg_points_away',
    'attacking_returns', 'clean_sheet_points', 'was_home', 'opponent_strength_factor'
]

# Define enhanced features
enhanced_features = basic_features + [
    'rolling_points_5', 'minutes_trend_5', 'minutes_consistency',
    'recent_value', 'home_clean_sheet_rate', 'away_clean_sheet_rate',
    'opponent_attack_strength', 'recent_goals', 'recent_assists',
    'bonus_trend', 'team_clean_sheets_last_5', 'team_goals_conceded_last_5',
    'next_3_avg_difficulty', 'next_3_min_difficulty', 'next_3_max_difficulty'
]

# Check if all features exist in the dataframe
available_enhanced_features = [f for f in enhanced_features if f in df.columns]
missing_features = [f for f in enhanced_features if f not in df.columns]

if missing_features:
    print(f"\nWarning: The following enhanced features are missing from the data: {missing_features}")
    print("Available columns:", df.columns.tolist())
    print("Proceeding with available features only.")
    features = available_enhanced_features
else:
    features = enhanced_features

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
    ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3))
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
    ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3))
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
plt.title('Feature Importance for Defender Points Prediction')
plt.tight_layout()
plt.savefig('defender_feature_importance.png')
print("Feature importance plot saved to 'defender_feature_importance.png'")

# Evaluate model on test data
test_size = int(len(X) * 0.2)
X_test = X.iloc[-test_size:]
y_test = y.iloc[-test_size:]
test_predictions = final_pipeline.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)
print(f"\nTest MAE: {test_mae:.2f}")

# Save the model
print("\nSaving model to 'defender_model_enhanced.pkl'...")
with open('defender_model_enhanced.pkl', 'wb') as f:
    pickle.dump(final_pipeline, f)

# Also save a version with the original name for compatibility
print("Also saving as 'defender_model.pkl' for compatibility...")
with open('defender_model.pkl', 'wb') as f:
    pickle.dump(final_pipeline, f)

# Save feature list used for training (helpful for prediction)
with open('defender_model_features.txt', 'w') as f:
    f.write('\n'.join(features))
print(f"Feature list saved to 'defender_model_features.txt'")

print("\nModel training completed successfully!")