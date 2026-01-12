import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

# Configuation
file_name = 'cleaned_data/processed_corn_data.csv'
n_folds = 5     # 5-Fold Validaiton
n_iter = 50     # Trying 50 different combinations (Higher = better, but slower)

# 1. Load Data & Prepare
df = pd.read_csv(file_name)

if 'District' not in df.columns:
    print("Error: 'District' column missing. Cannot perform GroupKFold.")
    exit()

# Defining Features (X) and Target (y)
# Using ALL available features (Soil + Rain + Temp) to get maximum performance
drop_cols = ['District', 'State', 'Yield_per_Ha']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['Yield_per_Ha']
groups = df['District']

print(f" Training on {X.shape[1]} features: {list(X.columns)} ")

# 2. Define the Hyperparameter Grid
# This is the "Search Space" the AI will explore
param_grid = {
    # 1. Architecture (How big is the brain?)
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [3, 4, 5, 6], # keep low (3-6) to prevent memorization/overfitting

    # 2. Speed (How fast does it correct errors?)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],

    # 3. Randomness (Preventing Overfitting)
    'subsample': [0.6, 0.7, 0.8, 0.9],          # Use only % of rows per tree
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],   # Use only % of columns per tree

    # 4. Regularization (Penalty for complexity)
   'reg_alpha': [0, 0.1, 1, 10],             # L1 Regularization
    'reg_lambda': [0, 1, 10]                 # L2 Regularization

}

# 3. Setup The Search ("Brain")
# Must manually generate the GroupKFold splits to pass to the Search
# This ensures we respect the "Don't split Districts" rule during tuning

gkf = GroupKFold(n_splits=n_folds)
cv_splits = list(gkf.split(X, y, groups=groups))

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

print(f"   Searching {n_iter} random combinations across {n_folds} folds...")
print(f"   (This involves fitting {n_iter * n_folds} models. Please wait...)")

search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=n_iter,
    scoring='neg_root_mean_squared_error',      # Optimize for Lowest RMSE
    cv=cv_splits,                               # Use our custom Group splits
    verbose=1,
    random_state=42,
    n_jobs=-1
    # refit=True
)

# Run the Search
search.fit(X, y)

#4. Report Results
best_model = search.best_estimator_
best_params = search.best_params_
best_rmse = -search.best_score_ # Flip sign back to positive RMSE

print("\n" + "="*40)
print(f"Champion Model Found.")
print("="*40)
print(f"Best RMSE (Validation): {best_rmse:.4f}")
print("Best Parameters:")
for param, value in best_params.items():
    print(f" - {param}: {value}")

print("\n" + "="*40)
print(f"CALCULATING REAL ACCURACY...")
print("="*40)

# Using the exact same grouping strategy (GroupKFold)
# This forces the model to predict on districts it has NEVER seen.
cv_scores = cross_val_score(
    best_model, 
    X, 
    y, 
    groups=groups, 
    cv=gkf, 
    scoring='r2' # <--- We explicitly ask for R-Squared accuracy
)

print(f"Training Score (Glitch): {best_model.score(X, y):.4f} (99% - Ignore this)")
print(f"Validation Score (Truth): {cv_scores.mean():.4f} ({(cv_scores.mean()*100):.2f}%)")

# # Saving the Model
# best_model.save_model('best_corn_xgboost.json')
# print("\n Saved best model to 'best_corn_xgboost.json'")

# Saving the Model using joblib
model_filename = "best_corn_xgboost.pkl"
joblib.dump(best_model, model_filename)
print(f"\nSaved best model to '{model_filename}' (using joblib)")

# Feature Importance Check
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- New Drivers of Yield (XGBoost) ---")
print(importance.head(5))