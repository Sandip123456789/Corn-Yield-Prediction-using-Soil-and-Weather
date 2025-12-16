import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Configuation
file_name = 'processed_corn_data.csv'
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
    'subsample': []

    # 4. Regularization (Penalty for complexity)
}