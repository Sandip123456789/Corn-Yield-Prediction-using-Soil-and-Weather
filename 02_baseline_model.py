import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

'Step 1: Prepare the Data'
# We use the 'cleaned or preprocessed' data from the previous step
# X = Inputs (Features)
# y = Output (Target)

file_name = 'processed_corn_data.csv'
n_folds = 5 # split data into 5 batches of Districts for cross-validation

#1. Load Clean Data
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print("Error: processed_corn_data not found!!")
    exit()

# 2. Setup Features & Groups
# We need 'District' for the GroupKFold (to avoid spatial leakage)
if 'District' not in df.columns:
    print("Error: 'District' column is missing. The model cannot validate correctly.")
    exit()

groups = df['District']
y = df['Yield_per_Ha']

# Function: To Train and Evaluate Model
def run_experiment(name, features_to_drop):
    print(f"\n Experiment: {name} - Dropping features: {features_to_drop}")

    # Drop features
    drop_cols = features_to_drop + ['District', 'State', 'Yield_per_Ha']
    # Only drop if they exist (safety check)
    actual_drop = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=actual_drop)
    print(f" Training on {len(X.columns)} features: {list(X.columns)}")

    # Initialize Group K-Fold
    gkf = GroupKFold(n_splits=n_folds)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    scores = []
    rmse_list = []
    feature_importances = np.zeros(len(X.columns))

    # The loop for Group K-Fold Cross-Validation
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Scoring
        # Handle cases with zero variance in test fold
        if len(np.unique(y_test)) > 1:
            scores.append(r2_score(y_test, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        
        feature_importances += model.feature_importances_

    # Average Results
    avg_r2 = np.mean(scores) if scores else 0.0
    avg_rmse = np.mean(rmse_list)
    avg_importances = feature_importances / n_folds

    print(f"RESULTS for {name}:")
    print(f"Average R² Score (Accuracy):   {avg_r2:.4f}")
    print(f"Average Root Mean Squared Error (RMSE): {avg_rmse:.4f}")

    # Return feature importances for analysis
    return avg_r2, avg_importances, X.columns

# Experiment A: FULL MODEL (With Temperature)
r2_full, imp_full, cols_full = run_experiment("Baseline (With Temp)", [])

# Experiment B: PHYSICS ONLY (No Temperature)
# We drop all temperature columns to force the model to look at Soil/Rain
temp_cols = ['Min_Temp', 'Max_Temp', 'Avg_Temp']
r2_phys, imp_phys, cols_phys = run_experiment("Physics Only (Blindfold)", temp_cols)

# REPORTING & VISUALIZATION
print("\n" + "="*40)
print(f"FINAL SCORECARD")
print("="*40)
print(f"1. Baseline (Geography/Temp): {r2_full:.4f}")
print(f"2. Physics Only (Soil/Rain):  {r2_phys:.4f}")

# Plot Feature Importance for the Physics Model
# We want to know: If we take away Temp, does Soil actually matter?
feat_df = pd.DataFrame({'Feature': cols_phys, 'Importance': imp_phys})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=True)
plt.title(f"Physics Model Drivers (R² = {r2_phys:.2f})")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
