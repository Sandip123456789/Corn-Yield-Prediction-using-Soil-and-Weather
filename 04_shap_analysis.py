import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

# Configuration
model_file = 'best_corn_xgboost.pkl'
data_file = 'processed_corn_data.csv'

print(" Starting SHAP Interpretation ...")

# 1. Load Data & Model
try:
    df = pd.read_csv(data_file)
    model = joblib.load(model_file)
except FileNotFoundError:
    print(" Error: Missing model or Data file.")
    exit()

# 2. Prepare Features (X)
# Must match the exact columns used during training
drop_cols = ['Yield_per_Ha', 'State', 'District']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])

print(f" Analyzing {len(X)} samples...")

# 3. Calculate SHAP Values
# SHAP explains the output of the model. It tells us, for every single row,
# how much each feature pushed the prediction UP or DOWN.
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 4. PLOT 1: THE SUMMARY (Beeswarm)
# This shows the direction of the relationship.
# Red = High Value of Feature, Blue = Low Value of Feature
# Right = Higher Yield, Left = Lower Yield
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary: How Features Impact Yield")
plt.tight_layout()
plt.show()

# 5. PLOT 2: THE PHYSICS CHECK (Dependence)
# Let's check the #1 driver: Max_Temp.
# Does the curve look like a hill (Goldilocks zone)?
top_feature = 'Max_Temp' 
if top_feature in X.columns:
    #plt.figure(figsize=(8, 5))
    shap.dependence_plot(top_feature, shap_values.values, X, show=False)
    plt.title(f"Physics Check: Impact of {top_feature} on Yield")
    plt.grid(True, alpha=0.3)
    plt.show()

# Let's check #2: Precipitation
second_feature = 'Avg_Precipitation'
if second_feature in X.columns:
    #plt.figure(figsize=(8, 5))
    shap.dependence_plot(second_feature, shap_values.values, X, show=False)
    plt.title(f"Physics Check: Impact of {second_feature} on Yield")
    plt.grid(True, alpha=0.3)
    plt.show()