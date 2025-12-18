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