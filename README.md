Corn Yield Prediction System using Soil and Weather

A "Gray Box" Machine Learning system that predicts corn yield efficiency based on soil physics and weather patterns, optimized for biological plausibility over raw statistical accuracy.

## OVERVIEW:

This project addresses a common failure mode in agricultural ML: Spatial Overfitting. Standard models often memorize the "temperature signature" of specific high-yield districts rather than learning the agronomic drivers of growth.

This system forces the model to learn Soil Physics (Clay/Sand/Silt interactions) and Weather Constraints (Heat Stress & Water Limits) by using a custom Group K-Fold Validation strategy and Regularized XGBoost.

Key Achievements:
1. Forensic Data Cleaning: Identified and removed synthetic "Ghost Data" (Crossriver Anomaly) and impossible physical outliers (Max_Temp = -2477°C).
2. Leakage Elimination: Removed mathematical proxies (Total_Production, Area_Ha) to ensure the model predicts efficiency (t/ha), not just definitions.
3. Physics-Aware Tuning: Optimized XGBoost with colsample_bytree: 0.7 to prevent the model from relying solely on temperature as a location proxy.
4. Discovery: The model independently discovered the "Heat Cliff" (yield crash > 30.5°C) and the "Water Buffer" effect.

## PROJECT PIPELINE:

The project is structured into 5 distinct phases, mimicking a professional Data Science lifecycle.

          Phase,      Script,              Objective,                       Key Technique
      1. Engineering, 01_data_engineering.py, Forensic Cleaning & Constraints, "Filter: Drops impossibilities (pH<0, Temp<-50)."
      2. Validation , 02_baseline_model.py, Baseline & Ablation Study, "Group K-Fold: Prevents ""Neighbor Cheating"" by splitting data by District."
      3. Optimization, 03_xgboost_tuning.py, Hyperparameter Tuning, RandomizedSearchCV: Found max_depth: 6 to balance complexity/generalization.
      4. Analysis, 04_shap_analysis.py, Interpretability (Black Box Opening), SHAP Beeswarm: Visualized the impact of Soil Texture and Heat Stress.
      5. Deployment, 05_deployment_app.py, Production Interface, "Streamlit: Interactive DSS with ""AI Insights"" wrapper logic."

## MODEL PERFORMANCE: ACCURACY vs. REALITY

We compared two architectures. The "Statistical Winner" was rejected in favor of the "Scientific Winner."

1. The Baseline (Random Forest)

         RMSE: 0.0996 (Very Low Error)
   
         Accuracy (R²): ~97%
   
         Verdict: Rejected.

      Reason: Feature Importance showed 98% reliance on Min_Temp. The model had "memorized" the temperature barcodes of specific districts. It failed to generalize to physics (Ablation Score: 0.68).

3. The Champion (XGBoost)

        RMSE: 0.1867 (Higher Error)

        Accuracy (R²): ~83%

        Verdict: Deployed.

      Reason: Feature Importance is balanced (Max_Temp 28%, Rain 19%, Soil 15%). It successfully captures non-linear biological constraints (Diminishing returns of rainfall). It is robust to location changes.

## INSIGHTS:

The final deployed model understands three critical agronomic rules without being explicitly programmed:

    Heat Cliff: Yield increases with temperature up to 30°C, then crashes vertically. (Heat Stress).

    Water Buffer: High clay content and rainfall mitigate the damage of high heat.

    Soil Texture: Silt is preferred over Sand in rain-fed systems due to water retention.

(These insights were verified via SHAP Dependence Plots in Phase 4).

## HOW TO RUN

Follow these steps to reproduce the project locally.

### 1. Prerequisites
Ensure you have ***Python 3.8+*** installed.

### 2. Installation
It is recommended to use a virtual environment to avoid conflicts.

          ```bash
          # Clone the repository
          git clone [https://github.com/Sandip123456789/Corn-Yield-Prediction-using-Soil-and-Weather.git](https://github.com/Sandip123456789/Corn-Yield-Prediction-using-Soil-and-Weather.git)
          cd Corn-Yield-Prediction-using-Soil-and-Weather

          # Create a virtual environment
          python -m venv venv

          # Activate environment
          # On Windows:
          venv\Scripts\activate
          # On Mac/Linux:
          source venv/bin/activate

          # Install dependencies using the provided file
          pip install -r requirements.txt

### 3. Data Setup
   
- Download the dataset from [Mendeley Data](https://data.mendeley.com/datasets/dkv6b3xj99/1).
- Rename the file to raw_corn_data.xlsx (if necessary) and place it inside data folder.

### 4. Execution Pipeline
Run the scripts in this specific order:

#### A. Clean the Data:
   Bash
   
        python 01_data_engineering.py

   (Output: cleaned_data/processed_corn_data.csv - approx 1830 clean rows)

#### B. Train the Model:
   Bash

        python 03_xgboost_tuning.py

   (Output: best_corn_xgboost.pkl - Trains the XGBoost model)

#### C. Launch Dashboard:
   Bash

        streamlit run 05_deployment_app.py

   (Output: Opens the interactive web app in your browser.)

## KNOWN CHALLENGES & RESOLUTIONS

        Outlier (-2477): SHAP analysis revealed a row with Max_Temp = -2477. Fixed by implementing a "Nuclear Filter" in 00_corn_yield_de.py.

        Serialization Conflict: xgboost wrapper caused TypeError when saving to JSON. Resolved by switching to joblib for robust object serialization.

        Streamlit Context Warning: ScriptRunContext errors suppressed via warnings.filterwarnings("ignore") for clean UX.

## LICENSE

This project is for educational and research purposes. Data provided by Mendeley Data.

Author: Sandip Chaudhary Status: Operational Production Build

References: [1] Olisah, C., Smith, L., Morolake, L., & Ojukwu, O. (2024). Single Point Corn Yield Data - Weather, Soil, Cultivation Area, and Yield for Precision Agriculture. Mendeley Data, V1, doi: 10.17632/dkv6b3xj99.1 [2] Lundberg, S. M. and Lee, S.-I. (2017). 'A Unified Approach to Interpreting Model Predictions'. Advances in Neural Information Processing Systems, 30, 4765-4774.  [3] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16. [4] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Müller, A., Nothman, J., Louppe, G., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2012). Scikit-learn: Machine learning in Python.
