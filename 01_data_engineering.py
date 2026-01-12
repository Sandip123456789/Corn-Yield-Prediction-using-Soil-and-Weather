import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""# CONFIGURATION
INPUT_FILE = 'data/raw_corn_data.xlsx'
OUTPUT_FILE = 'cleaned_data/processed_corn_data.csv'"""

# ---- Step 1: Load & Clean Data -----
file_path = 'data/raw_corn_data.xlsx' 
#1. Reload the data
df = pd.read_excel(file_path)
print(f"Raw Data Loaded: {len(df)} rows.")

# ----- Fix Columns Names -----
# Current columns names are messy (mixed caps, hyphens)
rename_cols = {
    'Abia': 'State',
    'District': 'District',
    'Average_avg-Temp': 'Avg_Temp',
    'Average-Min Temp': 'Min_Temp',
    'Average-max-temp': 'Max_Temp',
    'avg-precipitation': 'Avg_Precipitation',
    'avg-windSpeed': 'Wind_Speed',
    'PH': 'pH',
    'Crop Yield': 'Total_Production',
    'Hectare': 'Area_Ha'
    # Note: If there are Clay/Sand columns, ensure they match too.
    # Usually they are 'Clay', 'Sand', 'Silt' in this dataset.
}

df = df.rename(columns=rename_cols)


# Identify the outlier rows (max temp < -50)
print("\n Looking for outliers...")
outliers = df[df['Max_Temp'] < -50]

if len(outliers) > 0:
    print(f" Found {len(outliers)} outlier(s).")
    print(f" Bad Row Indices: {outliers.index.tolist()}")
    print(f" Bad Values: {outliers['Max_Temp'].values}")
else:
    print(" No outliers found in raw data.")

# Using explicit method to drop outliers
# Instead of boolean logic, we drop the specific indices found above
if len(outliers) > 0:
    print(f" Dropping outliers: {outliers.index.tolist()}")
    df = df.drop(outliers.index)
    print(f" Dropped successful. Remaining rows: {len(df)}")


initial_count = len(df)

# --- STEP 2: The Surgical Cleaning ---
# A. Create a Temporary Efficiency Metric to spot outliers
# We need to know Yield per Hectare to spot the "fake 12.0" values
df['Temp_Yield_Efficiency'] = df['Total_Production'] / df['Area_Ha']

# B. Apply Filters
# Filter 1: Physics - pH must be > 0 (removes -1000 error codes)
# Filter 2: Forensics - Yield must be < 10 (removes the Crossriver copy-paste error)
# Filter 3: Validity - Area must be > 0 (dividing by zero is impossible)
# Filter 4: Max_Temp must be within reasonable bounds (temp below -50 is bullshit)
clean_df = df[
    (df['pH'] > 0) & 
    (df['Temp_Yield_Efficiency'] < 10) & 
    (df['Area_Ha'] > 0 &
    (df['Max_Temp'] > -50))  # Safety double-check
].copy()

# Creating official Target Variable on the clean data
clean_df['Yield_per_Ha'] = clean_df['Total_Production'] / clean_df['Area_Ha']

# LEAKAGE REMOVAL
# Dropping 'Total_Production' and 'Area_Ha' now
# This guarantees that my future models doesn't cheat.
cols_to_drop = ['Total_Production', 'Area_Ha', 'Temp_Yield_Efficiency']
final_df = clean_df.drop(columns=cols_to_drop)

# REPORTING & SAVING
dropped_rows = initial_count - len(final_df)
print(f"\nCLEANING REPORT:")
print(f"- Original Rows: {initial_count}")
print(f"- Dropped Rows:  {dropped_rows}")
print(f"- Final Rows:    {len(final_df)}")
print(f"- Columns Kept:  {list(final_df.columns)}")

# Save to disk
export_df = final_df

output_filename = 'cleaned_data/processed_corn_data.csv'
export_df.to_csv(output_filename, index=False)

print(f"[ETL] Success. Clean data saved to '{output_filename}'.")
print("Ready for Modeling...")