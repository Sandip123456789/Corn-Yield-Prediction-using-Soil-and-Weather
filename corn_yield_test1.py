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

initial_columns = set(rename_cols.values())

# --- STEP 2: The Surgical Cleaning ---
# A. Create a Temporary Efficiency Metric to spot outliers
# We need to know Yield per Hectare to spot the "fake 12.0" values
df['Temp_Yield_Efficiency'] = df['Total_Production'] / df['Area_Ha']

# B. Apply Filters
# Filter 1: Physics - pH must be > 0 (removes -1000 error codes)
# Filter 2: Forensics - Yield must be < 10 (removes the Crossriver copy-paste error)
# Filter 3: Validity - Area must be > 0 (dividing by zero is impossible)
clean_df = df[
    (df['pH'] > 0) & 
    (df['Temp_Yield_Efficiency'] < 10) & 
    (df['Area_Ha'] > 0)
].copy()

# Creating official Target Variable on the clean data
clean_df['Yield_per_Ha'] = clean_df['Total_Production'] / clean_df['Area_Ha']

# Double check we dropped the bad rows
# We filter again just to be safe: remove anything > 10 tonnes/ha (the fake line)
final_df = clean_df[clean_df['Yield_per_Ha'] < 10].copy()

print(f"Original Rows: {len(df)}")
print(f"Final Cleaned Rows: {len(final_df)}")
print(f"Dropped {len(df) - len(final_df)} rows of bad data.")

# --- STEP 3: CORRELATION MATRIX ---
# Now we ask: "What actually drives Yield?"
# We drop non-numeric columns (State/District) for the math
numeric_df = final_df.select_dtypes(include=['float64', 'int64'])

#Calculate Correlation Matrix
corr_matrix = numeric_df.corr()

# ---- Visualize Distributions ----
# Scatter Plot: Yield_per_Ha vs pH
plt.figure(figsize=(12, 6))
# We use 'hue' to show a 3rd dimension: Clay percentage.
# This helps us see if soil type changes the pH/Yield relationship.
sns.scatterplot(
    data=clean_df,
    x='pH',
    y='Yield_per_Ha',
    hue='Clay',
    palette='viridis',
    alpha=0.7
)

plt.title("Corrected Physics: Yield Efficiency vs Soil pH")
plt.xlabel("Soil pH (0-14)")
plt.ylabel("Yield (Production / Hectare)")
plt.grid(True, alpha=0.3)
plt.show()

# Heatmap: Correlation Matrix
plt.figure(figsize=(12, 6))

# We use Correlation Heatmap to visualize
sns.heatmap(
    corr_matrix,
    annot=True, # Write the numbers in the boxes
    fmt=".2f",  # 2 decimal places
    cmap='coolwarm', # Blue = Negative corr, Red = Positive corr
    center=0
)

plt.title("Heatmap: What Correlates with Yield?")
plt.show()

"""export_df = final_df

output_filename = 'cleaned_data/processed_corn_data.csv'
export_df.to_csv(output_filename, index=False)"""

"""print(f"[ETL] Success. Clean data saved to '{output_filename}'.")
print(f"Original Rows: {len(df)}")
print(f"Cleaned Rows: {len(final_df)}")"""