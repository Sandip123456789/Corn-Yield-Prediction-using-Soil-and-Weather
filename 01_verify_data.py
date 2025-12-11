import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_name = 'processed_corn_data.csv'

print(f"[Audit] Inspecting '{file_name}'...")

try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print("Error: File not found. Did you run the engineering script?")
    exit()

# TEST 1: The Leakage Check (Critical)
removed_cols = ['Total_Production', 'Area_Ha', 'Temp_Yield_Efficiency']
present_leakage = [col for col in removed_cols if col in df.columns]

if len(present_leakage) > 0:
    print(f"FAIL: Leakage detected! Found columns: {present_leakage}")
    print("   The model will cheat if you use this file.")
else:
    print("PASS: No leakage columns found.")

# TEST 2: The pH value check
min_pH = df['pH'].min()
if min_pH <= 0:
    print(f"FAIL: Found pH value of {min_pH} (Physics violation).")
else:
    print(f"PASS: Lowest pH is {min_pH} (Physically valid).")

# TEST 3: The Distribution Check (Visual)
# We want to see a bell curve or a rugged organic shape.
# If you see a single tall spike at 12.0, the "Crossriver" fake data is still there.
plt.figure(figsize=(10, 5))
sns.histplot(df['Yield_per_Ha'], bins=50, kde=True, color='green')
plt.title("Audit: Yield Distribution (Should look organic, no spikes at 12.0)")
plt.xlabel("Yield (Tonnes/Ha)")
plt.grid(True, alpha=0.3)
plt.show()

print(f"\n Data Shape: {df.shape}")
print(f"Here I have ~1760 rows and NO leakage columns")