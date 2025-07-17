import pandas as pd
import glob
import os

# Path to your oil data files
data_dir = 'C:/Users/adoma/source/repos/GHG_monitoring/data/GIS'
output_file = os.path.join(data_dir, 'compiled_oil_data.csv')

# Find all relevant CSVs (update pattern if needed)
csv_files = glob.glob(os.path.join(data_dir, 'Oil_data_*.csv'))

dfs = []
for file in csv_files:
    try:
        # Try reading with header=0, skip bad lines
        df = pd.read_csv(file, header=0, encoding='latin1', on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(f"Loaded {file} with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print(f"Could not load {file}: {e}")

if not dfs:
    print("No data loaded. Please check your files.")
    exit(1)

# Find the intersection of all columns (columns present in every file)
common_cols = set(dfs[0].columns)
for df in dfs[1:]:
    common_cols &= set(df.columns)
common_cols = list(common_cols)

# Keep only common columns in all DataFrames
dfs = [df[common_cols] for df in dfs]

# Concatenate all DataFrames
compiled_df = pd.concat(dfs, ignore_index=True)

# Save to CSV
compiled_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Compiled data saved to {output_file} with {compiled_df.shape[0]} rows and {compiled_df.shape[1]} columns.") 