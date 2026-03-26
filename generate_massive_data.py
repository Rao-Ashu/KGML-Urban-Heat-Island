import pandas as pd
import numpy as np
import os

# Starting Data Augmentation Pipeline

# 1. Load Original Data

df = pd.read_csv('Data.csv')
print(f"Original Data Loaded: {len(df)} rows.")

# Create a proper datetime index (The data is recorded in January of each year)
# Years are 13 to 21, so mapping them to 2013-2021
years = ['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', 
         '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01']
df['Date'] = pd.to_datetime(years)
df.set_index('Date', inplace=True)

# Drop the old 'year' column and any empty/unnamed columns
df.drop(columns=['year'], inplace=True, errors='ignore')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# 2. Strategy 1: Temporal Interpolation (Applying Temporal Interpolation (Yearly -> Monthly))
#Since land cover changes gradually, by using mathematical (Spline/Linear) interpolation yearly data can be converted into monthly or quarterly data.
#Current Data: 9 Years = 9 rows.
#Monthly Interpolation: 9 Years × 12 Months = 108 rows.

# Upsample to Monthly ('MS' = Month Start) and Interpolate using polynomial curve
df_monthly = df.resample('MS').interpolate(method='polynomial', order=2)

# Ensure no negative areas exist due to mathematical curve dipping
df_monthly[df_monthly < 0] = 0 
print(f"Interpolation Complete: Dataset expanded to {len(df_monthly)} rows.")




# 3. Strategy 2: Synthetic data augmentation (creating random variations with noise to simulate different scenarios)
# To generate parallel "synthetic" cities. By taking existing data and create hundreds of slightly altered copies by injecting Gaussian noise.
# This simulates slight variations in measurement errors or alternate urban growth paths.
# Current Data: 9 rows.
# Augmented Data: 9 rows × 50 variations = 450 rows.

def augment_data(base_df, num_variations=50, noise_level=0.02):
    print(f"Applying Augmentation (Creating {num_variations} synthetic variations with {noise_level*100}% noise)...")
    augmented_dfs = []
    
    # Keep the original monthly data as variation 0
    base_df_copy = base_df.copy()
    base_df_copy['Variation'] = 0
    augmented_dfs.append(base_df_copy)
    
    for i in range(1, num_variations + 1):
        synthetic_df = base_df.copy()
        synthetic_df['Variation'] = i
        
        # Add random Gaussian noise to each land cover column
        for col in ['Vegetation', 'Barren', 'Water', 'Buildup']:
            # Noise is proportional to the standard deviation of the column
            noise = np.random.normal(0, noise_level * synthetic_df[col].std(), len(synthetic_df))
            synthetic_df[col] = synthetic_df[col] + noise
            # Ensure no negative values after noise
            synthetic_df[col] = synthetic_df[col].clip(lower=0)
            
        augmented_dfs.append(synthetic_df)
        
    return pd.concat(augmented_dfs)




# Generate massive dataset
df_massive = augment_data(df_monthly, num_variations=50, noise_level=0.02)


# 4. Save the final dataset
output_filename = 'Data_Massive.csv'
df_massive.to_csv(output_filename)
print(f"--- Pipeline Complete ---")
print(f"Massive dataset saved to '{output_filename}' with {len(df_massive)} rows!")