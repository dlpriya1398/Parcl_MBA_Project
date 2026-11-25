import pandas as pd

# Load your large CSV file
df = pd.read_csv('properties.csv')
chunk_size = 2000  # Splits into 2,000-row chunks

# Save each chunk into a separate CSV file
for i, start in enumerate(range(0, len(df), chunk_size)):
    df_chunk = df.iloc[start : start + chunk_size]
    file_name = f'properties_part_{i+1}.csv'
    df_chunk.to_csv(file_name, index=False)
    print(f'Saved {file_name} ({len(df_chunk)} rows)')
