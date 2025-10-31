import pandas as pd

input_file = 'data/35985678-0d79-46b4-9ed6-6f13308a1d24_2b8d7ca6b279f25493cd9fc72b1d8a69.csv'
output_file = 'data/market_prices_cleaned.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('Telangana')
cleaned_lines = [lines[0]]
for line in lines[1:]:
    if line.strip():
        cleaned_lines.append('Telangana' + line.strip())
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(cleaned_lines))

print(f"Cleaned data saved to {output_file}")

try:
    df = pd.read_csv(output_file)
    print("\nSample of cleaned data:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
except Exception as e:
    print(f"Error reading cleaned file: {e}")
