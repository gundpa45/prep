import pandas as pd

df = pd.read_csv('data/student-mat.csv', sep=';')

print("=== DATASET SHAPE ===")
print(df.shape)

print("\n=== COLUMNS & DTYPES ===")
for col, dtype in df.dtypes.items():
    print(f"{col}: {dtype}")

print("\n=== MISSING VALUES ===")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")

print("\n=== SAMPLE ROWS ===")
print(df.head(2))

target = 'G3'
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nTarget Candidate:", target)
print("Numerical cols:", num_cols)
print("Categorical cols:", cat_cols)
