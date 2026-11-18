import pandas as pd
import numpy as np

df = pd.read_csv("data/landmarks.csv")

# Clean gesture labels
df["gesture"] = df["gesture"].str.strip()



print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicates:", df.duplicated().sum())

print("\nGesture distribution:")
print(df["gesture"].value_counts())

# Save cleaned dataset
df.to_csv("data/landmarks.csv", index=False)

