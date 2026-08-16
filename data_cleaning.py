"""
DATA CLEANING & PREPROCESSING
==============================
1. Import dataset & inspect structure
2. Identify missing values, duplicates, inconsistent entries
3. Clean: handle nulls, remove duplicates, fix data types
4. Prepare for analysis
5. (Bonus) Save cleaned dataset as new CSV

"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# -------------------------------------------------
# STEP 1: IMPORT DATASET & INSPECT STRUCTURE
# -------------------------------------------------
df = pd.read_csv("raw_dataset.csv")

print("=" * 60)
print("STEP 1: INITIAL INSPECTION")
print("=" * 60)
print("\nShape (rows, columns):", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nData types & non-null counts:")
print(df.info())
print("\nSummary statistics:\n", df.describe(include="all"))

# -------------------------------------------------
# STEP 2: IDENTIFY ISSUES
# -------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: IDENTIFYING ISSUES")
print("=" * 60)

# Missing values (including 'N/A' text and blank strings, which pandas
# does NOT catch automatically as NaN)
df.replace(["N/A", "n/a", "", " ", "  "], np.nan, inplace=True)
print("\nMissing values per column:\n", df.isnull().sum())

# Duplicate rows (exact duplicates)
print("\nExact duplicate rows:", df.duplicated().sum())

# Duplicate rows based on a business key (email is a better duplicate
# indicator than a full-row match, since minor casing differs)
print("Duplicate emails (case-insensitive):",
      df["email"].str.lower().duplicated().sum())

# Inconsistent categorical entries
print("\nUnique 'gender' values found:", df["gender"].dropna().unique())

# Inconsistent date formats (mixed separators / orders)
print("Unique 'signup_date' formats sample:", df["signup_date"].dropna().unique())

# Wrong data types
print("\nCurrent dtypes:\n", df.dtypes)

# -------------------------------------------------
# STEP 3: CLEAN THE DATA
# -------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: CLEANING")
print("=" * 60)

# 3a. Standardize text fields: trim whitespace, fix casing
df["name"] = df["name"].str.strip().str.title()
df["city"] = df["city"].str.strip().str.title()
df["email"] = df["email"].str.strip().str.lower()

# 3b. Standardize the 'gender' column into consistent categories
gender_map = {
    "m": "Male", "male": "Male",
    "f": "Female", "female": "Female",
}
df["gender"] = df["gender"].str.strip().str.lower().map(gender_map)

# 3c. Fix data types
df["age"] = pd.to_numeric(df["age"], errors="coerce")           # text -> numeric
df["purchase_amount"] = pd.to_numeric(df["purchase_amount"], errors="coerce")
df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")  # mixed formats -> datetime
df["customer_id"] = df["customer_id"].astype("Int64")

# 3d. Remove duplicate records
#     Use the business key (email) since it identifies the same person
#     even when name casing/spacing differs
before = len(df)
df = df.drop_duplicates(subset=["email"], keep="first")
print(f"\nRemoved {before - len(df)} duplicate record(s) based on email")

# 3e. Handle missing values
#     - Drop rows missing a name or email (can't identify the customer)
#     - Fill missing numeric values with the column median
#     - Fill missing gender with 'Unknown' rather than dropping the row
df = df.dropna(subset=["name", "email"])
df["age"] = df["age"].fillna(df["age"].median())
df["purchase_amount"] = df["purchase_amount"].fillna(df["purchase_amount"].median())
df["gender"] = df["gender"].fillna("Unknown")
df["signup_date"] = df["signup_date"].ffill()

# 3f. Reset index after dropping rows
df = df.reset_index(drop=True)

# -------------------------------------------------
# STEP 4: PREPARE FOR ANALYSIS
# -------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: FINAL CLEANED DATASET")
print("=" * 60)
print("\nShape after cleaning:", df.shape)
print("\nFinal dtypes:\n", df.dtypes)
print("\nRemaining missing values:\n", df.isnull().sum())
print("\nCleaned data:\n", df)

# -------------------------------------------------
# STEP 5 (BONUS): SAVE CLEANED DATASET
# -------------------------------------------------
df.to_csv("cleaned_dataset.csv", index=False)
print("\nSaved cleaned dataset -> cleaned_dataset.csv")
