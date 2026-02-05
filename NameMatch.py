import pandas as pd

# =========================
# FILES
# =========================

CONTACT_FILE = "Credit_Union_Contact_List__Credit_Unions_1B10B_PG.xlsx"
NCUA_FILE = "NCUA Data end 2025.xlsx"
OUTPUT_FILE = "Credit_Union_Merged_Asset_Only.xlsx"

# =========================
# LOAD DATA
# =========================

contacts = pd.read_excel(CONTACT_FILE)
ncua = pd.read_excel(NCUA_FILE)

# =========================
# NORMALIZE ASSETS
# =========================

contacts["Assets"] = (
    contacts["Assets"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype("int64")
)

ncua["TotalAssets"] = (
    ncua["TotalAssets"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype("int64")
)

# =========================
# MERGE ON ASSETS ONLY
# =========================

merged = contacts.merge(
    ncua,
    left_on="Assets",
    right_on="TotalAssets",
    how="left",
    suffixes=("_Contact", "_NCUA"),
    indicator=True
)

# =========================
# CLASSIFY RESULTS
# =========================

# Count how many NCUA rows match each contact
match_counts = (
    merged
    .groupby(merged.index)
    .size()
)

merged["MatchCount"] = merged.index.map(match_counts)

merged["MatchStatus"] = merged["MatchCount"].apply(
    lambda x: "Match" if x == 1 else
              "Ambiguous" if x > 1 else
              "No Match"
)

# =========================
# OUTPUT
# =========================

merged.to_excel(OUTPUT_FILE, index=False)

print("✅ Asset-only merge complete")
print("✅ Output file:", OUTPUT_FILE)# Install dependencies (run this first in a Colab cell)
!pip install pandas rapidfuzz chardet --quiet

# Import libraries
import pandas as pd
from rapidfuzz import fuzz
from google.colab import files
import chardet

# 1️⃣ Upload CSV
print("Please upload your CSV file with two columns of names.")
uploaded = files.upload()

filename = list(uploaded.keys())[0]
print(f"Loaded file: {filename}")

# 2️⃣ Detect file encoding
with open(filename, "rb") as f:
    encoding = chardet.detect(f.read())["encoding"]

print(f"Detected encoding: {encoding}")

# 3️⃣ Load CSV (safe fallback)
try:
    df = pd.read_csv(filename, encoding=encoding)
except UnicodeDecodeError:
    print("Fallback to latin1 encoding.")
    df = pd.read_csv(filename, encoding="latin1")

# 4️⃣ Validate columns
if df.shape[1] < 2:
    raise ValueError("CSV must have at least two columns for name comparison.")

col1, col2 = df.columns[:2]
print(f"Comparing columns: '{col1}' vs '{col2}'")

# 5️⃣ Preserve original values for output
df[f"{col1}_orig"] = df[col1]
df[f"{col2}_orig"] = df[col2]

# 6️⃣ Normalize for matching
def normalize_name(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

df["_norm1"] = df[col1].apply(normalize_name)
df["_norm2"] = df[col2].apply(normalize_name)

# 7️⃣ Fuzzy match
df["Score"] = df.apply(
    lambda x: fuzz.token_sort_ratio(x["_norm1"], x["_norm2"]),
    axis=1
)

threshold = 80
df["Match"] = df["Score"].apply(lambda x: "Yes" if x >= threshold else "No")

# 8️⃣ Title-case output names
def title_case(x):
    if pd.isna(x):
        return ""
    return str(x).strip().title()

df[col1] = df[f"{col1}_orig"].apply(title_case)
df[col2] = df[f"{col2}_orig"].apply(title_case)

# 9️⃣ Clean up helper columns
df.drop(columns=[f"{col1}_orig", f"{col2}_orig", "_norm1", "_norm2"], inplace=True)

# 🔟 Save output
output_filename = "fuzzy_matched.csv"
df.to_csv(output_filename, index=False, encoding="utf-8")

print(f"Done! Output saved as {output_filename}")
files.download(output_filename)
