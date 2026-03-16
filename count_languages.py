import pandas as pd

# Path to the large CSV file
csv_path = "xlsum_all_train.csv"

# Read only the first row to get column names
header = pd.read_csv(csv_path, nrows=0)
print("Columns:", list(header.columns))

# Try to infer the language column name (commonly 'language', 'lang', etc.)
# You may need to adjust this if the column name is different
lang_col = None
for col in header.columns:
    if col.lower() in ["language", "lang"]:
        lang_col = col
        break

if lang_col is None:
    print("Could not find a language column. Please check the column names above and update the script.")
else:
    # Read only the language column in chunks to avoid memory issues
    unique_langs = set()
    for chunk in pd.read_csv(csv_path, usecols=[lang_col], chunksize=100000):
        unique_langs.update(chunk[lang_col].unique())
    print(f"Number of unique languages: {len(unique_langs)}")
    print("Languages:", sorted(unique_langs))

    # List of South Asian languages (ISO codes and names)
    south_asian = [
        "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "si", "ta", "te", "ur",
        "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Nepali", "Odia", "Punjabi", "Sinhala", "Tamil", "Telugu", "Urdu"
    ]
    sa_langs = [l for l in unique_langs if str(l).lower() in [x.lower() for x in south_asian]]
    print(f"South Asian languages present: {len(sa_langs)}")
    print("South Asian languages:", sorted(sa_langs))
