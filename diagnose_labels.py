"""Quick diagnostic: check pseudo-label quality and student issues."""
import pandas as pd

langs = ["nepali", "amharic", "pashto", "hausa"]

for lang in langs:
    path = f"preprocessed_data/{lang}_finetuned_teacher_labels/train.csv"
    try:
        df = pd.read_csv(path)
        print(f"\n{'='*60}")
        print(f"{lang.upper()} — {path}")
        print(f"{'='*60}")
        print(f"Columns: {list(df.columns)}")
        print(f"Rows: {len(df)}")
        if "teacher_summary" in df.columns:
            ts = df["teacher_summary"]
            print(f"teacher_summary null count: {ts.isna().sum()}")
            print(f"teacher_summary empty count: {(ts.astype(str).str.strip() == '').sum()}")
            print(f"teacher_summary avg length (chars): {ts.astype(str).str.len().mean():.0f}")
            print(f"Sample [0]: {str(ts.iloc[0])[:150]}")
            print(f"Sample [1]: {str(ts.iloc[1])[:150]}")
        else:
            print("WARNING: teacher_summary column MISSING!")
    except Exception as e:
        print(f"{lang}: ERROR — {e}")
