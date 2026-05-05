import pandas as pd

def build_applicant_from_dict(d, expected_cols):
    df = pd.DataFrame([d])

    for c in df.select_dtypes(include=['object', 'str']).columns:
        df[c] = df[c].str.strip() # col values

    # ensure if column exists
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Input Columns: {missing}")
    return df[expected_cols]



