import pandas as pd
import re 
import numpy as np
# df = pd.read_csv('brief_ckd_diagnoses.csv')
df = pd.read_csv('../newer_results/fixed_ckd_responses.csv')
pattern = re.compile(r"\**\s*Risk Score:\s*\**\s*(\d+)\s*/", re.IGNORECASE)

rows = []
count = 0

for index, row in df.iterrows():
    text = row["response"]
    if not isinstance(text, str):
        count += 1
        risk = None
    else:
        m = pattern.search(text)
        risk = int(m.group(1)) if m else None

    rows.append({
        "Risk Score": row['risk_score'],
        "Prompted Ethnicity": row["prompted_ethnicity"],
        "Original Ethnicity": row["original_ethnicity"]
    })

output = pd.DataFrame(rows)
output.to_csv("../newer_results/fixed_150_CKD_extracted_risk_scores.csv")

print("Count", count)

