import pandas as pd
import re 
# df = pd.read_csv('brief_ckd_diagnoses.csv')
df = pd.read_csv('../2nd_attempt_100_ckd_diagnoses.csv')

pattern = re.compile(r"\**\s*Risk Score:\s*\**\s*(\d+)\s*/", re.IGNORECASE)

rows = []
for index, row in df.iterrows():
    text = row["response"]
    m = pattern.search(text)
    risk = int(m.group(1)) if m else None

    rows.append({
        "Risk Score": risk,
        "Prompted Ethnicity": row["prompted_ethnicity"],
        "Original Ethnicity": row["original_ethnicity"]
    })

output = pd.DataFrame(rows)
output.to_csv("../results/100_CKD_extracted_risk_scores.csv")
    

