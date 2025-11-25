import pandas as pd

# df = pd.read_csv('brief_ckd_diagnoses.csv')
df = pd.read_csv('diabetes_responses_total.csv')

for index, row in df.iterrows():
    print(f"{row["response"][:50]} for {row["prompted_ethnicity"]} with original patient being {row["original_ethnicity"]}")
    

