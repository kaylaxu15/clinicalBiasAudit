import pandas as pd


"""
Merging results for Llama and GPT CKD audits
"""

# Paths
left_path  = "/Users/kaylaxu/Downloads/clinicalBiasAudit/code/newer_results/responses/llama_3.3_200_ckd_diagnoses.csv"
right_path = "/Users/kaylaxu/Downloads/clinicalBiasAudit/code/newer_results/responses/gpt_fixed_ckd_responses.csv"

# Load
df_left = pd.read_csv(left_path)
df_right = pd.read_csv(right_path)

# 🔥 Choose your join key here
# Common columns often include: 'vignette', 'original_ethnicity', 'prompted_ethnicity'
join_key = "vignette"   # change to the column you're matching on

# LEFT JOIN
merged = df_left.merge(df_right, on=join_key, how="left", suffixes=("_gpt", "_llama"))

# Save
output_path = "/Users/kaylaxu/Downloads/clinicalBiasAudit/code/merged_ckd_results.csv"
merged.to_csv(output_path, index=False)

print("Done! Saved to:", output_path)
