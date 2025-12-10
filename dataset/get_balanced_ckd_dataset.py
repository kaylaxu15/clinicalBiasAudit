import pandas as pd

df = pd.read_csv("../dataset/ckd_dataset.csv")

# Separate the dataframe by Diagnosis value
diagnosis_0 = df[df['Diagnosis'] == 0]
diagnosis_1 = df[df['Diagnosis'] == 1]

# Sample 75 rows from each group
# Use random_state for reproducibility
diagnosis_0_sampled = diagnosis_0.sample(n=100, random_state=42)
diagnosis_1_sampled = diagnosis_1.sample(n=100, random_state=42)

# Combine the sampled dataframes
df_balanced = pd.concat([diagnosis_0_sampled, diagnosis_1_sampled], ignore_index=True)

df_balanced.to_csv("../dataset/ckd_dataset_balanced.csv", index=False)
