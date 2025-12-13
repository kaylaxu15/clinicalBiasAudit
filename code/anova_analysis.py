import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

"""
ANOVA and posthoc Tukey tests
"""
condition = "ckd"
ethnicity_map = {0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"}

model = "gpt" # llama or gpt

results_df = pd.read_csv(f"final_merged/merged_{condition}_results.csv")
dataset_df = pd.read_csv(f"newer_results/dataset/{condition}_dataset.csv")
ground_truth_df = pd.read_csv(f"ground_truth/{condition}_ground_truths.csv")
ground_truth_col = "ground_truth"

# Merge ground_truth_df into results_df
results_df = results_df.merge(
    ground_truth_df[["vignette", "ground_truth"]],
    on="vignette",
    how="left"
)
for i, p in enumerate(results_df[f"prompted_ethnicity_{model}"]):
    if not isinstance(p, str):
        print("HERE", i)

results_df = results_df[(results_df[f"risk_score_{model}"] != '') & (results_df[f"risk_score_{model}"].notna())]
# for converting dataset columns to indicated ethnicity of patient record
eth_map = {
    "race:AfricanAmerican": "Black",
    "race:Hispanic": "Hispanic",
    "race:Asian": "Asian",
    "race:Caucasian": "Caucasian"
}

original_ethnicities = results_df[f"original_ethnicity_{model}"]
prompted_race = results_df[f"prompted_ethnicity_{model}"]

groundTruth = [float(i) for i in results_df["ground_truth"]]
risk_scores = results_df[f"risk_score_{model}"]

print(len(original_ethnicities))
print(results_df["ground_truth"])
print((results_df["ground_truth"] == 0).sum())  # Count 0s
print((results_df["ground_truth"] == 1).sum())  # Count 1s

data = {
    "OriginalPatient": original_ethnicities,
    
    "PromptedRace": prompted_race,
    
    "RiskScore": risk_scores,
    f"True_{condition}": groundTruth
}

label = condition
df = pd.DataFrame(data)
# Fit the two-way ANOVA model
model = ols(f"RiskScore ~ C(PromptedRace) + C(True_{label}) + C(PromptedRace):C(True_{label})", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\n=== ANOVA Results ===")
print(anova_table)

# Post-hoc Tukey test
print("\n=== Tukey HSD Post-hoc Test ===")
tukey = pairwise_tukeyhsd(endog=df['RiskScore'],
                          groups=df['PromptedRace'],
                          alpha=0.05)
print(tukey)

# plot boxplot for risk scores
plt.figure(figsize=(10,6))
sns.boxplot(
    x="PromptedRace", 
    y="RiskScore", 
    hue=f"True_{label}", 
    data=df,
    palette={0: "skyblue", 1: "orange"},
    hue_order=[0, 1]
)

model_name = "GPT_OSS"
plt.title(f"Risk Scores by Prompted Race and True {condition.capitalize()} Status for {model_name}")
plt.xlabel("Prompted Race")
plt.ylabel("Risk Score")

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles, 
    [f"No {label} (0)", f"Has {label} (1)"], 
    title=f"True {label}",
    loc='lower right'  # Add this line
)

plt.tight_layout()
plt.show()