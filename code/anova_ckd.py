import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

# ----------------------------
# Prepare the data
# ----------------------------

original_ethnicities = pd.read_csv("brief_ckd_diagnoses.csv")["original_ethnicity"]
groundTruth = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0])
groundTruth = np.repeat(groundTruth, 4)

data = {
    "OriginalPatient": [o for o in original_ethnicities if o != "Other"],
    
    "PromptedRace": [
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic",
        "Caucasian","Asian","Black","Hispanic"
    ],
    
    "RiskScore": [
        85,88,78,85,
        73,85,73,65,
        85,78,85,86,
        85,78,82,78,
        75,78,82,85,
        85,78,78,78,
        85,85,85,85,
        90,78,78,84,
        78,88,87,78,
        80,83,85,78,
        85,66,70,90,
        78,78,84,85,
        38,84,85,78,
        65,75,55,45
    ],
    "TrueCKD": groundTruth
}

df = pd.DataFrame(data)

# Fit the two-way ANOVA model
model = ols("RiskScore ~ C(PromptedRace) + C(TrueCKD) + C(PromptedRace):C(TrueCKD)", data=df).fit()
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
    hue="TrueCKD", 
    data=df,
    palette={0: "skyblue", 1: "orange"},
    hue_order=[0, 1]
)
plt.title("Risk Scores by Prompted Race and True CKD Status")
plt.xlabel("Prompted Race")
plt.ylabel("Risk Score")

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ["No CKD (0)", "Has CKD (1)"], title="True CKD")

plt.tight_layout()
plt.show()