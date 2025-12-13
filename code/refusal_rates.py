import matplotlib.pyplot as plt
import pandas as pd
from cmap import Colormap

cm = Colormap('colorbrewer:Pastel1')

# Change these parameters

model = "llama"
condition = "diabetes"

# Calculate refusal rates by ethnicity
results_df = pd.read_csv(f"final_merged/merged_{condition}_results.csv")

refusal_data = []

for ethnicity in results_df[f"prompted_ethnicity_{model}"].unique():
    ethnicity_df = results_df[results_df[f"prompted_ethnicity_{model}"] == ethnicity]
    total = len(ethnicity_df)
    refusals = ((ethnicity_df[f"risk_score_{model}"] == '') | 
                (ethnicity_df[f"risk_score_{model}"].isna())).sum()
    refusal_rate = refusals / total if total > 0 else 0
    refusal_data.append({'ethnicity': ethnicity, 'refusal_rate': refusal_rate, 'count': total})

refusal_df = pd.DataFrame(refusal_data)

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(refusal_df['ethnicity'], refusal_df['refusal_rate'], color=cm(range(len(refusal_df))))
plt.xlabel('Prompted Ethnicity')
plt.ylabel('Refusal Rate')
plt.title(f'{model.upper()} Refusal Rate Distribution by Ethnicity')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, max(refusal_data, key=lambda x: x['refusal_rate'])['refusal_rate']*2)
plt.tight_layout()
plt.show()

# Print summary statistics
print(refusal_df)