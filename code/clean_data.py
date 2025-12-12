import pandas as pd


"""
This module was to check which risk score cells were empty (sometimes the model either refused to respond or 
gave a blank response, so distinguishing between these two was necessary).

"""
# Load the dataframe
df = pd.read_csv('/Users/kaylaxu/Downloads/clinicalBiasAudit/code/final_merged/merged_diabetes_results.csv')

# Check for empty cells in risk_score column for both dataframes
# Assuming risk_score_gpt and risk_score_llama are separate dataframes or separate columns

# If they are columns in the same dataframe:
if 'risk_score_gpt' in df.columns:
    gpt_empty = df['risk_score_gpt'].isna().sum()
    print(f"Empty cells in 'risk_score_gpt': {gpt_empty}")
    print(f"Total rows: {len(df)}")
    print(f"Percentage empty: {(gpt_empty/len(df))*100:.2f}%\n")
    
    if gpt_empty > 0:
        print("Rows with empty 'risk_score_gpt':")
        empty_gpt_df = df[df['risk_score_gpt'].isna()]
        print(empty_gpt_df.index.tolist())
        print("\nVignettes for empty risk_score_gpt:")
        for idx, row in empty_gpt_df.iterrows():
            print(f"\n--- Row {idx} ---")
            if 'vignette' in df.columns:
                print(row['vignette'][:200])
            else:
                print("No 'vignette' column found")
        print()

if 'risk_score_llama' in df.columns:
    llama_empty = df['risk_score_llama'].isna().sum()
    print(f"Empty cells in 'risk_score_llama': {llama_empty}")
    print(f"Total rows: {len(df)}")
    print(f"Percentage empty: {(llama_empty/len(df))*100:.2f}%\n")
    
    if llama_empty > 0:
        print("Rows with empty 'risk_score_llama':")
        empty_llama_df = df[df['risk_score_llama'].isna()]
        print(empty_llama_df.index.tolist())
        print("\nVignettes for empty risk_score_llama:")
        for idx, row in empty_llama_df.iterrows():
            print(f"\n--- Row {idx} ---")
            if 'vignette' in df.columns:
                print(row['vignette'][:200])
            else:
                print("No 'vignette' column found")
        print()