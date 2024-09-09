import pandas as pd

# Load the data. Select results for analysis
df = pd.read_csv('../results/results_no_en.csv')

# Calculate the average scores for BLEU, METEOR, and BERT-F1 by domain
score_avg_per_domain = df.groupby('Domain').agg({
    'BLEU': 'mean',
    'METEOR': 'mean',
    'BERT-F1': 'mean'
}).reset_index()

# Sort the DataFrame by each metric
bleu_sorted = score_avg_per_domain.sort_values(by='BLEU', ascending=False).reset_index(drop=True)
meteor_sorted = score_avg_per_domain.sort_values(by='METEOR', ascending=False).reset_index(drop=True)
bert_f1_sorted = score_avg_per_domain.sort_values(by='BERT-F1', ascending=False).reset_index(drop=True)

# Create the final DataFrame to display the results as shown in the image
final_df = pd.DataFrame({
    'Domain (BLEU)': bleu_sorted['Domain'],
    'BLEU Average': bleu_sorted['BLEU'],
    'Domain (METEOR)': meteor_sorted['Domain'],
    'METEOR Average': meteor_sorted['METEOR'],
    'Domain (BERT-F1)': bert_f1_sorted['Domain'],
    'BERT-F1 Average': bert_f1_sorted['BERT-F1'],
})

# Display the final table
print(final_df)

# Save the result. Make sure the filename matches the dataset
final_df.to_csv('../results/analysis/average_scores_per_domain (no-en).csv', index=False)