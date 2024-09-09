import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data. Select results for analysis
df = pd.read_csv('../results/results_ru_en.csv')

# Choose metrics
metrics = ['BLEU', 'METEOR', 'BERT-F1']

# Normalize the selected metrics
for metric in metrics:
    df[f'{metric}_normalized'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

# Group by Service to calculate the mean normalized scores
normalized_scores = df.groupby('Service')[[f'{metric}_normalized' for metric in metrics]].mean()

# Rename the columns to remove "_normalized" suffix
normalized_scores.columns = metrics

# Plot Heatmap with selected metrics (for visualization purposes, not used in the paper)
plt.figure(figsize=(6, 4))  # Compact figure size
sns.heatmap(normalized_scores, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, annot_kws={"size": 9})
# Add title and labels with smaller font sizes
plt.title('Normalized Scores Heatmap by Translator', fontsize=12)
plt.ylabel('')
# Reduce tick size for clarity
plt.xticks(rotation=45, fontsize=9)
plt.yticks(fontsize=9)
# Tight layout for compactness
plt.tight_layout()
# Show heatmap
plt.show()

# Calculate Composite Scores with and without BERT-F1
normalized_scores['Composite Score (with BERT-F1)'] = normalized_scores.mean(axis=1)
# normalized_scores['Composite Score (without BERT-F1)'] = normalized_scores[['BLEU', 'METEOR']].mean(axis=1)

# Print Composite Scores
print(normalized_scores['Composite Score (with BERT-F1)'])
# print(normalized_scores['Composite Score (without BERT-F1)'])

# Plot Composite Score (with BERT-F1)
plt.figure(figsize=(6, 4))  # Compact figure size
ax1 = normalized_scores['Composite Score (with BERT-F1)'].sort_values(ascending=True).plot(kind='barh', color='skyblue')
# Add exact scores above each bar
for i, v in enumerate(normalized_scores['Composite Score (with BERT-F1)'].sort_values(ascending=True)):
    ax1.text(v - 0.07, i, f'{v:.3f}', color='black', va='center', fontsize=9)
# Add title and labels with smaller font sizes
plt.title('Composite Score by Service', fontsize=12)
ax1.set_ylabel('')
# Fix X-axis scale for consistency across plots (0 to (whatever the largest score is+))
ax1.set_xlim([0, 0.6])
# Reduce tick size for clarity
plt.xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=9)
plt.yticks(fontsize=9)
# Tight layout for compactness
plt.tight_layout()
# Show the plot
plt.show()

# Plot Composite Score (without BERT-F1)
# plt.figure(figsize=(6, 4))  # Compact figure size
# ax2 = normalized_scores['Composite Score (without BERT-F1)'].sort_values(ascending=True).plot(kind='barh', color='lightcoral')
# Add exact scores above each bar
# for i, v in enumerate(normalized_scores['Composite Score (without BERT-F1)'].sort_values(ascending=True)):
    # ax2.text(v - 0.05, i, f'{v:.3f}', color='white', va='center', fontsize=9)
# Add title and labels with smaller font sizes
# plt.title('Composite Score by Service (Excluding BERT-F1)', fontsize=12)
# ax2.set_ylabel('')
# Fix X-axis scale for consistency across plots (0 to 0.4)
# ax2.set_xlim([0, 0.4])
# Reduce tick size and number of ticks for clarity
# plt.xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4], fontsize=9)
# plt.yticks(fontsize=9)
# Tight layout for compactness
# plt.tight_layout()
# Show the plot
# plt.show()