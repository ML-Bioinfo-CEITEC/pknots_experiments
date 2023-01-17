import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Load positive and negative dataset
df_pos = pd.read_csv('SPOUT_knotted.csv', sep=';').sample(frac=1, random_state=42)
df_neg = pd.read_csv('Rossmann_unknotted.csv').sample(frac=1, random_state=42)

# Preprocess datasets
df_pos['label'] = 1
df_neg['label'] = 0

df_pos['len'] = df_pos['seq'].str.len()
df_neg['len'] = df_neg['seq'].str.len()

# Sequence length is considered in bins of size 5
df_pos['len'] = df_pos['len'].apply(lambda x: x - x % 5)
df_neg['len'] = df_neg['len'].apply(lambda x: x - x % 5)

df_pos = df_pos.sort_values(by=['len'])
df_neg = df_neg.sort_values(by=['len'])

df_pos = df_pos.reset_index(drop=True)
df_neg = df_neg.reset_index(drop=True)

pos = 0
neg = 0

df = pd.DataFrame({'seq': [], 'label': []})

# Make new dataset with similar positive and negative sequence length distribution

while pos < len(df_pos) and neg < len(df_neg):
    if df_pos['len'][pos] == df_neg['len'][neg]:
        tmp = pd.DataFrame({'seq': [df_pos['seq'][pos]], 'label': [df_pos['label'][pos]]})
        df = pd.concat([df, tmp], ignore_index=True)
        tmp = pd.DataFrame({'seq': [df_neg['seq'][neg]], 'label': [df_neg['label'][neg]]})
        df = pd.concat([df, tmp], ignore_index=True)
        pos += 1
        neg += 1
    elif df_pos['len'][pos] > df_neg['len'][neg]:
        neg += 1
    else:
        pos += 1

print(df)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('Knots_train_set.csv', index=False)
test_df.to_csv('Knots_test_set.csv', index=False)

seqs_pos = df[df['label'] == 1]['seq'].str.len().tolist()
seqs_neg = df[df['label'] == 0]['seq'].str.len().tolist()

plt.hist(seqs_pos, bins=np.linspace(0, 1000, 100), alpha=0.5, label='knotted')

plt.hist(seqs_neg, bins=np.linspace(0, 1000, 100), alpha=0.5, color='orange', label='unknotted')

plt.legend()
plt.xlabel('Sequence length')
plt.ylabel('count')

plt.show()
