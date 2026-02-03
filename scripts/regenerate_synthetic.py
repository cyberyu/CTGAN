"""
Regenerate synthetic data with proper 5% fraud rate
"""
import pandas as pd
from ctgan import CTGAN
import warnings
warnings.filterwarnings('ignore')

print('Loading trained CTGAN model...')
ctgan = CTGAN.load('ctgan_fraud_model_improved.pkl')

# Generate with proper fraud rate (~5% like real data)
n_fraud = 20000
n_normal = 380000

print(f'Generating {n_fraud:,} fraud samples...')
syn_fraud = ctgan.sample(n_fraud, condition_column='Is_Fraud', condition_value=1)

print(f'Generating {n_normal:,} normal samples...')
syn_normal = ctgan.sample(n_normal, condition_column='Is_Fraud', condition_value=0)

# Combine and shuffle
synthetic = pd.concat([syn_fraud, syn_normal], ignore_index=True)
synthetic = synthetic.sample(frac=1, random_state=42).reset_index(drop=True)

# Add branch_id
synthetic['branch_id'] = 1

print(f'\nFinal synthetic data:')
print(f'  Total: {len(synthetic):,}')
print(f'  Fraud rate: {synthetic["Is_Fraud"].mean()*100:.2f}%')
print(f'  Fraud count: {(synthetic["Is_Fraud"]==1).sum():,}')

# Save
synthetic.to_csv('synthetic_fraud_data.csv', index=False)
print(f'\n✓ Saved to synthetic_fraud_data.csv')
