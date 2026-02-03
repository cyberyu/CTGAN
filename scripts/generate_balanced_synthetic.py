"""
Generate synthetic data and manually balance fraud rate to match real data (~5%)
"""
import pandas as pd
from ctgan import CTGAN
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('GENERATING SYNTHETIC DATA WITH BALANCED FRAUD RATE')
print('=' * 80)
print()

print('Loading trained CTGAN model...')
ctgan = CTGAN.load('ctgan_fraud_model_improved.pkl')
print('✓ Model loaded')
print()

# Generate large sample and filter to get desired fraud rate
print('Strategy: Generate large sample, then balance fraud/non-fraud manually')
print()

# Generate 600k samples to have enough fraud cases
n_total = 600000
print(f'Generating {n_total:,} synthetic samples...')
synthetic_large = ctgan.sample(n_total)

fraud_count = (synthetic_large['Is_Fraud'] == 1).sum()
normal_count = (synthetic_large['Is_Fraud'] == 0).sum()

print(f'Generated samples:')
print(f'  Total: {len(synthetic_large):,}')
print(f'  Fraud: {fraud_count:,} ({fraud_count/len(synthetic_large)*100:.2f}%)')
print(f'  Normal: {normal_count:,}')
print()

# Separate fraud and normal
fraud_samples = synthetic_large[synthetic_large['Is_Fraud'] == 1].copy()
normal_samples = synthetic_large[synthetic_large['Is_Fraud'] == 0].copy()

# Calculate how many we need for 5% fraud rate in 400k samples
target_total = 400000
target_fraud_rate = 0.05
target_fraud = int(target_total * target_fraud_rate)
target_normal = target_total - target_fraud

print(f'Target for 400k samples with 5% fraud:')
print(f'  Fraud needed: {target_fraud:,}')
print(f'  Normal needed: {target_normal:,}')
print()

# Sample to get target counts
if len(fraud_samples) >= target_fraud:
    fraud_final = fraud_samples.sample(n=target_fraud, random_state=42)
    print(f'✓ Selected {target_fraud:,} fraud samples')
else:
    print(f'⚠ Only {len(fraud_samples):,} fraud samples available, using all')
    # Oversample to reach target
    fraud_final = fraud_samples.sample(n=target_fraud, replace=True, random_state=42)
    print(f'  (oversampled with replacement)')

if len(normal_samples) >= target_normal:
    normal_final = normal_samples.sample(n=target_normal, random_state=42)
    print(f'✓ Selected {target_normal:,} normal samples')
else:
    print(f'⚠ Only {len(normal_samples):,} normal samples available')
    normal_final = normal_samples

# Combine and shuffle
synthetic_final = pd.concat([fraud_final, normal_final], ignore_index=True)
synthetic_final = synthetic_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Add branch_id
synthetic_final['branch_id'] = 1

print()
print('=' * 80)
print('FINAL SYNTHETIC DATA')
print('=' * 80)
print(f'Total samples: {len(synthetic_final):,}')
print(f'Fraud cases: {(synthetic_final["Is_Fraud"]==1).sum():,}')
print(f'Normal cases: {(synthetic_final["Is_Fraud"]==0).sum():,}')
print(f'Fraud rate: {synthetic_final["Is_Fraud"].mean()*100:.2f}%')
print()

# Save
synthetic_final.to_csv('synthetic_fraud_data.csv', index=False)
print('✓ Saved to: synthetic_fraud_data.csv')
print()
print('Next step: Run 5th_fraud_synthetic_approach.py to test performance')
print('=' * 80)
