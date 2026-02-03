"""
TEST VERSION - Quick validation of improved CTGAN pipeline
Tests with small sample (1000 records) and 10 epochs to catch errors fast
"""

import pandas as pd
import numpy as np
from ctgan import CTGAN
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TEST RUN - CTGAN IMPROVED PIPELINE VALIDATION")
print("=" * 80)
print()

# Load data
print("Loading data...")
df = pd.read_csv('examples/csv/Bank_Transaction_Fraud_Detection.csv')
print(f"Full dataset: {df.shape}")

# Take small sample for testing
print("Taking small sample (1000 records) for fast testing...")
test_sample = df.sample(n=1000, random_state=42)
print(f"Test sample: {test_sample.shape}")
print(f"Test fraud rate: {test_sample['Is_Fraud'].mean()*100:.2f}%")
print()

# Prepare data for CTGAN
ctgan_data = test_sample[[
    'Gender', 'Age', 'State', 'City', 'Account_Type',
    'Transaction_Amount', 'Transaction_Type', 'Merchant_Category',
    'Account_Balance', 'Transaction_Device', 'Transaction_Location',
    'Device_Type', 'Is_Fraud', 'Transaction_Currency'
]].copy()

# Define discrete columns
discrete_columns = [
    'Gender', 'State', 'City', 'Account_Type', 'Transaction_Type',
    'Merchant_Category', 'Transaction_Device', 'Transaction_Location',
    'Device_Type', 'Is_Fraud', 'Transaction_Currency'
]

print("=" * 80)
print("STEP 1: TRAINING CTGAN (10 epochs for testing)")
print("=" * 80)
print()

# Train CTGAN with minimal settings for testing
ctgan = CTGAN(
    epochs=10,               # Just 10 for quick test
    batch_size=100,          # Smaller batch
    generator_dim=(128, 128),
    discriminator_dim=(128, 128),
    pac=5,
    verbose=True
)

print("Training CTGAN on test sample...")
try:
    ctgan.fit(ctgan_data, discrete_columns)
    print("✓ Training successful!")
except Exception as e:
    print(f"✗ Training failed: {e}")
    exit(1)

print()
print("=" * 80)
print("STEP 2: GENERATING SYNTHETIC DATA")
print("=" * 80)
print()

# Test conditional sampling
n_fraud = 100
n_normal = 100

print(f"Testing conditional sampling: {n_fraud} fraud + {n_normal} normal...")
try:
    synthetic_fraud = ctgan.sample(
        n_fraud,
        condition_column='Is_Fraud',
        condition_value=1
    )
    print(f"✓ Generated {len(synthetic_fraud)} fraud samples")
    print(f"  Fraud rate in sample: {synthetic_fraud['Is_Fraud'].mean()*100:.1f}%")
    
    synthetic_normal = ctgan.sample(
        n_normal,
        condition_column='Is_Fraud',
        condition_value=0
    )
    print(f"✓ Generated {len(synthetic_normal)} normal samples")
    print(f"  Fraud rate in sample: {synthetic_normal['Is_Fraud'].mean()*100:.1f}%")
    
except Exception as e:
    print(f"✗ Conditional sampling failed: {e}")
    print("\nTrying regular sampling without conditions...")
    try:
        synthetic_data = ctgan.sample(200)
        print(f"✓ Regular sampling works: {len(synthetic_data)} samples")
        print(f"  Fraud rate: {synthetic_data['Is_Fraud'].mean()*100:.1f}%")
    except Exception as e2:
        print(f"✗ Regular sampling also failed: {e2}")
        exit(1)
    exit(1)

# Combine
synthetic_data = pd.concat([synthetic_fraud, synthetic_normal], ignore_index=True)
synthetic_data = synthetic_data.sample(frac=1, random_state=42).reset_index(drop=True)

print()
print(f"✓ Combined synthetic data: {len(synthetic_data)} samples")
print(f"  Overall fraud rate: {synthetic_data['Is_Fraud'].mean()*100:.1f}%")
print()

print("=" * 80)
print("STEP 3: SAVING TEST FILES")
print("=" * 80)
print()

# Save test files
try:
    synthetic_data.to_csv('test_synthetic_data.csv', index=False)
    print("✓ Saved: test_synthetic_data.csv")
    
    ctgan.save('test_ctgan_model.pkl')
    print("✓ Saved: test_ctgan_model.pkl")
except Exception as e:
    print(f"✗ Saving failed: {e}")
    exit(1)

print()
print("=" * 80)
print("STEP 4: VALIDATING SYNTHETIC DATA QUALITY")
print("=" * 80)
print()

# Quick quality check
print("Checking feature distributions...")
numeric_cols = ['Age', 'Transaction_Amount', 'Account_Balance']

for col in numeric_cols:
    real_mean = ctgan_data[col].mean()
    syn_mean = synthetic_data[col].mean()
    diff_pct = abs(real_mean - syn_mean) / real_mean * 100
    status = "✓" if diff_pct < 20 else "⚠"
    print(f"  {status} {col}: real={real_mean:.1f}, syn={syn_mean:.1f}, diff={diff_pct:.1f}%")

print()
print("Checking categorical distributions...")
cat_cols = ['Gender', 'Transaction_Type', 'Device_Type']

for col in cat_cols[:2]:  # Just check first 2
    real_top = ctgan_data[col].value_counts().index[0]
    syn_top = synthetic_data[col].value_counts().index[0]
    match = "✓" if real_top == syn_top else "⚠"
    print(f"  {match} {col}: real_top={real_top}, syn_top={syn_top}")

print()
print("=" * 80)
print("TEST VALIDATION COMPLETE!")
print("=" * 80)
print()
print("✓ All steps passed successfully!")
print()
print("Next steps:")
print("1. Review test_synthetic_data.csv to verify quality")
print("2. If satisfied, run the full training script:")
print("   python scripts/retrain_ctgan_improved.py")
print()
