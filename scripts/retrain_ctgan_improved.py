"""
Retrain CTGAN with improved settings for fraud detection
Use conditional training and more epochs
"""

import pandas as pd
import numpy as np
from ctgan import CTGAN
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RETRAINING CTGAN WITH IMPROVED SETTINGS FOR FRAUD DETECTION")
print("=" * 80)
print()

# Load data
print("Loading data...")
df = pd.read_csv('examples/csv/Bank_Transaction_Fraud_Detection.csv')
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['Is_Fraud'].mean()*100:.2f}%")
print()

# Prepare data for CTGAN
ctgan_data = df[[
    'Gender', 'Age', 'State', 'City', 'Account_Type',
    'Transaction_Amount', 'Transaction_Type', 'Merchant_Category',
    'Account_Balance', 'Transaction_Device', 'Transaction_Location',
    'Device_Type', 'Is_Fraud', 'Transaction_Currency'
]].copy()

print(f"CTGAN training data: {ctgan_data.shape}")
print()

# Define discrete columns
discrete_columns = [
    'Gender', 'State', 'City', 'Account_Type', 'Transaction_Type',
    'Merchant_Category', 'Transaction_Device', 'Transaction_Location',
    'Device_Type', 'Is_Fraud', 'Transaction_Currency'
]

print("=" * 80)
print("TRAINING CTGAN WITH IMPROVED SETTINGS")
print("=" * 80)
print("Settings:")
print("  - Epochs: 1000 (vs 600 before)")
print("  - Batch size: 500")
print("  - Generator dim: (256, 256) - larger for better learning")
print("  - Discriminator dim: (256, 256) - larger for better learning")
print("  - pac: 10 - helps with mode collapse")
print("  - Using conditional sampling on Is_Fraud")
print()

# Train CTGAN with improved settings
ctgan = CTGAN(
    epochs=1000,             # More epochs for better learning
    batch_size=500,
    generator_dim=(256, 256), # Larger network
    discriminator_dim=(256, 256),
    pac=10,                   # Helps prevent mode collapse
    verbose=True
)

print("Training CTGAN (this will take a while)...")
print()
ctgan.fit(ctgan_data, discrete_columns)

print()
print("=" * 80)
print("GENERATING SYNTHETIC DATA WITH CONDITIONAL SAMPLING")
print("=" * 80)
print()

# Generate synthetic data with conditional sampling on fraud
# Generate 50% fraud and 50% non-fraud to ensure balanced learning
n_fraud = 200000
n_normal = 200000

# Generate synthetic data with condition vectors
# CTGAN uses condition_column and condition_value parameters
print(f"Generating {n_fraud:,} fraud samples with conditional sampling...")
synthetic_fraud = ctgan.sample(
    n_fraud,
    condition_column='Is_Fraud',
    condition_value=1
)

print(f"Generating {n_normal:,} normal samples with conditional sampling...")
synthetic_normal = ctgan.sample(
    n_normal,
    condition_column='Is_Fraud',
    condition_value=0
)

# Combine
synthetic_data = pd.concat([synthetic_fraud, synthetic_normal], ignore_index=True)

# Shuffle
synthetic_data = synthetic_data.sample(frac=1, random_state=42).reset_index(drop=True)

print()
print(f"✓ Generated {len(synthetic_data):,} synthetic samples")
print("Synthetic fraud distribution:")
print(synthetic_data['Is_Fraud'].value_counts())
print(f"Synthetic fraud rate: {synthetic_data['Is_Fraud'].mean()*100:.2f}%")
print()

# Save
print("Saving synthetic data...")
synthetic_data.to_csv('synthetic_fraud_data_improved.csv', index=False)
file_size = len(synthetic_data) * len(synthetic_data.columns) * 8 / (1024*1024)
print(f"✓ Saved to: synthetic_fraud_data_improved.csv (~{file_size:.1f} MB)")
print()

# Save model
print("Saving CTGAN model...")
ctgan.save('ctgan_fraud_model_improved.pkl')
print("✓ Saved model to: ctgan_fraud_model_improved.pkl")
print()

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print()
print("Next steps:")
print("1. Replace 'synthetic_fraud_data.csv' with 'synthetic_fraud_data_improved.csv'")
print("2. Rerun 5th_fraud_synthetic_approach.py to test improved synthetic data")
print()
