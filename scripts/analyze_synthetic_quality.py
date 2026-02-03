"""
Analyze synthetic data quality for fraud detection
Check if fraud patterns are preserved
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SYNTHETIC DATA QUALITY ANALYSIS FOR FRAUD DETECTION")
print("=" * 80)
print()

# Load real and synthetic data
print("Loading data...")
real_data = pd.read_csv('examples/csv/Bank_Transaction_Fraud_Detection.csv')
synthetic_data = pd.read_csv('synthetic_fraud_data.csv')

print(f"Real data: {len(real_data):,} records")
print(f"Synthetic data: {len(synthetic_data):,} records")
print()

# Fraud rates
print("Fraud Rates:")
print(f"  Real:      {real_data['Is_Fraud'].mean()*100:.2f}%")
print(f"  Synthetic: {synthetic_data['Is_Fraud'].mean()*100:.2f}%")
print()

# Check numeric features for fraud vs non-fraud
numeric_cols = ['Age', 'Transaction_Amount', 'Account_Balance']

print("=" * 80)
print("FRAUD PATTERN ANALYSIS - Numeric Features")
print("=" * 80)
print()

for col in numeric_cols:
    print(f"\n{col}:")
    print("  Real Data:")
    real_fraud = real_data[real_data['Is_Fraud'] == 1][col]
    real_normal = real_data[real_data['Is_Fraud'] == 0][col]
    print(f"    Fraud:     mean={real_fraud.mean():.2f}, std={real_fraud.std():.2f}")
    print(f"    Normal:    mean={real_normal.mean():.2f}, std={real_normal.std():.2f}")
    print(f"    Difference: {abs(real_fraud.mean() - real_normal.mean()):.2f}")
    
    print("  Synthetic Data:")
    syn_fraud = synthetic_data[synthetic_data['Is_Fraud'] == 1][col]
    syn_normal = synthetic_data[synthetic_data['Is_Fraud'] == 0][col]
    print(f"    Fraud:     mean={syn_fraud.mean():.2f}, std={syn_fraud.std():.2f}")
    print(f"    Normal:    mean={syn_normal.mean():.2f}, std={syn_normal.std():.2f}")
    print(f"    Difference: {abs(syn_fraud.mean() - syn_normal.mean()):.2f}")

# Check categorical patterns
print()
print("=" * 80)
print("FRAUD PATTERN ANALYSIS - Categorical Features")
print("=" * 80)
print()

cat_cols = ['Gender', 'Transaction_Type', 'Device_Type', 'Transaction_Device']

for col in cat_cols:
    print(f"\n{col} - Fraud Rate by Category:")
    print("  Real Data:")
    real_fraud_rate = real_data.groupby(col)['Is_Fraud'].mean().sort_values(ascending=False)
    for cat, rate in real_fraud_rate.head(5).items():
        print(f"    {cat}: {rate*100:.2f}%")
    
    print("  Synthetic Data:")
    syn_fraud_rate = synthetic_data.groupby(col)['Is_Fraud'].mean().sort_values(ascending=False)
    for cat, rate in syn_fraud_rate.head(5).items():
        print(f"    {cat}: {rate*100:.2f}%")

# Feature correlation with fraud
print()
print("=" * 80)
print("FEATURE IMPORTANCE - Correlation with Fraud")
print("=" * 80)
print()

# Encode categorical features
real_encoded = real_data.copy()
syn_encoded = synthetic_data.copy()

# Drop UUID columns first
uuid_cols = ['Transaction_ID', 'Customer_ID']
for col in uuid_cols:
    if col in real_encoded.columns:
        real_encoded = real_encoded.drop(col, axis=1)
    if col in syn_encoded.columns:
        syn_encoded = syn_encoded.drop(col, axis=1)

encode_cols = ['Gender', 'State', 'City', 'Account_Type', 'Transaction_Type',
               'Merchant_Category', 'Transaction_Device', 'Transaction_Location',
               'Device_Type', 'Transaction_Currency']

for col in encode_cols:
    if col in real_encoded.columns:
        le = LabelEncoder()
        real_encoded[col] = le.fit_transform(real_encoded[col].astype(str))
        
        # Encode synthetic with same mapping
        syn_encoded[col] = syn_encoded[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Calculate correlations (numeric columns only)
print("Top 10 features correlated with fraud:")
print("\nReal Data:")
real_numeric = real_encoded.select_dtypes(include=[np.number])
real_corr = real_numeric.corr()['Is_Fraud'].abs().sort_values(ascending=False)
for feat, corr in list(real_corr.items())[1:11]:  # Skip Is_Fraud itself
    print(f"  {feat}: {corr:.4f}")

print("\nSynthetic Data:")
syn_numeric = syn_encoded.select_dtypes(include=[np.number])
syn_corr = syn_numeric.corr()['Is_Fraud'].abs().sort_values(ascending=False)
for feat, corr in list(syn_corr.items())[1:11]:
    print(f"  {feat}: {corr:.4f}")

# Check if synthetic data preserves discriminative power
print()
print("=" * 80)
print("DISCRIMINATIVE POWER CHECK")
print("=" * 80)
print()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

# Quick test: train on synthetic, test on real
X_syn = syn_encoded.drop(['Is_Fraud', 'Bank_Branch', 'branch_id'], axis=1, errors='ignore')
y_syn = syn_encoded['Is_Fraud']

X_real = real_encoded.drop(['Is_Fraud', 'Bank_Branch', 'branch_id'], axis=1, errors='ignore')
y_real = real_encoded['Is_Fraud']

# Align columns
common_cols = list(set(X_syn.columns) & set(X_real.columns))
X_syn = X_syn[common_cols]
X_real = X_real[common_cols]

# Train on synthetic
X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(
    X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn
)

print("Training Random Forest on synthetic data...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_syn_train, y_syn_train)

# Test on synthetic
y_syn_pred = rf.predict(X_syn_test)
syn_auc = roc_auc_score(y_syn_test, rf.predict_proba(X_syn_test)[:, 1])
syn_f1 = f1_score(y_syn_test, y_syn_pred)

print(f"Test on Synthetic: AUC={syn_auc:.4f}, F1={syn_f1:.4f}")

# Test on real
y_real_pred = rf.predict(X_real)
real_auc = roc_auc_score(y_real, rf.predict_proba(X_real)[:, 1])
real_f1 = f1_score(y_real, y_real_pred)

print(f"Test on Real:      AUC={real_auc:.4f}, F1={real_f1:.4f}")
print()

if syn_auc > 0.6 and real_auc < 0.55:
    print("⚠ WARNING: Synthetic data has patterns but they don't transfer to real data!")
    print("   The GAN may have learned spurious correlations.")
elif syn_auc < 0.55:
    print("⚠ WARNING: Synthetic data lacks discriminative fraud patterns!")
    print("   The GAN may not be preserving fraud characteristics well.")
else:
    print("✓ Synthetic data preserves some fraud patterns.")

print()
print("=" * 80)
