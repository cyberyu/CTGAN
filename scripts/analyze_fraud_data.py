"""Quick analysis of Bank Transaction Fraud Detection dataset"""
import pandas as pd
import numpy as np

print("Loading Bank Transaction Fraud Detection dataset...")
df = pd.read_csv('examples/csv/Bank_Transaction_Fraud_Detection.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")

print(f"\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nData Types:")
print(df.dtypes)

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nTarget Variable (Is_Fraud) Distribution:")
print(df['Is_Fraud'].value_counts())
print(f"Fraud rate: {df['Is_Fraud'].mean()*100:.2f}%")

print(f"\n" + "="*80)
print("MERCHANT_ID ANALYSIS")
print("="*80)

# Check if Merchant_ID column exists
merchant_col = None
for col in df.columns:
    if 'merchant' in col.lower() or 'merchant_id' in col.lower():
        merchant_col = col
        break

if merchant_col:
    print(f"\nMerchant column found: '{merchant_col}'")
    
    unique_merchants = df[merchant_col].nunique()
    total_rows = len(df)
    avg_rows_per_merchant = total_rows / unique_merchants
    
    print(f"\nUnique Merchant IDs: {unique_merchants:,}")
    print(f"Total Rows: {total_rows:,}")
    print(f"Average rows per Merchant ID: {avg_rows_per_merchant:.2f}")
    
    # Distribution of rows per merchant
    merchant_counts = df[merchant_col].value_counts()
    print(f"\nRows per Merchant Distribution:")
    print(f"  Min rows: {merchant_counts.min()}")
    print(f"  Max rows: {merchant_counts.max()}")
    print(f"  Median rows: {merchant_counts.median():.0f}")
    print(f"  Mean rows: {merchant_counts.mean():.2f}")
    print(f"  Std rows: {merchant_counts.std():.2f}")
    
    # Top 10 merchants by transaction count
    print(f"\nTop 10 Merchants by Transaction Count:")
    print(merchant_counts.head(10))
    
    # Fraud rate by merchant (sample)
    print(f"\nFraud rate analysis:")
    merchant_fraud = df.groupby(merchant_col)['Is_Fraud'].agg(['sum', 'count', 'mean'])
    merchant_fraud.columns = ['Fraud_Count', 'Total_Transactions', 'Fraud_Rate']
    merchant_fraud = merchant_fraud.sort_values('Fraud_Rate', ascending=False)
    
    print(f"\nTop 10 Merchants by Fraud Rate (with at least 10 transactions):")
    high_volume = merchant_fraud[merchant_fraud['Total_Transactions'] >= 10]
    print(high_volume.head(10))
    
else:
    print("\n⚠ WARNING: No Merchant_ID column found!")
    print("Available columns that might be relevant:")
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['id', 'merchant', 'store', 'shop']):
            print(f"  - {col}")

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing = df.isnull().sum()
if missing.sum() > 0:
    print("\nColumns with missing values:")
    print(missing[missing > 0])
else:
    print("\nNo missing values found!")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
