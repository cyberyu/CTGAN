"""Analyze Bank_Branch groupings in fraud detection dataset"""
import pandas as pd

df = pd.read_csv('examples/csv/Bank_Transaction_Fraud_Detection.csv')

print("="*80)
print("BANK_BRANCH ANALYSIS")
print("="*80)

unique_branches = df['Bank_Branch'].nunique()
total_rows = len(df)
avg_rows_per_branch = total_rows / unique_branches

print(f"\nUnique Bank Branches: {unique_branches:,}")
print(f"Total Rows: {total_rows:,}")
print(f"Average rows per Bank Branch: {avg_rows_per_branch:.2f}")

# Distribution of rows per branch
branch_counts = df['Bank_Branch'].value_counts()
print(f"\nRows per Bank Branch Distribution:")
print(f"  Min rows: {branch_counts.min()}")
print(f"  Max rows: {branch_counts.max()}")
print(f"  Median rows: {branch_counts.median():.0f}")
print(f"  Mean rows: {branch_counts.mean():.2f}")
print(f"  Std rows: {branch_counts.std():.2f}")

# Top 10 branches by transaction count
print(f"\nTop 10 Bank Branches by Transaction Count:")
print(branch_counts.head(10))

# Bottom 10 branches
print(f"\nBottom 10 Bank Branches by Transaction Count:")
print(branch_counts.tail(10))

# Fraud rate by branch
print(f"\n" + "="*80)
print("FRAUD ANALYSIS BY BANK BRANCH")
print("="*80)

branch_fraud = df.groupby('Bank_Branch')['Is_Fraud'].agg(['sum', 'count', 'mean'])
branch_fraud.columns = ['Fraud_Count', 'Total_Transactions', 'Fraud_Rate']
branch_fraud = branch_fraud.sort_values('Total_Transactions', ascending=False)

print(f"\nTop 10 Branches by Transaction Volume:")
print(branch_fraud.head(10))

print(f"\nTop 10 Branches by Fraud Rate:")
fraud_sorted = branch_fraud.sort_values('Fraud_Rate', ascending=False)
print(fraud_sorted.head(10))

print(f"\nFraud Rate Statistics across all branches:")
print(f"  Mean fraud rate: {branch_fraud['Fraud_Rate'].mean()*100:.2f}%")
print(f"  Median fraud rate: {branch_fraud['Fraud_Rate'].median()*100:.2f}%")
print(f"  Min fraud rate: {branch_fraud['Fraud_Rate'].min()*100:.2f}%")
print(f"  Max fraud rate: {branch_fraud['Fraud_Rate'].max()*100:.2f}%")

print("\n" + "="*80)
print("RECOMMENDATION FOR SPLITTING")
print("="*80)
print(f"\nWith {unique_branches} unique branches and avg {avg_rows_per_branch:.0f} rows per branch,")
print("we can select 20% of branches as 'holdout' (similar to Client 3 in churn example)")
print(f"This would give us approximately {int(unique_branches * 0.2)} holdout branches")
print(f"with about {int(avg_rows_per_branch * unique_branches * 0.2):,} holdout transactions")
