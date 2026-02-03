"""
Optimize baseline fraud detection model
Find best XGBoost hyperparameters and SMOTE ratio for F1-score
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    precision_recall_curve, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("OPTIMIZING BASELINE FRAUD DETECTION MODEL")
print("=" * 80)
print()

# Load data
print("Loading data...")
df = pd.read_csv('examples/csv/Bank_Transaction_Fraud_Detection.csv')

# Select holdout branches
np.random.seed(42)
holdout_branches = ['Ziro Branch', 'Puducherry Branch']

# Split data
holdout_mask = df['Bank_Branch'].isin(holdout_branches)
holdout_data = df[holdout_mask].copy()
available_data = df[~holdout_mask].copy()

print(f"Available branches: {len(available_data):,} transactions")
print(f"Holdout branches: {len(holdout_data):,} transactions")
print(f"Holdout fraud rate: {holdout_data['Is_Fraud'].mean()*100:.2f}%")
print()

# Prepare available data
available_processed = available_data[[ 
    'Gender', 'Age', 'State', 'City', 'Account_Type',
    'Transaction_Amount', 'Transaction_Type', 'Merchant_Category',
    'Account_Balance', 'Transaction_Device', 'Transaction_Location',
    'Device_Type', 'Is_Fraud', 'Transaction_Currency'
]].copy()

# Encode categorical columns
cat_cols = ['Gender', 'State', 'City', 'Account_Type', 'Transaction_Type',
            'Merchant_Category', 'Transaction_Device', 'Transaction_Location',
            'Device_Type', 'Transaction_Currency']

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    available_processed[col] = le.fit_transform(available_processed[col].astype(str))
    label_encoders[col] = le

X = available_processed.drop('Is_Fraud', axis=1)
y = available_processed['Is_Fraud']

# Split into train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,}")
print(f"Validation set: {len(X_val):,}")
print(f"Training fraud rate: {y_train.mean()*100:.2f}%")
print()

# Prepare holdout test set
holdout_test = holdout_data[[
    'Gender', 'Age', 'State', 'City', 'Account_Type',
    'Transaction_Amount', 'Transaction_Type', 'Merchant_Category',
    'Account_Balance', 'Transaction_Device', 'Transaction_Location',
    'Device_Type', 'Is_Fraud', 'Transaction_Currency'
]].copy()

for col in cat_cols:
    le = label_encoders[col]
    holdout_test[col] = holdout_test[col].astype(str).map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

X_test = holdout_test.drop('Is_Fraud', axis=1)
y_test = holdout_test['Is_Fraud']

print("=" * 80)
print("HYPERPARAMETER SEARCH")
print("=" * 80)
print()

# Grid search over SMOTE ratios and XGBoost parameters
smote_ratios = [0.5, 0.67, 1.0]  # 33%, 40%, 50% fraud
max_depths = [5, 6, 7]
learning_rates = [0.05, 0.1]
n_estimators_list = [300, 400]

best_f1 = 0
best_params = None
best_model = None
best_threshold = None

total_combinations = len(smote_ratios) * len(max_depths) * len(learning_rates) * len(n_estimators_list)
current = 0

print(f"Total combinations to test: {total_combinations}")
print()

for smote_ratio in smote_ratios:
    print(f"\nTesting SMOTE ratio: {smote_ratio} (target ~{smote_ratio/(1+smote_ratio)*100:.0f}% fraud)")
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy=smote_ratio, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    fraud_pct = y_train_balanced.mean() * 100
    print(f"  After SMOTE: {len(y_train_balanced):,} samples, {fraud_pct:.1f}% fraud")
    
    for max_depth in max_depths:
        for learning_rate in learning_rates:
            for n_estimators in n_estimators_list:
                current += 1
                
                # Train model
                model = xgb.XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.2,
                    scale_pos_weight=1.5,
                    random_state=42,
                    eval_metric='aucpr',
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    verbosity=0
                )
                model.fit(X_train_balanced, y_train_balanced)
                
                # Find optimal threshold on validation set
                y_val_proba = model.predict_proba(X_val)[:, 1]
                precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_val_proba)
                f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
                optimal_idx = np.argmax(f1_scores[:-1])
                optimal_threshold = thresholds[optimal_idx]
                val_f1 = f1_scores[optimal_idx]
                
                # Evaluate on validation
                y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
                val_f1_actual = f1_score(y_val, y_val_pred)
                val_precision = precision_score(y_val, y_val_pred)
                val_recall = recall_score(y_val, y_val_pred)
                
                if val_f1_actual > best_f1:
                    best_f1 = val_f1_actual
                    best_params = {
                        'smote_ratio': smote_ratio,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators
                    }
                    best_model = model
                    best_threshold = optimal_threshold
                    
                    print(f"  ✓ NEW BEST [{current}/{total_combinations}]: depth={max_depth}, lr={learning_rate}, n_est={n_estimators}")
                    print(f"    Val F1={val_f1_actual:.4f}, Prec={val_precision:.4f}, Rec={val_recall:.4f}, Thresh={optimal_threshold:.4f}")

print()
print("=" * 80)
print("BEST MODEL FOUND")
print("=" * 80)
print(f"Best validation F1-score: {best_f1:.4f}")
print(f"Best parameters: {best_params}")
print(f"Best threshold: {best_threshold:.4f}")
print()

# Test on holdout
print("=" * 80)
print("TESTING ON HOLDOUT BRANCHES")
print("=" * 80)
y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

test_f1 = f1_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"Holdout Test Performance:")
print(f"  F1-Score:   {test_f1:.4f}")
print(f"  Precision:  {test_precision:.4f}")
print(f"  Recall:     {test_recall:.4f}")
print(f"  AUC-ROC:    {test_auc:.4f}")
print()

# Count predictions
tp = ((y_test == 1) & (y_test_pred == 1)).sum()
fp = ((y_test == 0) & (y_test_pred == 1)).sum()
tn = ((y_test == 0) & (y_test_pred == 0)).sum()
fn = ((y_test == 1) & (y_test_pred == 0)).sum()

print(f"Confusion Matrix:")
print(f"  True Positives:  {tp} (fraud caught)")
print(f"  False Positives: {fp} (false alarms)")
print(f"  True Negatives:  {tn} (correctly cleared)")
print(f"  False Negatives: {fn} (fraud missed)")
print()
print("=" * 80)
