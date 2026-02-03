"""
Bank Transaction Fraud Detection - CTGAN Synthetic Data Approach
==================================================================

This script implements the CTGAN synthetic data approach for fraud detection
similar to 3rd_synthetic_approach.py but applied to bank transaction data:

1. Select 2 random bank branches as "holdout branches" (similar to Client 3)
2. Combine all data with branch_id indicator (1=available, 3=holdout)
3. Train CTGAN on ALL real data (including holdout branches)
4. Generate synthetic data
5. Train classifier ONLY on synthetic data
6. Test on real holdout branch data
7. Baseline comparison: train on real available data, test on real holdout

Target: Is_Fraud (fraud detection)
Grouping: Bank_Branch (2 holdout branches out of 145 total)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, precision_recall_curve
)
from ctgan import CTGAN
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BANK TRANSACTION FRAUD DETECTION - CTGAN SYNTHETIC DATA APPROACH")
print("Simulating Privacy-Preserving ML: Holdout Branches Strategy")
print("=" * 80)
print()

# Load data
print("Loading bank transaction fraud detection dataset...")
data = pd.read_csv('examples/csv/Bank_Transaction_Fraud_Detection.csv')
print(f"Dataset shape: {data.shape}")
print(f"Fraud rate: {data['Is_Fraud'].mean()*100:.2f}%")
print()

# ============================================================================
# STEP 1: SPLIT DATA - HOLDOUT BRANCHES
# ============================================================================
print("=" * 80)
print("STEP 1: SPLITTING DATA - SIMULATING HOLDOUT BRANCHES")
print("=" * 80)

# Select 2 random branches as holdout
unique_branches = data['Bank_Branch'].unique()
n_holdout = 2
np.random.seed(42)
holdout_branches = np.random.choice(unique_branches, size=n_holdout, replace=False)

print(f"Total unique branches: {len(unique_branches)}")
print(f"Holdout branches selected: {n_holdout}")
print(f"Holdout branches: {list(holdout_branches)}")
print()

# Split data
holdout_data = data[data['Bank_Branch'].isin(holdout_branches)].copy()
available_data = data[~data['Bank_Branch'].isin(holdout_branches)].copy()

print(f"Holdout branch records: {len(holdout_data)} ({len(holdout_data)/len(data)*100:.1f}%)")
print(f"Available branch records: {len(available_data)} ({len(available_data)/len(data)*100:.1f}%)")
print()
print("Holdout branches fraud distribution:")
print(holdout_data['Is_Fraud'].value_counts().sort_index())
print(f"Holdout fraud rate: {holdout_data['Is_Fraud'].mean()*100:.2f}%")
print()
print("Available branches fraud distribution:")
print(available_data['Is_Fraud'].value_counts().sort_index())
print(f"Available fraud rate: {available_data['Is_Fraud'].mean()*100:.2f}%")
print()

# ============================================================================
# STEP 2: PREPARE DATA FOR CTGAN
# ============================================================================
print("=" * 80)
print("STEP 2: PREPARING DATA FOR CTGAN")
print("=" * 80)

# Columns to drop (not useful for modeling or too high cardinality)
cols_to_drop = [
    'Customer_ID', 'Customer_Name', 'Transaction_ID', 'Transaction_Date',
    'Transaction_Time', 'Customer_Contact', 'Transaction_Description',
    'Customer_Email', 'Bank_Branch', 'Merchant_ID'
]

# Prepare data
available_processed = available_data.drop(columns=cols_to_drop)
holdout_processed = holdout_data.drop(columns=cols_to_drop)

# Add branch_id indicator (1=available, 3=holdout)
available_processed['branch_id'] = 1
holdout_processed['branch_id'] = 3

# Combine all data for CTGAN training
all_data = pd.concat([available_processed, holdout_processed], ignore_index=True)

print(f"Combined dataset for CTGAN: {all_data.shape}")
print(f"Columns: {list(all_data.columns)}")
print()

# Identify discrete columns
discrete_columns = [
    'Gender', 'State', 'City', 'Account_Type', 'Transaction_Type',
    'Merchant_Category', 'Transaction_Device', 'Transaction_Location',
    'Device_Type', 'Is_Fraud', 'Transaction_Currency', 'branch_id'
]

print(f"Discrete columns for CTGAN: {discrete_columns}")
print()

# ============================================================================
# STEP 3: TRAIN CTGAN ON ALL REAL DATA
# ============================================================================
print("=" * 80)
print("STEP 3: TRAINING CTGAN ON ALL REAL DATA")
print("=" * 80)
print("NOTE: CTGAN sees holdout branch data to learn patterns, but classifier will NOT!")
print("Training with 600 epochs for high-quality synthetic data...")
print()

# Check if synthetic data already exists
import os
if os.path.exists('synthetic_fraud_data.csv'):
    print("="*80)
    print("✓ FOUND EXISTING SYNTHETIC DATA - SKIPPING TRAINING")
    print("="*80)
    file_size = os.path.getsize('synthetic_fraud_data.csv') / (1024*1024)
    print(f"Loading existing synthetic data: synthetic_fraud_data.csv ({file_size:.2f} MB)")
    print()
    
    # Skip directly to loading
    import pandas as pd
    synthetic_data = pd.read_csv('synthetic_fraud_data.csv')
    print(f"✓ Loaded {len(synthetic_data):,} synthetic samples")
    print("Synthetic data fraud distribution:")
    print(synthetic_data['Is_Fraud'].value_counts().sort_index())
    print(f"Synthetic fraud rate: {synthetic_data['Is_Fraud'].mean()*100:.2f}%")
    print()
    print("Skipping to Step 5: Training Classifier...")
    print()
    
    # Jump to encoding section - need to set flag
    skip_training = True
else:
    skip_training = False

if not skip_training:
    ctgan = CTGAN(epochs=600, verbose=True, batch_size=500)
    ctgan.fit(all_data, discrete_columns=discrete_columns)

    print("\nCTGAN training complete!")
    print()

    # Save the trained model
    print("Saving CTGAN model...")
    try:
        ctgan.save('ctgan_fraud_model.pkl')
        if os.path.exists('ctgan_fraud_model.pkl'):
            file_size = os.path.getsize('ctgan_fraud_model.pkl') / (1024*1024)
            print(f"✓ Model saved successfully: ctgan_fraud_model.pkl ({file_size:.2f} MB)")
        else:
            print("✗ Warning: Model save command completed but file not found")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    print()

    # ============================================================================
    # STEP 4: GENERATE SYNTHETIC DATA
    # ============================================================================
    print("=" * 80)
    print("STEP 4: GENERATING SYNTHETIC DATA")
    print("=" * 80)

    # Generate synthetic data (2x the original size for better coverage)
    n_synthetic = len(all_data) * 2
    print(f"Generating {n_synthetic:,} synthetic samples...")
    synthetic_data = ctgan.sample(n_synthetic)

    print(f"Synthetic data shape: {synthetic_data.shape}")
    print()
    print("Synthetic data branch_id distribution:")
    print(synthetic_data['branch_id'].value_counts().sort_index())
    print()
    print("Synthetic data fraud distribution:")
    print(synthetic_data['Is_Fraud'].value_counts().sort_index())
    print(f"Synthetic fraud rate: {synthetic_data['Is_Fraud'].mean()*100:.2f}%")
    print()

    # Save synthetic data before encoding
    print("Saving synthetic data...")
    try:
        synthetic_data.to_csv('synthetic_fraud_data.csv', index=False)
        if os.path.exists('synthetic_fraud_data.csv'):
            file_size = os.path.getsize('synthetic_fraud_data.csv') / (1024*1024)
            print(f"✓ Synthetic data saved: synthetic_fraud_data.csv ({file_size:.2f} MB, {len(synthetic_data):,} rows)")
        else:
            print("✗ Warning: CSV save completed but file not found")
    except Exception as e:
        print(f"✗ Error saving synthetic data: {e}")
    print()

# ============================================================================
# STEP 5: TRAINING CLASSIFIER ON SYNTHETIC DATA ONLY
# ============================================================================
print("=" * 80)
print("STEP 5: TRAINING CLASSIFIER ON SYNTHETIC DATA ONLY")
print("=" * 80)
print("Encoding categorical columns for XGBoost...")
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns (object dtype)
cat_columns = synthetic_data.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {cat_columns}")

# Create label encoders
label_encoders = {}
for col in cat_columns:
    le = LabelEncoder()
    synthetic_data[col] = le.fit_transform(synthetic_data[col].astype(str))
    label_encoders[col] = le

# Also encode the same columns in holdout/available data
for col in cat_columns:
    if col in available_processed.columns:
        available_processed[col] = label_encoders[col].transform(available_processed[col].astype(str))
    if col in holdout_processed.columns:
        holdout_processed[col] = label_encoders[col].transform(holdout_processed[col].astype(str))

print("Encoding complete!")
print()

# ============================================================================
# STEP 5: TRAIN CLASSIFIER ON SYNTHETIC DATA ONLY
# ============================================================================
print("=" * 80)
print("STEP 5: TRAINING CLASSIFIER ON SYNTHETIC DATA ONLY")
print("=" * 80)

# Prepare synthetic training data
X_synthetic = synthetic_data.drop('Is_Fraud', axis=1)
y_synthetic = synthetic_data['Is_Fraud']

# Split synthetic data into train/val
X_syn_train, X_syn_val, y_syn_train, y_syn_val = train_test_split(
    X_synthetic, y_synthetic, test_size=0.2, random_state=42, stratify=y_synthetic
)

print(f"Synthetic training set: {len(X_syn_train):,}")
print(f"Synthetic validation set: {len(X_syn_val):,}")
print()

# Calculate class imbalance
n_neg = (y_syn_train == 0).sum()
n_pos = (y_syn_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"Synthetic data class imbalance ratio: {scale_pos_weight:.2f}")
print(f"Fraud cases in training: {n_pos:,} ({n_pos/len(y_syn_train)*100:.2f}%)")
print()

# Apply SMOTE to balance the training data
print("Applying SMOTE oversampling to balance training data...")
smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Oversample to 30% fraud
X_syn_train_balanced, y_syn_train_balanced = smote.fit_resample(X_syn_train, y_syn_train)

n_pos_after = (y_syn_train_balanced == 1).sum()
print(f"After SMOTE: {len(y_syn_train_balanced):,} samples")
print(f"Fraud cases after SMOTE: {n_pos_after:,} ({n_pos_after/len(y_syn_train_balanced)*100:.2f}%)")
print()

# Train XGBoost with optimal parameters (same as baseline for fair comparison)
print("Training XGBoost classifier on balanced synthetic data...")
synthetic_model = xgb.XGBClassifier(
    max_depth=6,              # Optimal from grid search
    learning_rate=0.05,
    n_estimators=500,         # Optimal from grid search
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    scale_pos_weight=1.5,
    random_state=42,
    eval_metric='aucpr',
    reg_alpha=0.1,
    reg_lambda=1.0
)

synthetic_model.fit(X_syn_train_balanced, y_syn_train_balanced)
print("Model trained on synthetic data!")
print()

# ============================================================================
# STEP 6: TEST ON REAL HOLDOUT BRANCH DATA
# ============================================================================
print("=" * 80)
print("STEP 6: TESTING ON REAL HOLDOUT BRANCH DATA")
print("=" * 80)
print("This simulates predicting for branches whose real data was never seen by classifier")
print()

# Prepare holdout test data
X_holdout_test = holdout_processed.drop('Is_Fraud', axis=1)
y_holdout_test = holdout_processed['Is_Fraud']

print(f"Holdout test set size: {len(X_holdout_test):,}")
print(f"Holdout fraud distribution: {dict(y_holdout_test.value_counts().sort_index())}")
print()

# Predict on holdout test set with optimized threshold
print("Finding optimal classification threshold on validation set...")
y_val_proba = synthetic_model.predict_proba(X_syn_val)[:, 1]
precision_vals, recall_vals, thresholds = precision_recall_curve(y_syn_val, y_val_proba)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
optimal_idx = np.argmax(f1_scores[:-1])
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.4f} (default=0.5)")
print(f"Expected - Precision: {precision_vals[optimal_idx]:.4f}, Recall: {recall_vals[optimal_idx]:.4f}, F1: {f1_scores[optimal_idx]:.4f}")
print()

# Apply optimal threshold to holdout predictions
y_proba_syn = synthetic_model.predict_proba(X_holdout_test)[:, 1]
y_pred_syn = (y_proba_syn >= optimal_threshold).astype(int)

# Calculate metrics
acc_syn = accuracy_score(y_holdout_test, y_pred_syn)
bacc_syn = balanced_accuracy_score(y_holdout_test, y_pred_syn)
prec_syn = precision_score(y_holdout_test, y_pred_syn)
rec_syn = recall_score(y_holdout_test, y_pred_syn)
f1_syn = f1_score(y_holdout_test, y_pred_syn)
auc_syn = roc_auc_score(y_holdout_test, y_proba_syn)
cm_syn = confusion_matrix(y_holdout_test, y_pred_syn)

print("=" * 80)
print("RESULTS: SYNTHETIC-TRAINED MODEL ON REAL HOLDOUT BRANCH DATA")
print("=" * 80)
print()
print("Performance Metrics:")
print(f"  Accuracy:          {acc_syn:.4f}")
print(f"  Balanced Accuracy: {bacc_syn:.4f}")
print(f"  Precision:         {prec_syn:.4f}")
print(f"  Recall:            {rec_syn:.4f}")
print(f"  F1-Score:          {f1_syn:.4f}")
print(f"  AUC-ROC:           {auc_syn:.4f}")
print()
print("Confusion Matrix:")
print(cm_syn)
print()
print(f"  True Negatives:  {cm_syn[0, 0]}")
print(f"  False Positives: {cm_syn[0, 1]}")
print(f"  False Negatives: {cm_syn[1, 0]}")
print(f"  True Positives:  {cm_syn[1, 1]}")
print()
print("Detailed Classification Report:")
print(classification_report(y_holdout_test, y_pred_syn, 
                          target_names=['Not Fraud', 'Fraud']))
print()

# ============================================================================
# STEP 7: BASELINE COMPARISON - REAL DATA APPROACH
# ============================================================================
print("=" * 80)
print("STEP 7: BASELINE COMPARISON - REAL DATA APPROACH")
print("=" * 80)
print("Training on REAL available branch data, testing on REAL holdout branch data")

# Prepare real available data
X_real = available_processed.drop('Is_Fraud', axis=1)
y_real = available_processed['Is_Fraud']

# Split into train/val
X_real_train, X_real_val, y_real_train, y_real_val = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

print(f"Real training set: {len(X_real_train):,}")

# Calculate class imbalance
n_neg_real = (y_real_train == 0).sum()
n_pos_real = (y_real_train == 1).sum()
scale_pos_weight_real = n_neg_real / n_pos_real
print(f"Real class imbalance ratio: {scale_pos_weight_real:.2f}")
print(f"Fraud cases in real training: {n_pos_real:,} ({n_pos_real/len(y_real_train)*100:.2f}%)")
print()

# Apply SMOTE with optimal ratio from grid search (23% fraud)
print("Applying SMOTE to real training data (optimal: 23% fraud)...")
smote_real = SMOTE(sampling_strategy=0.3, random_state=42)  # Results in ~23% fraud
X_real_train_balanced, y_real_train_balanced = smote_real.fit_resample(X_real_train, y_real_train)
n_pos_real_after = (y_real_train_balanced == 1).sum()
print(f"After SMOTE: {len(y_real_train_balanced):,} samples")
print(f"Fraud cases after SMOTE: {n_pos_real_after:,} ({n_pos_real_after/len(y_real_train_balanced)*100:.2f}%)")
print()

# Train model with optimal hyperparameters from grid search
real_model = xgb.XGBClassifier(
    max_depth=6,              
    learning_rate=0.05,       
    n_estimators=500,         # Optimal from grid search
    min_child_weight=2,       
    subsample=0.8,           
    colsample_bytree=0.8,    
    gamma=0.2,                
    scale_pos_weight=1.5,     
    random_state=42,
    eval_metric='aucpr',
    reg_alpha=0.1,            
    reg_lambda=1.0            
)
real_model.fit(X_real_train_balanced, y_real_train_balanced)
print("Model trained on real available branch data!")
print()

# Find optimal threshold for real model maximizing F1
print("Finding optimal threshold for real model...")
y_real_val_proba = real_model.predict_proba(X_real_val)[:, 1]
precision_vals_real, recall_vals_real, thresholds_real = precision_recall_curve(y_real_val, y_real_val_proba)

# Find threshold that maximizes F1-score
f1_scores_real = 2 * (precision_vals_real * recall_vals_real) / (precision_vals_real + recall_vals_real + 1e-10)
optimal_idx_real = np.argmax(f1_scores_real[:-1])
optimal_threshold_real = thresholds_real[optimal_idx_real]

print(f"Optimal threshold: {optimal_threshold_real:.4f}")
print(f"Expected - Precision: {precision_vals_real[optimal_idx_real]:.4f}, Recall: {recall_vals_real[optimal_idx_real]:.4f}, F1: {f1_scores_real[optimal_idx_real]:.4f}")
print()

# Test on holdout with optimal threshold
y_pred_proba_real = real_model.predict_proba(X_holdout_test)[:, 1]
y_pred_real = (y_pred_proba_real >= optimal_threshold_real).astype(int)

# Calculate metrics
acc_real = accuracy_score(y_holdout_test, y_pred_real)
bacc_real = balanced_accuracy_score(y_holdout_test, y_pred_real)
prec_real = precision_score(y_holdout_test, y_pred_real)
rec_real = recall_score(y_holdout_test, y_pred_real)
f1_real = f1_score(y_holdout_test, y_pred_real)
auc_real = roc_auc_score(y_holdout_test, y_pred_proba_real)

print("Baseline Performance (Real→Real):")
print(f"  Accuracy:          {acc_real:.4f}")
print(f"  Balanced Accuracy: {bacc_real:.4f}")
print(f"  Precision:         {prec_real:.4f}")
print(f"  Recall:            {rec_real:.4f}")
print(f"  F1-Score:          {f1_real:.4f}")
print(f"  AUC-ROC:           {auc_real:.4f}")
print()

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("=" * 80)
print("FINAL COMPARISON: SYNTHETIC VS REAL DATA APPROACH")
print("=" * 80)
print()

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Synthetic Approach': [acc_syn, bacc_syn, prec_syn, rec_syn, f1_syn, auc_syn],
    'Real Data Baseline': [acc_real, bacc_real, prec_real, rec_real, f1_real, auc_real],
    'Difference': [
        acc_syn - acc_real,
        bacc_syn - bacc_real,
        prec_syn - prec_real,
        rec_syn - rec_real,
        f1_syn - f1_real,
        auc_syn - auc_real
    ]
})

print(comparison.to_string(index=False))
print()

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("1. CTGAN was trained on ALL real data (including holdout branches)")
print("2. Classifier was trained ONLY on synthetic data")
print("3. Both models tested on real holdout branch data")
print(f"4. Performance gap (Synthetic vs Real): {f1_syn - f1_real:.4f} F1-Score")
print()

if abs(f1_syn - f1_real) < 0.05:
    print("✓ EXCELLENT: Synthetic approach performs nearly as well as real data!")
elif abs(f1_syn - f1_real) < 0.15:
    print("⚠ GOOD: Reasonable performance, consider tuning CTGAN epochs or architecture")
else:
    print("⚠ CAUTION: Significant performance gap - consider more CTGAN epochs or TVAE")

print()
print("5. This demonstrates CTGAN's 'Machine Learning Efficacy' metric")
print("6. Privacy benefit: Classifier never saw real holdout branch data directly")
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
