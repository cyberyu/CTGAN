"""
Bank Transaction Fraud Detection - TVAE Latent Space Nearest Neighbor Approach
===============================================================================

This script implements the TVAE latent space nearest neighbor strategy for fraud detection
similar to 4th_vae_latent_nn.py but applied to bank transaction data:

1. Select 2 random bank branches as holdout
2. Train TVAE on available branch data only (NOT holdout)
3. Encode all data into TVAE's transformed feature space
4. For each holdout branch transaction, find nearest neighbor in available data
5. Use matched available branch data as "synthetic proxy" for holdout
6. Train classifier on these proxy records
7. Test on real holdout branch data

Target: Is_Fraud (fraud detection)
Grouping: Bank_Branch (2 holdout branches out of 145 total)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, precision_recall_curve
)
from ctgan import TVAE
import torch
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BANK TRANSACTION FRAUD DETECTION - TVAE LATENT SPACE NEAREST NEIGHBOR")
print("Privacy-Preserving ML using Latent Space Similarity")
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
print("STEP 1: SPLITTING DATA - HOLDOUT BRANCHES")
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
# STEP 2: PREPARE DATA FOR TVAE (NUMERICAL ENCODING)
# ============================================================================
print("=" * 80)
print("STEP 2: PREPARING DATA - ENCODE ALL FEATURES AS NUMERICAL")
print("=" * 80)

# Columns to drop
cols_to_drop = [
    'Customer_ID', 'Customer_Name', 'Transaction_ID', 'Transaction_Date',
    'Transaction_Time', 'Customer_Contact', 'Transaction_Description',
    'Customer_Email', 'Bank_Branch', 'Merchant_ID'
]

# Prepare data
available_processed = available_data.drop(columns=cols_to_drop).copy()
holdout_processed = holdout_data.drop(columns=cols_to_drop).copy()

# Identify discrete columns
discrete_columns = [
    'Gender', 'State', 'City', 'Account_Type', 'Transaction_Type',
    'Merchant_Category', 'Transaction_Device', 'Transaction_Location',
    'Device_Type', 'Is_Fraud', 'Transaction_Currency'
]

print(f"Available data shape: {available_processed.shape}")
print(f"Holdout data shape: {holdout_processed.shape}")
print(f"Columns: {list(available_processed.columns)}")
print(f"Discrete columns: {discrete_columns}")
print()

# ============================================================================
# STEP 3: TRAIN TVAE ON AVAILABLE BRANCH DATA ONLY
# ============================================================================
print("=" * 80)
print("STEP 3: TRAINING TVAE ON AVAILABLE BRANCH DATA ONLY")
print("=" * 80)
print("NOTE: TVAE learns latent space from available branches, holdout NOT used for training!")
print()

# Check if TVAE model already exists
import os
if os.path.exists('tvae_fraud_model.pkl'):
    print("="*80)
    print("✓ FOUND EXISTING TVAE MODEL - SKIPPING TRAINING")
    print("="*80)
    print("Loading existing model: tvae_fraud_model.pkl")
    print()
    
    tvae = TVAE.load('tvae_fraud_model.pkl')
    print("✓ Model loaded successfully")
    print()
else:
    print("Training with 600 epochs...")
    print()

    # Train TVAE on available data only
    tvae = TVAE(
        epochs=600,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        verbose=True,
        batch_size=500
    )

    tvae.fit(available_processed, discrete_columns=discrete_columns)

    print("\nTVAE training complete!")
    print()

    # Save the trained model
    print("Saving TVAE model...")
    try:
        tvae.save('tvae_fraud_model.pkl')
        if os.path.exists('tvae_fraud_model.pkl'):
            file_size = os.path.getsize('tvae_fraud_model.pkl') / (1024*1024)
            print(f"✓ Model saved: tvae_fraud_model.pkl ({file_size:.2f} MB)")
        else:
            print("⚠ Model save verification failed")
    except Exception as e:
        print(f"⚠ Error saving model: {e}")
    print()

# ============================================================================
# STEP 4: ENCODE DATA INTO LATENT SPACE
# ============================================================================
print("=" * 80)
print("STEP 4: ENCODING DATA INTO TRANSFORMED FEATURE SPACE")
print("=" * 80)
print("Using TVAE transformer to project data into feature vectors...")
print()

# Access the transformer from TVAE
transformer = tvae.transformer

# Transform data to TVAE's internal format - pass DataFrames
available_transformed = transformer.transform(available_processed)
holdout_transformed = transformer.transform(holdout_processed)

# Use transformed data as our feature space
z_available = available_transformed
z_holdout = holdout_transformed

print(f"Available transformed data shape: {z_available.shape}")
print(f"Holdout transformed data shape: {z_holdout.shape}")
print(f"Feature space dimensionality: {z_available.shape[1]}")
print("(Using TVAE's transformed feature space as latent representation)")
print()

# ============================================================================
# STEP 5: NEAREST NEIGHBOR LOOKUP IN FEATURE SPACE
# ============================================================================
print("=" * 80)
print("STEP 5: FINDING NEAREST NEIGHBORS IN TRANSFORMED FEATURE SPACE")
print("=" * 80)
print("For each holdout branch transaction, finding closest available transaction...")
print()

# Use KNN to find nearest neighbors
nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn_model.fit(z_available)

# Find nearest neighbors for holdout data
distances, indices = nn_model.kneighbors(z_holdout)

print(f"Found {len(indices):,} nearest neighbors")
print(f"Average distance in feature space: {distances.mean():.4f}")
print(f"Min distance: {distances.min():.4f}")
print(f"Max distance: {distances.max():.4f}")
print()

# Extract the matched available records (the "synthetic proxy" data)
matched_indices = indices.flatten()
proxy_data = available_processed.iloc[matched_indices].reset_index(drop=True)

print(f"Proxy data shape: {proxy_data.shape}")
print(f"Proxy data fraud distribution:")
print(proxy_data['Is_Fraud'].value_counts().sort_index())
print(f"Proxy fraud rate: {proxy_data['Is_Fraud'].mean()*100:.2f}%")
print()

# Save proxy data before encoding
print("Saving proxy data...")
import os
try:
    proxy_data.to_csv('proxy_fraud_data.csv', index=False)
    if os.path.exists('proxy_fraud_data.csv'):
        file_size = os.path.getsize('proxy_fraud_data.csv') / (1024*1024)
        print(f"✓ Proxy data saved: proxy_fraud_data.csv ({file_size:.2f} MB, {len(proxy_data):,} rows)")
    else:
        print("✗ Warning: CSV save completed but file not found")
except Exception as e:
    print(f"✗ Error saving proxy data: {e}")
print()

# ============================================================================
# STEP 6: TRAINING CLASSIFIER ON PROXY DATA
# ============================================================================
print("=" * 80)
print("STEP 6: TRAINING CLASSIFIER ON PROXY DATA")
print("=" * 80)
print("Using nearest neighbor matches from available branches as training data...")
print()

# Encode categorical columns for XGBoost
print("Encoding categorical columns for XGBoost...")
from sklearn.preprocessing import LabelEncoder

# Identify ALL categorical columns from available_processed (contains all original data)
cat_columns = available_processed.select_dtypes(include=['object']).columns.tolist()
# Remove target column if present
if 'Is_Fraud' in cat_columns:
    cat_columns.remove('Is_Fraud')
print(f"Categorical columns to encode: {cat_columns}")

# Create label encoders and fit on available_processed (largest dataset with all categories)
label_encoders = {}
for col in cat_columns:
    le = LabelEncoder()
    available_processed[col] = le.fit_transform(available_processed[col].astype(str))
    label_encoders[col] = le

# Apply same encoders to proxy_data
for col in cat_columns:
    if col in proxy_data.columns:
        le = label_encoders[col]
        proxy_data[col] = proxy_data[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# Apply same encoders to holdout_processed (handle unseen categories)
for col in cat_columns:
    if col in holdout_processed.columns:
        le = label_encoders[col]
        holdout_processed[col] = holdout_processed[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

print("Encoding complete!")
print()

# Prepare features and target for proxy data
X_proxy = proxy_data.drop('Is_Fraud', axis=1)
y_proxy = proxy_data['Is_Fraud']

# Split proxy data into train and validation
X_proxy_train, X_proxy_val, y_proxy_train, y_proxy_val = train_test_split(
    X_proxy, y_proxy, test_size=0.2, random_state=42, stratify=y_proxy
)

print(f"Proxy training set: {len(X_proxy_train):,}")
print(f"Proxy validation set: {len(X_proxy_val):,}")
print()

# Calculate class imbalance
n_neg = (y_proxy_train == 0).sum()
n_pos = (y_proxy_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"Proxy data class imbalance ratio: {scale_pos_weight:.2f}")
print()

# Apply SMOTE to balance proxy training data (optimal: 23% fraud)
print("Applying SMOTE to proxy training data (optimal: 23% fraud)...")
smote_proxy = SMOTE(sampling_strategy=0.3, random_state=42)
X_proxy_train_balanced, y_proxy_train_balanced = smote_proxy.fit_resample(X_proxy_train, y_proxy_train)
n_pos_proxy_after = (y_proxy_train_balanced == 1).sum()
print(f"After SMOTE: {len(y_proxy_train_balanced):,} samples")
print(f"Fraud cases after SMOTE: {n_pos_proxy_after:,} ({n_pos_proxy_after/len(y_proxy_train_balanced)*100:.2f}%)")
print()

# Train XGBoost with optimal hyperparameters from grid search
print("Training XGBoost classifier on balanced proxy data...")
proxy_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
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

proxy_model.fit(X_proxy_train_balanced, y_proxy_train_balanced)
print("Model trained on proxy data!")
print()

# Find optimal threshold for proxy model
print("Finding optimal threshold for proxy model...")
y_proxy_val_proba = proxy_model.predict_proba(X_proxy_val)[:, 1]
precision_vals_proxy, recall_vals_proxy, thresholds_proxy = precision_recall_curve(y_proxy_val, y_proxy_val_proba)
f1_scores_proxy = 2 * (precision_vals_proxy * recall_vals_proxy) / (precision_vals_proxy + recall_vals_proxy + 1e-10)
optimal_idx_proxy = np.argmax(f1_scores_proxy[:-1])
optimal_threshold_proxy = thresholds_proxy[optimal_idx_proxy]
print(f"Optimal threshold: {optimal_threshold_proxy:.4f}")
print(f"Expected - Precision: {precision_vals_proxy[optimal_idx_proxy]:.4f}, Recall: {recall_vals_proxy[optimal_idx_proxy]:.4f}, F1: {f1_scores_proxy[optimal_idx_proxy]:.4f}")
print()

# ============================================================================
# STEP 7: TEST ON REAL HOLDOUT BRANCH DATA
# ============================================================================
print("=" * 80)
print("STEP 7: TESTING ON REAL HOLDOUT BRANCH DATA")
print("=" * 80)
print("This tests whether proxy data can predict for unseen branches...")
print()

# Prepare holdout test data
X_holdout_test = holdout_processed.drop('Is_Fraud', axis=1)
y_holdout_test = holdout_processed['Is_Fraud']

print(f"Holdout test set size: {len(X_holdout_test):,}")
print(f"Holdout fraud distribution: {dict(y_holdout_test.value_counts().sort_index())}")
print()

# Predict on holdout with optimal threshold
y_pred_proba_proxy = proxy_model.predict_proba(X_holdout_test)[:, 1]
y_pred_proxy = (y_pred_proba_proxy >= optimal_threshold_proxy).astype(int)

# Calculate metrics
acc_proxy = accuracy_score(y_holdout_test, y_pred_proxy)
bacc_proxy = balanced_accuracy_score(y_holdout_test, y_pred_proxy)
prec_proxy = precision_score(y_holdout_test, y_pred_proxy)
rec_proxy = recall_score(y_holdout_test, y_pred_proxy)
f1_proxy = f1_score(y_holdout_test, y_pred_proxy)
auc_proxy = roc_auc_score(y_holdout_test, y_pred_proba_proxy)
cm_proxy = confusion_matrix(y_holdout_test, y_pred_proxy)

print("=" * 80)
print("RESULTS: PROXY-TRAINED MODEL ON REAL HOLDOUT BRANCH DATA")
print("=" * 80)
print()
print("Performance Metrics:")
print(f"  Accuracy:          {acc_proxy:.4f}")
print(f"  Balanced Accuracy: {bacc_proxy:.4f}")
print(f"  Precision:         {prec_proxy:.4f}")
print(f"  Recall:            {rec_proxy:.4f}")
print(f"  F1-Score:          {f1_proxy:.4f}")
print(f"  AUC-ROC:           {auc_proxy:.4f}")
print()
print("Confusion Matrix:")
print(cm_proxy)
print()
print(f"  True Negatives:  {cm_proxy[0, 0]}")
print(f"  False Positives: {cm_proxy[0, 1]}")
print(f"  False Negatives: {cm_proxy[1, 0]}")
print(f"  True Positives:  {cm_proxy[1, 1]}")
print()
print("Detailed Classification Report:")
print(classification_report(y_holdout_test, y_pred_proxy, 
                          target_names=['Not Fraud', 'Fraud']))
print()

# ============================================================================
# STEP 8: BASELINE COMPARISON - REAL DATA APPROACH
# ============================================================================
print("=" * 80)
print("STEP 8: BASELINE COMPARISON - REAL DATA APPROACH")
print("=" * 80)
print("Training directly on ALL REAL available branch data (no NN matching)")
print()

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

# Apply SMOTE to real training data (optimal: 23% fraud)
print("Applying SMOTE to real training data (optimal: 23% fraud)...")
smote_real = SMOTE(sampling_strategy=0.3, random_state=42)
X_real_train_balanced, y_real_train_balanced = smote_real.fit_resample(X_real_train, y_real_train)
n_pos_real_after = (y_real_train_balanced == 1).sum()
print(f"After SMOTE: {len(y_real_train_balanced):,} samples")
print(f"Fraud cases after SMOTE: {n_pos_real_after:,} ({n_pos_real_after/len(y_real_train_balanced)*100:.2f}%)")
print()

# Train model with optimal hyperparameters from grid search
real_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
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

# Find optimal threshold for real model
print("Finding optimal threshold for real model...")
y_real_val_proba = real_model.predict_proba(X_real_val)[:, 1]
precision_vals_real, recall_vals_real, thresholds_real = precision_recall_curve(y_real_val, y_real_val_proba)
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
print("FINAL COMPARISON: TVAE LATENT NN VS REAL DATA APPROACH")
print("=" * 80)
print()

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'TVAE Latent NN Approach': [acc_proxy, bacc_proxy, prec_proxy, rec_proxy, f1_proxy, auc_proxy],
    'Real Data Baseline': [acc_real, bacc_real, prec_real, rec_real, f1_real, auc_real],
    'Difference': [
        acc_proxy - acc_real,
        bacc_proxy - bacc_real,
        prec_proxy - prec_real,
        rec_proxy - rec_real,
        f1_proxy - f1_real,
        auc_proxy - auc_real
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
print("1. TVAE was trained ONLY on available branch data (holdout NOT used for training)")
print("2. Holdout data was encoded into latent space using the trained transformer")
print("3. Nearest neighbors in feature space were found from available branches")
print("4. Classifier trained on these matched available branch records (proxy data)")
print(f"5. Performance gap (TVAE NN vs Real): {f1_proxy - f1_real:.4f} F1-Score")
print()

if abs(f1_proxy - f1_real) < 0.05:
    print("✓ EXCELLENT: TVAE Latent NN approach performs nearly as well as real data!")
elif abs(f1_proxy - f1_real) < 0.15:
    print("⚠ GOOD: Reasonable performance, consider tuning TVAE epochs or architecture")
else:
    print("⚠ CAUTION: Significant performance gap - may need more TVAE training")

print()
print("6. This demonstrates privacy-preserving ML using transformed feature similarity")
print("7. Privacy benefit: Classifier trained on available data selected by similarity")
print("8. Note: Using TVAE's transformed feature space (not true latent vectors)")
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
