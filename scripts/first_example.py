import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from ctgan import CTGAN

from ctgan import load_demo

import warnings
warnings.filterwarnings('ignore')

# Load the Adult income dataset
print("Loading Adult income dataset...")
real_data = load_demo()
print(f"Dataset shape: {real_data.shape}")
print(f"\nFeatures: {list(real_data.columns)}")
print(f"\nTarget distribution:\n{real_data['income'].value_counts()}")

# generate synthetic data using CTGAN  
print("\nGenerating synthetic data using CTGAN...")
print("NOTE: Training CTGAN with 300 epochs for better quality synthetic data")
print("This may take several minutes...")
# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

ctgan = CTGAN(epochs=300, verbose=True)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
print(f"Synthetic data shape: {synthetic_data.shape}")

# ============================================================================
# PART 1: CLASSIFICATION ON REAL DATA
# ============================================================================
print("\n" + "="*70)
print("PART 1: TRAINING XGBoost ON REAL DATA")
print("="*70)

# Prepare the data
# Separate features and target
X = real_data.drop('income', axis=1)
y = real_data['income']

# Encode target variable (income: <=50K -> 0, >50K -> 1)
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)
print(f"\nTarget classes: {label_encoder_y.classes_}")

# Identify categorical and numerical columns
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
numerical_columns = [
    'age', 'fnlwgt', 'education-num', 'capital-gain',
    'capital-loss', 'hours-per-week'
]

print(f"\nCategorical features: {categorical_columns}")
print(f"Numerical features: {numerical_columns}")

# Encode categorical features
X_encoded = X.copy()
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Split data: 70% train, 15% validation, 15% test
print("\n" + "="*70)
print("SPLITTING DATA")
print("="*70)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
)  # 0.176 * 0.85 ≈ 0.15 of total

print(f"Train set size: {len(X_train)} ({len(X_train)/len(X_encoded)*100:.1f}%)")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X_encoded)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X_encoded)*100:.1f}%)")

# Check class distribution in each set
print(f"\nTrain target distribution: {np.bincount(y_train)}")
print(f"Validation target distribution: {np.bincount(y_val)}")
print(f"Test target distribution: {np.bincount(y_test)}")

# Hyperparameter optimization
print("\n" + "="*70)
print("HYPERPARAMETER OPTIMIZATION")
print("="*70)

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Simplified grid search for faster execution
best_params = None
best_score = 0
best_model = None

print("\nTesting hyperparameter combinations...")
print("(Testing a subset of combinations for efficiency)\n")

# Test a subset of combinations
test_combinations = [
    {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200, 'min_child_weight': 1, 'subsample': 0.9, 'colsample_bytree': 0.9},
    {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 300, 'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'max_depth': 9, 'learning_rate': 0.1, 'n_estimators': 200, 'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0},
    {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 300, 'min_child_weight': 1, 'subsample': 0.9, 'colsample_bytree': 0.9},
    {'max_depth': 7, 'learning_rate': 0.1, 'n_estimators': 200, 'min_child_weight': 3, 'subsample': 0.9, 'colsample_bytree': 0.9},
]

for i, params in enumerate(test_combinations, 1):
    print(f"Testing combination {i}/{len(test_combinations)}")
    print(f"Parameters: {params}")
    
    model = xgb.XGBClassifier(
        **params,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Validation AUC-ROC: {val_auc:.4f}")
    print(f"  Validation F1-Score: {val_f1:.4f}")
    
    # Use F1 score as the optimization metric (balances precision and recall)
    if val_f1 > best_score:
        best_score = val_f1
        best_params = params
        best_model = model
        print(f"  *** New best model! ***")
    print()

print("="*70)
print("BEST HYPERPARAMETERS")
print("="*70)
print(f"Best validation F1-Score: {best_score:.4f}")
print(f"Best parameters: {best_params}")

# Train final model with best parameters on train+validation data
print("\n" + "="*70)
print("TRAINING FINAL MODEL ON TRAIN+VALIDATION DATA")
print("="*70)
final_model = xgb.XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
final_model.fit(X_train_full, y_train_full, verbose=False)
print("Final model training complete!")

# Evaluate on test set
print("\n" + "="*70)
print("FINAL TEST SET RESULTS")
print("="*70)
y_test_pred = final_model.predict(X_test)
y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\nTest Set Metrics:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  AUC-ROC:   {test_auc:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\n  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=label_encoder_y.classes_))

# Feature importance
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*70)
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# PART 2: CLASSIFICATION ON SYNTHETIC DATA
# ============================================================================
print("\n\n" + "="*70)
print("PART 2: TRAINING XGBoost ON SYNTHETIC DATA")
print("="*70)

# Prepare synthetic data
X_synth = synthetic_data.drop('income', axis=1)
y_synth = synthetic_data['income']

# Encode target variable
label_encoder_y_synth = LabelEncoder()
y_synth_encoded = label_encoder_y_synth.fit_transform(y_synth)
print(f"\nSynthetic target classes: {label_encoder_y_synth.classes_}")
print(f"Synthetic target distribution: {np.bincount(y_synth_encoded)}")

# Encode categorical features
X_synth_encoded = X_synth.copy()
label_encoders_synth = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_synth_encoded[col] = le.fit_transform(X_synth[col].astype(str))
    label_encoders_synth[col] = le

# Split synthetic data: 70% train, 15% validation, 15% test
print("\n" + "="*70)
print("SPLITTING SYNTHETIC DATA")
print("="*70)
X_train_full_synth, X_test_synth, y_train_full_synth, y_test_synth = train_test_split(
    X_synth_encoded, y_synth_encoded, test_size=0.15, random_state=42, stratify=y_synth_encoded
)
X_train_synth, X_val_synth, y_train_synth, y_val_synth = train_test_split(
    X_train_full_synth, y_train_full_synth, test_size=0.176, random_state=42, stratify=y_train_full_synth
)

print(f"Train set size: {len(X_train_synth)} ({len(X_train_synth)/len(X_synth_encoded)*100:.1f}%)")
print(f"Validation set size: {len(X_val_synth)} ({len(X_val_synth)/len(X_synth_encoded)*100:.1f}%)")
print(f"Test set size: {len(X_test_synth)} ({len(X_test_synth)/len(X_synth_encoded)*100:.1f}%)")

print(f"\nTrain target distribution: {np.bincount(y_train_synth)}")
print(f"Validation target distribution: {np.bincount(y_val_synth)}")
print(f"Test target distribution: {np.bincount(y_test_synth)}")

# Hyperparameter optimization on synthetic data
print("\n" + "="*70)
print("HYPERPARAMETER OPTIMIZATION (SYNTHETIC)")
print("="*70)

best_params_synth = None
best_score_synth = 0
best_model_synth = None

print("\nTesting hyperparameter combinations on synthetic data...")
print("(Testing same subset of combinations)\n")

for i, params in enumerate(test_combinations, 1):
    print(f"Testing combination {i}/{len(test_combinations)}")
    print(f"Parameters: {params}")
    
    model = xgb.XGBClassifier(
        **params,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train_synth, y_train_synth, verbose=False)
    
    # Evaluate on validation set
    y_val_pred_synth = model.predict(X_val_synth)
    y_val_pred_proba_synth = model.predict_proba(X_val_synth)[:, 1]
    
    val_accuracy = accuracy_score(y_val_synth, y_val_pred_synth)
    val_auc = roc_auc_score(y_val_synth, y_val_pred_proba_synth)
    val_f1 = f1_score(y_val_synth, y_val_pred_synth)
    
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Validation AUC-ROC: {val_auc:.4f}")
    print(f"  Validation F1-Score: {val_f1:.4f}")
    
    if val_f1 > best_score_synth:
        best_score_synth = val_f1
        best_params_synth = params
        best_model_synth = model
        print(f"  *** New best model! ***")
    print()

print("="*70)
print("BEST HYPERPARAMETERS (SYNTHETIC)")
print("="*70)
print(f"Best validation F1-Score: {best_score_synth:.4f}")
print(f"Best parameters: {best_params_synth}")

# Train final model with best parameters on synthetic train+validation data
print("\n" + "="*70)
print("TRAINING FINAL MODEL ON SYNTHETIC TRAIN+VALIDATION DATA")
print("="*70)
final_model_synth = xgb.XGBClassifier(
    **best_params_synth,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
final_model_synth.fit(X_train_full_synth, y_train_full_synth, verbose=False)
print("Final synthetic model training complete!")

# Evaluate on synthetic test set
print("\n" + "="*70)
print("FINAL TEST SET RESULTS (SYNTHETIC)")
print("="*70)
y_test_pred_synth = final_model_synth.predict(X_test_synth)
y_test_pred_proba_synth = final_model_synth.predict_proba(X_test_synth)[:, 1]

# Calculate metrics
test_accuracy_synth = accuracy_score(y_test_synth, y_test_pred_synth)
test_precision_synth = precision_score(y_test_synth, y_test_pred_synth)
test_recall_synth = recall_score(y_test_synth, y_test_pred_synth)
test_f1_synth = f1_score(y_test_synth, y_test_pred_synth)
test_auc_synth = roc_auc_score(y_test_synth, y_test_pred_proba_synth)

print(f"\nTest Set Metrics (Synthetic):")
print(f"  Accuracy:  {test_accuracy_synth:.4f}")
print(f"  Precision: {test_precision_synth:.4f}")
print(f"  Recall:    {test_recall_synth:.4f}")
print(f"  F1-Score:  {test_f1_synth:.4f}")
print(f"  AUC-ROC:   {test_auc_synth:.4f}")

print(f"\nConfusion Matrix (Synthetic):")
cm_synth = confusion_matrix(y_test_synth, y_test_pred_synth)
print(cm_synth)
print(f"\n  True Negatives:  {cm_synth[0, 0]}")
print(f"  False Positives: {cm_synth[0, 1]}")
print(f"  False Negatives: {cm_synth[1, 0]}")
print(f"  True Positives:  {cm_synth[1, 1]}")

print(f"\nDetailed Classification Report (Synthetic):")
print(classification_report(y_test_synth, y_test_pred_synth, target_names=label_encoder_y_synth.classes_))

# Feature importance for synthetic data
print("\n" + "="*70)
print("TOP 10 MOST IMPORTANT FEATURES (SYNTHETIC)")
print("="*70)
feature_importance_synth = pd.DataFrame({
    'feature': X_synth_encoded.columns,
    'importance': final_model_synth.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance_synth.head(10).to_string(index=False))

# ============================================================================
# PART 3: CROSS-VALIDATION - SYNTHETIC MODEL ON REAL DATA TEST SET
# ============================================================================
print("\n\n" + "="*70)
print("PART 3: TESTING SYNTHETIC-TRAINED MODEL ON REAL DATA")
print("="*70)
print("Training on synthetic data, testing on real data test set...")

# Use the best model trained on synthetic data to predict real test data
y_test_pred_cross = final_model_synth.predict(X_test)
y_test_pred_proba_cross = final_model_synth.predict_proba(X_test)[:, 1]

# Calculate metrics
test_accuracy_cross = accuracy_score(y_test, y_test_pred_cross)
test_precision_cross = precision_score(y_test, y_test_pred_cross)
test_recall_cross = recall_score(y_test, y_test_pred_cross)
test_f1_cross = f1_score(y_test, y_test_pred_cross)
test_auc_cross = roc_auc_score(y_test, y_test_pred_proba_cross)

print(f"\nCross-Validation Results (Synthetic→Real):")
print(f"  Accuracy:  {test_accuracy_cross:.4f}")
print(f"  Precision: {test_precision_cross:.4f}")
print(f"  Recall:    {test_recall_cross:.4f}")
print(f"  F1-Score:  {test_f1_cross:.4f}")
print(f"  AUC-ROC:   {test_auc_cross:.4f}")

print(f"\nConfusion Matrix (Synthetic→Real):")
cm_cross = confusion_matrix(y_test, y_test_pred_cross)
print(cm_cross)
print(f"\n  True Negatives:  {cm_cross[0, 0]}")
print(f"  False Positives: {cm_cross[0, 1]}")
print(f"  False Negatives: {cm_cross[1, 0]}")
print(f"  True Positives:  {cm_cross[1, 1]}")

print(f"\nDetailed Classification Report (Synthetic→Real):")
print(classification_report(y_test, y_test_pred_cross, target_names=label_encoder_y.classes_))

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n\n" + "="*70)
print("COMPREHENSIVE COMPARISON")
print("="*70)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Real→Real': [test_accuracy, test_precision, test_recall, test_f1, test_auc],
    'Synthetic→Synthetic': [test_accuracy_synth, test_precision_synth, test_recall_synth, test_f1_synth, test_auc_synth],
    'Synthetic→Real': [test_accuracy_cross, test_precision_cross, test_recall_cross, test_f1_cross, test_auc_cross],
})
comparison['Gap (Synth→Real vs Real→Real)'] = comparison['Synthetic→Real'] - comparison['Real→Real']

print("\nPerformance Comparison:")
print(comparison.to_string(index=False))

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print(f"1. Real→Real Performance (Baseline): F1={test_f1:.4f}, AUC={test_auc:.4f}")
print(f"2. Synthetic→Synthetic Performance: F1={test_f1_synth:.4f}, AUC={test_auc_synth:.4f}")
print(f"3. Synthetic→Real Performance: F1={test_f1_cross:.4f}, AUC={test_auc_cross:.4f}")
print(f"4. Transfer Gap (Synth→Real vs Real→Real): F1={test_f1_cross - test_f1:.4f}, AUC={test_auc_cross - test_auc:.4f}")
print("\nInterpretation:")
if abs(test_f1_cross - test_f1) < 0.05:
    print("✓ Synthetic data generalizes well - minimal performance gap!")
elif test_f1_cross > test_f1 * 0.9:
    print("✓ Synthetic data shows good quality - acceptable performance gap")
else:
    print("⚠ Significant performance gap - synthetic data may need improvement")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("="*70)