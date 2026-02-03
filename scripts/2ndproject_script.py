import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CUSTOMER CHURN PREDICTION WITH XGBOOST")
print("="*80)

# Load the customer churn dataset
print("\nLoading customer churn dataset...")
data = pd.read_csv('/mnt/ssd1/projects/CTGAN/examples/csv/customer_churn.csv')
print(f"Dataset shape: {data.shape}")
print(f"\nColumn names: {list(data.columns)}")
print(f"\nFirst few rows:")
print(data.head())

# Check for class imbalance
print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)
print(f"\nTarget distribution (Churn):")
print(data['Churn'].value_counts())
print(f"\nClass distribution percentage:")
print(data['Churn'].value_counts(normalize=True) * 100)

# Check for missing values
print(f"\nMissing values per column:")
print(data.isnull().sum())

# Data preprocessing
print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

# Drop columns that are not useful for prediction
columns_to_drop = ['Names', 'Onboard_date', 'Location', 'Company']
print(f"Dropping columns: {columns_to_drop}")

X = data.drop(columns_to_drop + ['Churn'], axis=1)
y = data['Churn']

print(f"\nFeatures after dropping: {list(X.columns)}")
print(f"Feature types:\n{X.dtypes}")

# Handle 'Account_Manager' if it's categorical (it appears to be binary 0/1)
# Age, Total_Purchase, Years, Num_Sites are numerical
print(f"\nFeature statistics:")
print(X.describe())

# IMPORTANT: Shuffle the data because churned customers are at the top
print("\n" + "="*80)
print("SHUFFLING DATA (Addressing Class Imbalance Issue)")
print("="*80)
print("NOTE: All churned samples are at the top. Shuffling with stratification...")

# Use sklearn's shuffle
from sklearn.utils import shuffle
X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

print(f"Data shuffled successfully!")
print(f"First 10 target values after shuffle: {list(y_shuffled.head(10))}")

# Split data: 70% train, 15% validation, 15% test WITH STRATIFICATION
print("\n" + "="*80)
print("SPLITTING DATA WITH STRATIFICATION")
print("="*80)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_shuffled, y_shuffled, test_size=0.15, random_state=42, stratify=y_shuffled
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full
)  # 0.176 * 0.85 ≈ 0.15 of total

print(f"Train set size: {len(X_train)} ({len(X_train)/len(X_shuffled)*100:.1f}%)")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X_shuffled)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X_shuffled)*100:.1f}%)")

# Check class distribution in each set
print(f"\nTrain target distribution:")
print(pd.Series(y_train).value_counts())
print(f"Validation target distribution:")
print(pd.Series(y_val).value_counts())
print(f"Test target distribution:")
print(pd.Series(y_test).value_counts())

# Calculate scale_pos_weight for imbalanced data
neg_samples = (y_train == 0).sum()
pos_samples = (y_train == 1).sum()
scale_pos_weight = neg_samples / pos_samples
print(f"\nClass imbalance ratio (neg/pos): {scale_pos_weight:.2f}")
print(f"Will use scale_pos_weight={scale_pos_weight:.2f} in XGBoost")

# Hyperparameter optimization
print("\n" + "="*80)
print("HYPERPARAMETER OPTIMIZATION")
print("="*80)

# Define hyperparameter combinations optimized for imbalanced data
test_combinations = [
    {
        'max_depth': 4, 
        'learning_rate': 0.1, 
        'n_estimators': 200, 
        'min_child_weight': 3,
        'subsample': 0.8, 
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'gamma': 0.1
    },
    {
        'max_depth': 6, 
        'learning_rate': 0.05, 
        'n_estimators': 300, 
        'min_child_weight': 5,
        'subsample': 0.9, 
        'colsample_bytree': 0.9,
        'scale_pos_weight': scale_pos_weight,
        'gamma': 0.2
    },
    {
        'max_depth': 5, 
        'learning_rate': 0.1, 
        'n_estimators': 250, 
        'min_child_weight': 1,
        'subsample': 0.85, 
        'colsample_bytree': 0.85,
        'scale_pos_weight': scale_pos_weight,
        'gamma': 0
    },
    {
        'max_depth': 7, 
        'learning_rate': 0.03, 
        'n_estimators': 400, 
        'min_child_weight': 3,
        'subsample': 0.8, 
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'gamma': 0.3
    },
    {
        'max_depth': 5, 
        'learning_rate': 0.1, 
        'n_estimators': 200, 
        'min_child_weight': 2,
        'subsample': 0.9, 
        'colsample_bytree': 0.9,
        'scale_pos_weight': scale_pos_weight,
        'gamma': 0.1
    },
]

best_params = None
best_score = 0
best_model = None

print("\nTesting hyperparameter combinations...")
print("Using F1-Score as optimization metric (good for imbalanced data)\n")

for i, params in enumerate(test_combinations, 1):
    print(f"Testing combination {i}/{len(test_combinations)}")
    print(f"Parameters: max_depth={params['max_depth']}, lr={params['learning_rate']}, "
          f"n_est={params['n_estimators']}, gamma={params['gamma']}")
    
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
    val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_f1 = f1_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Validation Balanced Accuracy: {val_balanced_acc:.4f}")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall: {val_recall:.4f}")
    print(f"  Validation F1-Score: {val_f1:.4f}")
    print(f"  Validation AUC-ROC: {val_auc:.4f}")
    
    # Use F1 score as the optimization metric (balances precision and recall)
    if val_f1 > best_score:
        best_score = val_f1
        best_params = params
        best_model = model
        print(f"  *** New best model! ***")
    print()

print("="*80)
print("BEST HYPERPARAMETERS")
print("="*80)
print(f"Best validation F1-Score: {best_score:.4f}")
print(f"Best parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")

# Train final model with best parameters on train+validation data
print("\n" + "="*80)
print("TRAINING FINAL MODEL ON TRAIN+VALIDATION DATA")
print("="*80)
final_model = xgb.XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
final_model.fit(X_train_full, y_train_full, verbose=False)
print("Final model training complete!")

# Evaluate on test set
print("\n" + "="*80)
print("FINAL TEST SET RESULTS")
print("="*80)
y_test_pred = final_model.predict(X_test)
y_test_pred_proba = final_model.predict_proba(X_test)[:, 1]

# Calculate metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

print(f"\nTest Set Metrics:")
print(f"  Accuracy:          {test_accuracy:.4f}")
print(f"  Balanced Accuracy: {test_balanced_acc:.4f}")
print(f"  Precision:         {test_precision:.4f}")
print(f"  Recall:            {test_recall:.4f}")
print(f"  F1-Score:          {test_f1:.4f}")
print(f"  AUC-ROC:           {test_auc:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\n  True Negatives (correctly predicted non-churn):  {cm[0, 0]}")
print(f"  False Positives (incorrectly predicted churn):   {cm[0, 1]}")
print(f"  False Negatives (missed churn):                  {cm[1, 0]}")
print(f"  True Positives (correctly predicted churn):      {cm[1, 1]}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn']))

# Feature importance
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# Additional insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. Dataset had severe class imbalance: {scale_pos_weight:.2f}:1 (non-churn:churn)")
print(f"2. Used scale_pos_weight to handle imbalance")
print(f"3. Shuffled data to ensure proper train/val/test split")
print(f"4. Best model achieved F1-Score: {test_f1:.4f} on test set")
print(f"5. Recall (detecting churners): {test_recall:.4f}")
print(f"6. Precision (accuracy of churn predictions): {test_precision:.4f}")

if test_recall < 0.7:
    print("\n⚠ Warning: Low recall - model may be missing many churners!")
    print("   Consider adjusting decision threshold or using SMOTE for better balance.")
elif test_recall > 0.85:
    print("\n✓ Good recall - model is catching most churners!")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
