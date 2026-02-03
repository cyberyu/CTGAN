import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score
)
from sklearn.utils import shuffle
from ctgan import CTGAN
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CUSTOMER CHURN PREDICTION - SYNTHETIC DATA APPROACH")
print("Simulating Privacy-Preserving ML: Client 3 Holdout Strategy")
print("="*80)

# Load the customer churn dataset
print("\nLoading customer churn dataset...")
data = pd.read_csv('/mnt/ssd1/projects/CTGAN/examples/csv/customer_churn.csv')
print(f"Dataset shape: {data.shape}")

# Shuffle data first
data_shuffled = shuffle(data, random_state=42).reset_index(drop=True)

# Step 1: Select 20% of customers as "Client 3" (holdout/not-usable data)
print("\n" + "="*80)
print("STEP 1: SPLITTING DATA - SIMULATING CLIENT 3 HOLDOUT")
print("="*80)

unique_names = data_shuffled['Names'].unique()
n_client3 = int(len(unique_names) * 0.2)
np.random.seed(42)
client3_names = np.random.choice(unique_names, n_client3, replace=False)

# Split into Client 1+2 (training available) and Client 3 (holdout)
client3_data = data_shuffled[data_shuffled['Names'].isin(client3_names)].copy()
client12_data = data_shuffled[~data_shuffled['Names'].isin(client3_names)].copy()

# Add client identifier column
client3_data['client_id'] = 3
client12_data['client_id'] = 1  # Treat as single client for simplicity

print(f"Total unique customers: {len(unique_names)}")
print(f"Client 3 (holdout) customers: {len(client3_names)} ({len(client3_names)/len(unique_names)*100:.1f}%)")
print(f"Client 1+2 (available) customers: {len(unique_names) - len(client3_names)}")
print(f"\nClient 3 records: {len(client3_data)} ({len(client3_data)/len(data_shuffled)*100:.1f}%)")
print(f"Client 1+2 records: {len(client12_data)} ({len(client12_data)/len(data_shuffled)*100:.1f}%)")
print(f"\nClient 3 churn distribution:")
print(client3_data['Churn'].value_counts())
print(f"\nClient 1+2 churn distribution:")
print(client12_data['Churn'].value_counts())

# Step 2: Prepare data for CTGAN (ALL real data including Client 3)
print("\n" + "="*80)
print("STEP 2: PREPARING DATA FOR CTGAN")
print("="*80)

# Combine all data for CTGAN training
all_data = pd.concat([client12_data, client3_data], axis=0).reset_index(drop=True)

# Drop columns not useful for modeling but keep Names temporarily
columns_to_drop = ['Onboard_date', 'Location', 'Company']
ctgan_data = all_data.drop(columns_to_drop, axis=1)

print(f"Combined dataset for CTGAN: {ctgan_data.shape}")
print(f"Columns: {list(ctgan_data.columns)}")

# Identify discrete columns for CTGAN
discrete_columns = ['client_id', 'Churn', 'Account_Manager']  # Binary/categorical columns
print(f"\nDiscrete columns for CTGAN: {discrete_columns}")

# Step 3: Train CTGAN on ALL real data (Client 1, 2, AND 3)
print("\n" + "="*80)
print("STEP 3: TRAINING CTGAN ON ALL REAL DATA")
print("="*80)
print("NOTE: CTGAN sees Client 3 data to learn patterns, but classifier will NOT!")
print("Training with 600 epochs for high-quality synthetic data...")

# Remove Names column for CTGAN (not useful for generation)
ctgan_training_data = ctgan_data.drop('Names', axis=1)

ctgan = CTGAN(epochs=600, verbose=True, batch_size=100)
ctgan.fit(ctgan_training_data, discrete_columns=discrete_columns)

print("\nCTGAN training complete!")

# Step 4: Generate synthetic data
print("\n" + "="*80)
print("STEP 4: GENERATING SYNTHETIC DATA")
print("="*80)

# Generate synthetic data (2x the original size for better coverage)
n_synthetic = len(all_data) * 2
print(f"Generating {n_synthetic} synthetic samples...")
synthetic_data = ctgan.sample(n_synthetic)

print(f"Synthetic data shape: {synthetic_data.shape}")
print(f"\nSynthetic data client distribution:")
print(synthetic_data['client_id'].value_counts())
print(f"\nSynthetic data churn distribution:")
print(synthetic_data['Churn'].value_counts())

# Step 5: Train classifier ONLY on synthetic data
print("\n" + "="*80)
print("STEP 5: TRAINING CLASSIFIER ON SYNTHETIC DATA ONLY")
print("="*80)

# Prepare synthetic data
X_synthetic = synthetic_data.drop(['Churn', 'client_id'], axis=1)
y_synthetic = synthetic_data['Churn']

# Split synthetic data for training/validation
X_train_syn, X_val_syn, y_train_syn, y_val_syn = train_test_split(
    X_synthetic, y_synthetic, test_size=0.2, random_state=42, stratify=y_synthetic
)

print(f"Synthetic training set: {len(X_train_syn)}")
print(f"Synthetic validation set: {len(X_val_syn)}")

# Calculate scale_pos_weight for imbalanced data
neg_samples = (y_train_syn == 0).sum()
pos_samples = (y_train_syn == 1).sum()
scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1
print(f"\nSynthetic data class imbalance ratio: {scale_pos_weight:.2f}")

# Train XGBoost on synthetic data with optimized hyperparameters
print("\nTraining XGBoost classifier on synthetic data...")
model_synthetic = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=250,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    gamma=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

model_synthetic.fit(X_train_syn, y_train_syn, verbose=False)
print("Model trained on synthetic data!")

# Step 6: Evaluate on REAL Client 3 data (the holdout set)
print("\n" + "="*80)
print("STEP 6: TESTING ON REAL CLIENT 3 DATA (HOLDOUT)")
print("="*80)
print("This simulates predicting for a client whose real data was never seen by classifier")

# Prepare Client 3 test data
X_client3 = client3_data.drop(['Names', 'Onboard_date', 'Location', 'Company', 'Churn', 'client_id'], axis=1)
y_client3 = client3_data['Churn']

print(f"\nClient 3 test set size: {len(X_client3)}")
print(f"Client 3 churn distribution: {y_client3.value_counts().to_dict()}")

# Predict on Client 3
y_client3_pred = model_synthetic.predict(X_client3)
y_client3_pred_proba = model_synthetic.predict_proba(X_client3)[:, 1]

# Calculate metrics
test_accuracy = accuracy_score(y_client3, y_client3_pred)
test_balanced_acc = balanced_accuracy_score(y_client3, y_client3_pred)
test_precision = precision_score(y_client3, y_client3_pred, zero_division=0)
test_recall = recall_score(y_client3, y_client3_pred, zero_division=0)
test_f1 = f1_score(y_client3, y_client3_pred, zero_division=0)
test_auc = roc_auc_score(y_client3, y_client3_pred_proba)

print(f"\n{'='*80}")
print("RESULTS: SYNTHETIC-TRAINED MODEL ON REAL CLIENT 3 DATA")
print(f"{'='*80}")
print(f"\nPerformance Metrics:")
print(f"  Accuracy:          {test_accuracy:.4f}")
print(f"  Balanced Accuracy: {test_balanced_acc:.4f}")
print(f"  Precision:         {test_precision:.4f}")
print(f"  Recall:            {test_recall:.4f}")
print(f"  F1-Score:          {test_f1:.4f}")
print(f"  AUC-ROC:           {test_auc:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_client3, y_client3_pred)
print(cm)
print(f"\n  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")

print(f"\nDetailed Classification Report:")
print(classification_report(y_client3, y_client3_pred, target_names=['No Churn', 'Churn'], zero_division=0))

# Step 7: Baseline comparison - Train on real Client 1+2, test on real Client 3
print("\n" + "="*80)
print("STEP 7: BASELINE COMPARISON - REAL DATA APPROACH")
print("="*80)
print("Training on REAL Client 1+2 data, testing on REAL Client 3 data")

X_client12 = client12_data.drop(['Names', 'Onboard_date', 'Location', 'Company', 'Churn', 'client_id'], axis=1)
y_client12 = client12_data['Churn']

# Split Client 1+2 for training/validation
X_train_real, X_val_real, y_train_real, y_val_real = train_test_split(
    X_client12, y_client12, test_size=0.2, random_state=42, stratify=y_client12
)

neg_samples_real = (y_train_real == 0).sum()
pos_samples_real = (y_train_real == 1).sum()
scale_pos_weight_real = neg_samples_real / pos_samples_real if pos_samples_real > 0 else 1

print(f"Real training set: {len(X_train_real)}")
print(f"Real class imbalance ratio: {scale_pos_weight_real:.2f}")

model_real = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=250,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight_real,
    gamma=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

model_real.fit(X_train_real, y_train_real, verbose=False)
print("Model trained on real Client 1+2 data!")

# Test on Client 3
y_client3_pred_real = model_real.predict(X_client3)
y_client3_pred_proba_real = model_real.predict_proba(X_client3)[:, 1]

test_accuracy_real = accuracy_score(y_client3, y_client3_pred_real)
test_balanced_acc_real = balanced_accuracy_score(y_client3, y_client3_pred_real)
test_precision_real = precision_score(y_client3, y_client3_pred_real, zero_division=0)
test_recall_real = recall_score(y_client3, y_client3_pred_real, zero_division=0)
test_f1_real = f1_score(y_client3, y_client3_pred_real, zero_division=0)
test_auc_real = roc_auc_score(y_client3, y_client3_pred_proba_real)

print(f"\nBaseline Performance (Real→Real):")
print(f"  Accuracy:          {test_accuracy_real:.4f}")
print(f"  Balanced Accuracy: {test_balanced_acc_real:.4f}")
print(f"  Precision:         {test_precision_real:.4f}")
print(f"  Recall:            {test_recall_real:.4f}")
print(f"  F1-Score:          {test_f1_real:.4f}")
print(f"  AUC-ROC:           {test_auc_real:.4f}")

# Final comparison
print("\n" + "="*80)
print("FINAL COMPARISON: SYNTHETIC VS REAL DATA APPROACH")
print("="*80)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Synthetic Approach': [test_accuracy, test_balanced_acc, test_precision, test_recall, test_f1, test_auc],
    'Real Data Baseline': [test_accuracy_real, test_balanced_acc_real, test_precision_real, test_recall_real, test_f1_real, test_auc_real],
})
comparison['Difference'] = comparison['Synthetic Approach'] - comparison['Real Data Baseline']

print(f"\n{comparison.to_string(index=False)}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print(f"1. CTGAN was trained on ALL real data (including Client 3)")
print(f"2. Classifier was trained ONLY on synthetic data")
print(f"3. Both models tested on real Client 3 holdout data")
print(f"4. Performance gap (Synthetic vs Real): {test_f1 - test_f1_real:.4f} F1-Score")

if abs(test_f1 - test_f1_real) < 0.05:
    print(f"\n✓ EXCELLENT: Synthetic approach performs nearly as well as real data!")
elif abs(test_f1 - test_f1_real) < 0.10:
    print(f"\n✓ GOOD: Synthetic approach shows acceptable performance drop")
else:
    print(f"\n⚠ CAUTION: Significant performance gap - consider more CTGAN epochs or TVAE")

print(f"\n5. This demonstrates CTGAN's 'Machine Learning Efficacy' metric")
print(f"6. Privacy benefit: Classifier never saw real Client 3 data directly")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
