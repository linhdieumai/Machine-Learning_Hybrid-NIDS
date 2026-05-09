import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load preprocessed data
print("Status: Loading data...")
try:
    X_train = np.load('X_sup_train.npy')
    y_train = np.load('y_sup_train.npy', allow_pickle=True).astype(int)
    X_test = np.load('X_test_for_sup.npy')
    y_test = np.load('y_test.npy', allow_pickle=True).astype(int)
except FileNotFoundError as e:
    print(f"Error: Missing .npy files. Run preprocessing first. ({e})")
    exit()

# 2. Initialize and Train XGBoost
# Settings optimized for SOICT project datasets
print("Status: Training XGBoost with optimized parameters...")
xgb_model = XGBClassifier(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=8, 
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='binary:logistic',
    eval_metric='logloss',
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# 3. Save model
joblib.dump(xgb_model, 'supervised_xgb_model.pkl')
print("Status: Model saved as 'supervised_xgb_model.pkl'")

# 4. Evaluation with Custom Threshold
custom_threshold = 0.45 
y_probs = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= custom_threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*55)
print(f"      RESULTS: XGBOOST OPTIMIZED (THRESHOLD: {custom_threshold})")
print("="*55)
print(f"Overall Accuracy:       {acc:.5f}")
print(f"True Positives:         {tp}")
print(f"False Positives:        {fp} ")
print(f"False Normals (Missed): {fn} ")
print("-" * 55)
print(f"Note: Standard XGBoost execution completed.")
print("="*55)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)']))