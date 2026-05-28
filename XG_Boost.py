import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================================================
# 1. Load preprocessed data
# =========================================================
print("Status: Loading data...")

X_train = np.load('X_sup_train.npy')
y_train = np.load('y_sup_train.npy', allow_pickle=True).astype(int)

X_test = np.load('X_test_for_sup.npy')
y_test = np.load('y_test.npy', allow_pickle=True).astype(int)

# =========================================================
# 2. Train Final XGBoost Model
# =========================================================
print("Status: Training final XGBoost model...")

xgb_model = XGBClassifier(
    n_estimators=100,
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

# =========================================================
# 3. Prediction
# =========================================================
custom_threshold = 0.45

y_probs = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= custom_threshold).astype(int)

# =========================================================
# 4. Accuracy
# =========================================================
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("FINAL RESULTS - XGBOOST")
print("=" * 60)

print(f"Overall Accuracy: {accuracy:.5f}")

# =========================================================
# 5. Classification Report
# =========================================================
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=['Not Fraud', 'Fraud']
    )
)

# =========================================================
# 6. Confusion Matrix
# =========================================================
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Not Fraud', 'Fraud']
)

disp.plot(
    cmap='Blues',
    ax=ax,
    colorbar=True
)

plt.title('Confusion Matrix - XGBoost')

plt.savefig('xgboost_confusion_matrix.png', dpi=300)

plt.show()