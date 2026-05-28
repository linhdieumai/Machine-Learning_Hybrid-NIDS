import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================
# 1. LOAD DATA
# =====================================
print("Loading data...")

X_train = np.load('X_sup_train.npy')
y_train = np.load('y_sup_train.npy', allow_pickle=True).astype(int)

X_test = np.load('X_test_for_sup.npy')
y_test = np.load('y_test.npy', allow_pickle=True).astype(int)

# =====================================
# 2. FEATURE SCALING
# =====================================
print("Scaling features...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# 3. TRAIN FIXED SVM MODEL
# =====================================
print("\nTraining final SVM model...")

model = SVC(
    C=1,
    kernel='rbf',
    gamma=0.01,   # FIXED (best from previous experiment)
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# 4. PREDICTION
# =====================================
y_pred = model.predict(X_test)

# =====================================
# 5. CLASSIFICATION REPORT
# =====================================
print("\n================ CLASSIFICATION REPORT ================")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Fraud", "Fraud"]
))

# =====================================
# 6. CONFUSION MATRIX
# =====================================
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n================ CONFUSION MATRIX ================")
print(cm)

print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# =====================================
# 7. METRICS SUMMARY
# =====================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n================ METRICS SUMMARY ================")
print(f"Accuracy : {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Recall   : {recall:.5f}")
print(f"F1-score : {f1:.5f}")

# =====================================
# 8. CONFUSION MATRIX VISUALIZATION
# =====================================
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Fraud", "Fraud"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.show()