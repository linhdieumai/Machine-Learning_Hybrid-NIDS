import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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
print("Status: Loading data...")

X_train = np.load('X_sup_train.npy')
y_train = np.load('y_sup_train.npy', allow_pickle=True).astype(int)

X_test = np.load('X_test_for_sup.npy')
y_test = np.load('y_test.npy', allow_pickle=True).astype(int)

# =====================================
# 2. HYPERPARAMETER TUNING
# =====================================
tree_counts = [5, 10, 20, 50, 100]

recall_scores = []
accuracy_scores = []
training_times = []

best_model = None
best_recall = -1

# =====================================
# 3. TRAINING LOOP
# =====================================
for n_trees in tree_counts:

    print(f"\nTraining Random Forest with {n_trees} trees...")

    start_time = time.time()

    rf_model = RandomForestClassifier(
        n_estimators=20,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = rf_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    accuracy_scores.append(acc)
    recall_scores.append(recall)
    training_times.append(training_time)

    print(f"Accuracy: {acc:.5f}")
    print(f"Recall:   {recall:.5f}")
    print(f"Time:     {training_time:.2f}s")

    # save best model (based on recall)
    if recall > best_recall:
        best_recall = recall
        best_model = rf_model

# =====================================
# 4. FINAL PREDICTION (BEST MODEL)
# =====================================
y_pred_best = best_model.predict(X_test)

# =====================================
# 5. CONFUSION MATRIX
# =====================================
cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()

print("\n================ CONFUSION MATRIX ================")
print(cm)

print("\n===== CONFUSION MATRIX DETAILS =====")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

# =====================================
# 6. FULL CLASSIFICATION REPORT
# =====================================
print("\n================ CLASSIFICATION REPORT ================")
print(classification_report(
    y_test,
    y_pred_best,
    target_names=['Not Fraud', 'Fraud']
))

# =====================================
# 7. METRICS SUMMARY
# =====================================
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)

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
    display_labels=['Not Fraud', 'Fraud']
)

disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# =====================================
# 9. PLOT TUNING RESULTS
# =====================================
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_xlabel('Number of Trees')
ax1.set_ylabel('Recall Score', color='red')
ax1.plot(tree_counts, recall_scores, marker='o', color='red', linewidth=2.5)
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_xticks(tree_counts)

ax2 = ax1.twinx()
ax2.set_ylabel('Training Time (s)', color='blue')
ax2.plot(tree_counts, training_times, marker='o', color='blue', linewidth=2.5)
ax2.tick_params(axis='y', labelcolor='blue')

plt.title("Random Forest Tuning Results")
plt.tight_layout()
plt.show()