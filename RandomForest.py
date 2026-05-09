import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load data
print("Status: Loading data...")
try:
    X_train = np.load('X_sup_train.npy')
    y_train = np.load('y_sup_train.npy', allow_pickle=True).astype(int)
    X_test = np.load('X_test_for_sup.npy')
    y_test = np.load('y_test.npy', allow_pickle=True).astype(int)
except FileNotFoundError as e:
    print(f"Error: .npy data files not found! ({e})")
    exit()

# 2. Training
print(f"Status: Training Random Forest")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    min_samples_split=2,
    random_state=42,
    n_jobs=-1 
)
rf_model.fit(X_train, y_train)

# 3. Save model
joblib.dump(rf_model, 'supervised_rf_model.pkl')

# 4. Evaluation
y_pred_rf = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred_rf)
cm = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*55)
print("           RESULTS: RANDOM FOREST")
print("="*55)
print(f"Overall Accuracy:       {acc:.5f}")
print(f"True Positives:         {tp}")
print(f"False Positives:        {fp}")
print(f"False Normals (Missed): {fn}")
print("-" * 55)
print(f"Note: Standard RF model execution completed.")
print("="*55)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Normal (0)', 'Attack (1)']))