import numpy as np
import joblib
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load models and test data
print("Status: Loading XGBoost and K-Means models...")
try:
    xgb_model = joblib.load('supervised_xgb_model.pkl')
    kmeans_package = joblib.load('unsupervised_kmeans_model.pkl')
    X_test = np.load('X_test_for_sup.npy')
    y_test = np.load('y_test.npy', allow_pickle=True).astype(int)

    kmeans = kmeans_package['model']
    threshold = kmeans_package['threshold']
except FileNotFoundError as e:
    print(f"Error: Missing model or data files! ({e})")
    exit()

# 2. Calculate probabilities and distances
print("Status: Calculating probabilities from XGBoost...")
y_probs = xgb_model.predict_proba(X_test)[:, 1]

print("Status: Calculating K-Means anomaly scores...")
distances = np.min(cdist(X_test, kmeans.cluster_centers_, 'euclidean'), axis=1)

# 3. Smart Hybrid Logic
final_preds = []
saved_by_kmeans = 0
xgb_threshold = 0.45 

for i in range(len(y_probs)):
    prob = y_probs[i]
    dist = distances[i]
    
    # Step A: Trust XGBoost if it's confident about an Attack
    if prob >= xgb_threshold:
        final_preds.append(1)
    
    # Step B: Collaborative filtering for potential Missed Attacks
    else:
        # Only recover if K-Means detects anomaly AND XGBoost has slight suspicion
        if dist > threshold and prob > 0.15:
            final_preds.append(1)
            saved_by_kmeans += 1
        else:
            final_preds.append(0)

# 4. Output Results
acc = accuracy_score(y_test, final_preds)
cm = confusion_matrix(y_test, final_preds)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*55)
print("        HYBRID SYSTEM: XGBOOST + K-MEANS")
print("="*55)
print(f"Overall Accuracy:       {acc:.5f}")
print(f"True Positives:         {tp}")
print(f"False Positives:        {fp} ")
print(f"False Normals (Missed): {fn} ")
print("-" * 55)
print(f"K-MEANS DETECTION BOOST: {saved_by_kmeans}")
print("="*55)

print("\nDetailed Classification Report:")
print(classification_report(y_test, final_preds, target_names=['Normal (0)', 'Attack (1)']))