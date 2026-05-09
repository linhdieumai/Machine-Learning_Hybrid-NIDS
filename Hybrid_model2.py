import numpy as np
import joblib
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load components
print("Status: Loading models and test data...")
try:
    rf_model = joblib.load('supervised_rf_model.pkl')
    anomaly_package = joblib.load('unsupervised_kmeans_model.pkl')
    X_test = np.load('X_test_for_sup.npy') 
    y_test = np.load('y_test.npy', allow_pickle=True).astype(int)
    
    kmeans = anomaly_package['model']
    threshold = anomaly_package['threshold']
except FileNotFoundError as e:
    print(f"Error: Missing data or model files! ({e})")
    exit()

# 2. Hybrid Prediction Logic
def hybrid_predict_v4(X, rf_threshold=0.5):
    rf_probs = rf_model.predict_proba(X)[:, 1]
    
    print("Status: Calculating K-Means distances...")
    distances = np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)
    
    final_preds = []
    saved_by_kmeans = 0
    
    for i in range(len(X)):
        prob = rf_probs[i]
        dist = distances[i]
        
        if prob >= rf_threshold:
            final_preds.append(1)
        else:
            # Logic tối ưu: Chỉ cứu khi dist > threshold và RF có sự nghi ngờ nhẹ (prob > 0.2)
            if dist > threshold and prob > 0.2:
                final_preds.append(1)
                saved_by_kmeans += 1
            else:
                final_preds.append(0)
                
    return np.array(final_preds), saved_by_kmeans

# 3. Execute
print("Status: Executing Smart Hybrid Logic...")
y_pred_hybrid, total_saved = hybrid_predict_v4(X_test, rf_threshold=0.5)

# 4. Evaluation
acc = accuracy_score(y_test, y_pred_hybrid)
cm = confusion_matrix(y_test, y_pred_hybrid)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*55)
print("        SMART HYBRID NIDS: ACCURACY OPTIMIZED")
print("="*55)
print(f"Overall Accuracy:       {acc:.5f}")
print(f"True Positives:         {tp}")
print(f"False Positives:        {fp}")
print(f"False Normals (Missed): {fn}")
print("-" * 55)
print(f"K-MEANS DETECTION BOOST: {total_saved}")
print("="*55)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_hybrid, target_names=['Normal (0)', 'Attack (1)']))