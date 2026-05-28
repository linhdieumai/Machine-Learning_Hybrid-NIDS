import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class HybridIDS:
    """
    Hệ thống phát hiện xâm nhập lai (Hybrid Intrusion Detection System)
    Tích hợp Luồng 1 (Supervised - SVM) và Luồng 2 (Unsupervised - Anomaly)
    """
    def __init__(self, sup_model_path, anomaly_model_path):
        print("Downloading model from members 3 and 4...")
        # Tải mô hình Học có giám sát (SVM)
        self.sup_model = joblib.load(sup_model_path)
        
        # Tải mô hình Dị thường từ TV 4
        anomaly_data = joblib.load(anomaly_model_path)
        self.anomaly_model = anomaly_data['model']
        self.threshold = anomaly_data['threshold']
        print("Install model successfully!")

    def predict_sup_only(self, X):
        """Kịch bản 1: Chỉ đi qua hàng rào lớp 1 (SVM)"""
        return self.sup_model.predict(X)

    def predict_integrated(self, X_sup, X_unsup):
        """Kịch bản 2: Hệ thống phòng thủ 2 lớp (Zero-day Scenario)"""
        # Bước 1: Đi qua mô hình Học có giám sát (SVM)
        final_predictions = self.sup_model.predict(X_sup)
        
        # Bước 2: Lọc các gói tin bị đánh giá là Normal (0)
        normal_indices = np.where(final_predictions == 0)[0]
        
        if len(normal_indices) > 0:
            X_normal_suspects = X_unsup[normal_indices]
            
            # --- LOGIC CHO ISOLATION FOREST ---
            iso_preds = self.anomaly_model.predict(X_normal_suspects)
            anomalies_mask = (iso_preds == -1).astype(int)
            
            # Cập nhật kết quả
            final_predictions[normal_indices] = np.maximum(
                final_predictions[normal_indices], 
                anomalies_mask
            )
            
        return final_predictions

class Evaluator:
    """Lớp đánh giá hiệu năng và xuất báo cáo biểu đồ"""
    @staticmethod
    def evaluate_and_compare(y_true, y_pred_sup, y_pred_integrated):
        cm_sup = confusion_matrix(y_true, y_pred_sup)
        cm_int = confusion_matrix(y_true, y_pred_integrated)

        tn1, fp1, fn1, tp1 = cm_sup.ravel()
        tn2, fp2, fn2, tp2 = cm_int.ravel()

        fpr_sup = fp1 / (fp1 + tn1) if (fp1 + tn1) > 0 else 0 
        fpr_int = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0
        
        print("\n" + "="*40)
        print("SYSTEM PERFORMANCE EVALUATION REPORT")
        print("="*40)
        print(f"1. PIPELINE 1 ONLY (SVM):")
        print(f"   - Missed attacks (False Negatives): {fn1} packets")
        print(f"   - Legitimate packets blocked (False Positives): {fp1} packets")
        print(f"   - False Positive Rate (FPR): {fpr_sup:.2%}")
                
        print(f"\n2. INTEGRATED SYSTEM (SVM + IsoForest):")
        print(f"   - Missed attacks (False Negatives): {fn2} packets")
        print(f"     => ACHIEVEMENT: Blocked {fn1 - fn2} additional unknown attacks!")
        print(f"   - Legitimate packets blocked (False Positives): {fp2} packets")
        print(f"     => TRADE-OFF: False Positive Rate increased by {(fpr_int - fpr_sup):.2%}.")
        print("="*40)

        Evaluator.plot_confusion_matrices(cm_sup, cm_int)
        Evaluator.plot_tradeoff(fn1, fn2, fp1, fp2)

    @staticmethod
    def plot_confusion_matrices(cm_sup, cm_int):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        labels = ['Normal (0)', 'Attack (1)']
        
        sns.heatmap(cm_sup, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=axes[0], annot_kws={"size": 14})
        axes[0].set_title('SVM Only\n(Misses many unknown attacks)', fontsize=14)
        axes[0].set_ylabel('Actual Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        sns.heatmap(cm_int, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=labels, yticklabels=labels, ax=axes[1], annot_kws={"size": 14})
        axes[1].set_title('Integrated System (SVM + IsoForest)\n(Catches unknown attacks but increases FP)', fontsize=14)
        axes[1].set_ylabel('Actual Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_tradeoff(fn_sup, fn_int, fp_sup, fp_int):
        labels = ['False Negative\n(Lower is safer)', 'False Positive\n(Trade-off)']
        sup_scores = [fn_sup, fp_sup]
        int_scores = [fn_int, fp_int]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        rects1 = ax.bar(x - width/2, sup_scores, width, label='SVM Only', color='#3498db')
        rects2 = ax.bar(x + width/2, int_scores, width, label='The Integrated System', color='#e74c3c')

        ax.set_ylabel('Number of Packets', fontsize=12)
        ax.set_title('Trade-off Analysis', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=12)
        ax.bar_label(rects1, padding=3, fontsize=11)
        ax.bar_label(rects2, padding=3, fontsize=11)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Loading NumPy datasets from Member 2...")
    X_test_sup = np.load('X_test_for_sup.npy')
    X_test_unsup = np.load('X_test_for_unsup.npy')
    y_test = np.load('y_test.npy')

    print("\nInitializing Intrusion Detection System...")
    # LƯU Ý QUAN TRỌNG: Gọi file .pkl của SVM ở đây
    ids = HybridIDS('supervised_svm_model.pkl', 'unsupervised_isoforest_model.pkl')
    
    print("\nRunning Scenario 1: SVM Only...")
    preds_sup = ids.predict_sup_only(X_test_sup)
    
    print("Running Scenario 2: 2-Layer Integrated System (SVM + IsoForest)...")
    preds_integrated = ids.predict_integrated(X_test_sup, X_test_unsup)
    
    Evaluator.evaluate_and_compare(y_test, preds_sup, preds_integrated)