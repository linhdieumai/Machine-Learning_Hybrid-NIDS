import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class HybridIDS:
    """
    Hệ thống phát hiện xâm nhập lai (Hybrid Intrusion Detection System)
    Tích hợp Luồng 1 (Supervised - random forest) và Luồng 2 (Unsupervised - Anomaly)
    """
    def __init__(self, tree_model_path, anomaly_model_path):
        print("Đang tải các mô hình từ Thành viên 3 và 4...")
        # Tải mô hình Cây quyết định từ TV 3
        self.tree_model = joblib.load(tree_model_path)
        
        # Tải mô hình Dị thường từ TV 4
        # Lưu ý: Yêu cầu TV 4 khi lưu file .pkl hãy lưu dưới dạng Dictionary 
        # Ví dụ: joblib.dump({'model': kmeans, 'threshold': 1.5}, 'anomaly.pkl')
        anomaly_data = joblib.load(anomaly_model_path)
        self.anomaly_model = anomaly_data['model']
        self.threshold = anomaly_data['threshold']
        print("Tải mô hình thành công!")

    def predict_tree_only(self, X):
        """Kịch bản 1: Chỉ đi qua hàng rào lớp 1 (random forest)"""
        return self.tree_model.predict(X)

    def predict_integrated(self, X_sup, X_unsup):
        """
        Kịch bản 2: Hệ thống phòng thủ 2 lớp (Zero-day Scenario)
        Nhận 2 tập X riêng biệt vì mô hình Supervised và Unsupervised có thể 
        được scale dữ liệu theo cách khác nhau bởi Thành viên 2.
        """
        # Bước 1: Đi qua mô hình Học có giám sát (Random Forest/XGBoost)
        final_predictions = self.tree_model.predict(X_sup)
        
        # Bước 2: Lọc các gói tin bị đánh giá là Normal (0)
        normal_indices = np.where(final_predictions == 0)[0]
        
        if len(normal_indices) > 0:
            # Lấy đúng dữ liệu của các gói tin này ở định dạng dành cho Unsupervised
            X_normal_suspects = X_unsup[normal_indices]
            
            # Đẩy qua KMeans để đo khoảng cách
            distances = self.anomaly_model.transform(X_normal_suspects)[:, 0]
            anomalies_mask = (distances > self.threshold).astype(int)
            
            # Cập nhật kết quả: nếu KMeans báo dị thường (1), đè lên kết quả Normal (0) cũ
            final_predictions[normal_indices] = np.maximum(
                final_predictions[normal_indices], 
                anomalies_mask
            )
            
        return final_predictions
class Evaluator:
    """
    Lớp đánh giá hiệu năng và xuất báo cáo biểu đồ
    """
    @staticmethod
    def evaluate_and_compare(y_true, y_pred_tree, y_pred_integrated):
        # 1. Tính toán Ma trận nhầm lẫn
        cm_tree = confusion_matrix(y_true, y_pred_tree)
        cm_int = confusion_matrix(y_true, y_pred_integrated)

        # Trích xuất True Negative, False Positive, False Negative, True Positive
        tn1, fp1, fn1, tp1 = cm_tree.ravel()
        tn2, fp2, fn2, tp2 = cm_int.ravel()

        # Tính toán các chỉ số thống kê cơ bản
        fpr_tree = fp1 / (fp1 + tn1) if (fp1 + tn1) > 0 else 0 # Tỷ lệ báo nhầm
        fpr_int = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0
        
        # In báo cáo ra Terminal
        print("\n" + "="*40)
        print("BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG HỆ THỐNG")
        print("="*40)
        print(f"1. CHỈ DÙNG LUỒNG 1 (Random Forest):")
        print(f"   - Số cuộc tấn công lọt lưới (False Negatives): {fn1} gói tin")
        print(f"   - Số gói tin sạch bị chặn nhầm (False Positives): {fp1} gói tin")
        print(f"   - Tỷ lệ báo động nhầm (FPR): {fpr_tree:.2%}")
        
        print(f"\n2. HỆ THỐNG TÍCH HỢP (Luồng 1 + Luồng 2):")
        print(f"   - Số cuộc tấn công lọt lưới (False Negatives): {fn2} gói tin")
        print(f"     => THÀNH TÍCH: Đã chặn thêm được {fn1 - fn2} cuộc tấn công lạ (Zero-day)!")
        print(f"   - Số gói tin sạch bị chặn nhầm (False Positives): {fp2} gói tin")
        print(f"     => TRADE-OFF: Tỷ lệ báo động nhầm tăng {(fpr_int - fpr_tree):.2%} so với ban đầu.")
        print("="*40)

        # 2. Hiển thị Biểu đồ
        Evaluator.plot_confusion_matrices(cm_tree, cm_int)
        Evaluator.plot_tradeoff(fn1, fn2, fp1, fp2)

    @staticmethod
    def plot_confusion_matrices(cm_tree, cm_int):
        """Vẽ 2 heatmap để so sánh Ma trận nhầm lẫn"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        labels = ['Normal (0)', 'Attack (1)']
        
        sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=axes[0], annot_kws={"size": 14})
        axes[0].set_title('Random Forest Only\n(Bỏ lọt nhiều tấn công lạ)', fontsize=14)
        axes[0].set_ylabel('Nhãn Thực Tế', fontsize=12)
        axes[0].set_xlabel('Mô Hình Dự Đoán', fontsize=12)
        
        sns.heatmap(cm_int, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=labels, yticklabels=labels, ax=axes[1], annot_kws={"size": 14})
        axes[1].set_title('Integrated System (Random Forest + Anomaly)\n(Bắt được tấn công lạ nhưng tăng báo nhầm)', fontsize=14)
        axes[1].set_ylabel('Nhãn Thực Tế', fontsize=12)
        axes[1].set_xlabel('Mô Hình Dự Đoán', fontsize=12)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_tradeoff(fn_tree, fn_int, fp_tree, fp_int):
        """Vẽ biểu đồ cột phân tích sự đánh đổi (Trade-off)"""
        labels = ['Bỏ lọt tấn công (FN)\n(Càng thấp càng an toàn)', 
                  'Báo động nhầm (FP)\n(Sự đánh đổi: Gây phiền nhiễu)']
        tree_scores = [fn_tree, fp_tree]
        int_scores = [fn_int, fp_int]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 6))
        rects1 = ax.bar(x - width/2, tree_scores, width, label='Chỉ dùng Random Forest', color='#3498db')
        rects2 = ax.bar(x + width/2, int_scores, width, label='Hệ thống Tích hợp', color='#e74c3c')

        ax.set_ylabel('Số lượng gói tin', fontsize=12)
        ax.set_title('Phân tích Sự Đánh Đổi (Trade-off Analysis)', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.legend(fontsize=12)

        # Thêm số liệu lên đỉnh các cột
        ax.bar_label(rects1, padding=3, fontsize=11)
        ax.bar_label(rects2, padding=3, fontsize=11)

        plt.tight_layout()
        plt.show()

# ==========================================
# KHỐI LỆNH THỰC THI VỚI DỮ LIỆU THẬT
# ==========================================
if __name__ == "__main__":
    print("Đang tải tập dữ liệu NumPy từ Thành viên 2...")
    # Tải các ma trận đặc trưng (Features)
    X_test_sup = np.load('X_test_for_sup.npy')
    X_test_unsup = np.load('X_test_for_unsup.npy')
    
    # Tải nhãn thực tế (Labels)
    y_test = np.load('y_test.npy')
    
    print(f"Kích thước tập Test Supervised: {X_test_sup.shape}")
    print(f"Kích thước tập Test Unsupervised: {X_test_unsup.shape}")
    print(f"Số lượng nhãn Test: {y_test.shape}")

    print("\nKhởi tạo Hệ thống Phát hiện xâm nhập...")
    # Khởi tạo class với tên file model thực tế
    # Bạn có thể đổi 'supervised_rf_model.pkl' thành 'supervised_xgb_model.pkl' nếu muốn test thử model nào tốt hơn
    ids = HybridIDS('supervised_rf_model.pkl', 'unsupervised_kmeans_model.pkl')
    
    # Kịch bản 1: Chỉ chạy luồng Học có giám sát
    print("\nTiến hành chạy kịch bản 1: Random Forest Only...")
    preds_tree = ids.predict_tree_only(X_test_sup)
    
    # Kịch bản 2: Chạy hệ thống tích hợp
    print("Tiến hành chạy kịch bản 2: Hệ thống Tích hợp 2 lớp (RF + KMeans)...")
    # Truyền cả 2 tập X vào để các mô hình dùng đúng định dạng dữ liệu
    preds_integrated = ids.predict_integrated(X_test_sup, X_test_unsup)
    
    # Chạy Evaluator để xuất biểu đồ và báo cáo
    Evaluator.evaluate_and_compare(y_test, preds_tree, preds_integrated)