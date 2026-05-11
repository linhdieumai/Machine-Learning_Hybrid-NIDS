import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
import os
 
def train_anomaly_detector():
    # 1. Định vị thư mục làm việc hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Hệ thống đang làm việc tại: {current_dir}")
 
    # 2. Tải dữ liệu huấn luyện
    print("Đang tải dữ liệu Normal từ khâu Tiền xử lý...")
    data_path = os.path.join(current_dir, 'X_unsup_train.npy')
    X_train_normal = np.load(data_path)
    print(f"-> Đã tải dữ liệu: {X_train_normal.shape[0]} mẫu, {X_train_normal.shape[1]} đặc trưng")
 
    # 3. Huấn luyện thuật toán Isolation Forest
    print("Đang huấn luyện mô hình Isolation Forest (Học không giám sát)...")
    #
    # contamination=0.01: Ước tính ~1% dữ liệu huấn luyện là bất thường (nhiễu).
    #   - Giá trị này ảnh hưởng trực tiếp đến threshold nội bộ của model.
    #   - Nếu FN vẫn còn nhiều -> tăng contamination (ví dụ: 0.03, 0.05).
    #   - Nếu FP quá nhiều   -> giảm contamination (ví dụ: 0.005).
    #
    # n_estimators=200: Số lượng cây Isolation (nhiều hơn KMeans n_clusters=5 cũ)
    #   -> Mô hình ổn định và chính xác hơn.
    #
    # max_samples='auto': Mỗi cây lấy mẫu 256 điểm (mặc định của sklearn)
    #   -> Đủ để phát hiện bất thường mà không cần toàn bộ tập dữ liệu.
    #
    iso_forest = IsolationForest(
        contamination=0.01,
        n_estimators=200,
        max_samples='auto',
        random_state=42,
        n_jobs=-1  # Dùng toàn bộ CPU để huấn luyện nhanh hơn
    )
    iso_forest.fit(X_train_normal)
    print("-> Huấn luyện Isolation Forest hoàn tất!")
 
    # 4. Tính toán Threshold từ anomaly score
    print("Đang tính toán Ngưỡng giới hạn (Threshold)...")
    #
    # decision_function() trả về anomaly score:
    #   - Score < 0  : Điểm bất thường (càng âm = càng nguy hiểm)
    #   - Score >= 0 : Điểm bình thường
    #
    # Ta dùng percentile 2% (thay vì 99% của KMeans cũ) để thiết lập threshold
    # nghiêm ngặt hơn -> Giảm False Negative.
    #
    scores = iso_forest.decision_function(X_train_normal)
    threshold = np.percentile(scores, 2)
    print(f"-> Ngưỡng Anomaly Score (Threshold) được thiết lập: {threshold:.4f}")
    print(f"   (Gói tin có score < {threshold:.4f} sẽ bị đánh dấu là TẤN CÔNG)")
 
    # 5. Kiểm tra nhanh trên tập huấn luyện
    print("\nKiểm tra nhanh trên tập huấn luyện...")
    train_predictions = iso_forest.predict(X_train_normal)
    # predict() trả về: 1 = Normal, -1 = Anomaly
    n_anomaly_detected = np.sum(train_predictions == -1)
    print(f"-> Số mẫu bị đánh dấu là Bất Thường trong tập Normal: "
          f"{n_anomaly_detected}/{X_train_normal.shape[0]} "
          f"({100*n_anomaly_detected/X_train_normal.shape[0]:.2f}%)")
 
    # 6. Đóng gói và lưu trữ
    print("\nĐang lưu trữ mô hình...")
    anomaly_model_package = {
        'model': iso_forest,
        'threshold': threshold,
        'model_type': 'IsolationForest'  # Ghi chú để Thành viên 5 nhận biết
    }
 
    save_path = os.path.join(current_dir, 'unsupervised_isoforest_model.pkl')
    joblib.dump(anomaly_model_package, save_path)
    print(f"-> Đã lưu thành công file: {save_path}")
    print("\n=== HƯỚNG DẪN TÍCH HỢP VÀO HYBRID IDS ===")
    print("Thay thế đoạn dự đoán K-Means cũ bằng đoạn sau:")
    print("""
    pkg = joblib.load('unsupervised_isoforest_model.pkl')
    iso_model  = pkg['model']
    threshold  = pkg['threshold']
 
    # score < threshold  ->  Tấn công (1)
    # score >= threshold ->  Bình thường (0)
    scores = iso_model.decision_function(X_test)
    anomaly_pred = np.where(scores < threshold, 1, 0)
    """)
 

if __name__ == "__main__":
    train_anomaly_detector()
 


 # threshold = 0.0185