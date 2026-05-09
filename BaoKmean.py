import numpy as np
import joblib
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import os

def train_anomaly_detector():
    # 1. Định vị thư mục làm việc hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Hệ thống đang làm việc tại: {current_dir}")
    
    # 2. Tải dữ liệu huấn luyện
    print("Đang tải dữ liệu Normal từ khâu Tiền xử lý...")
    data_path = os.path.join(current_dir, 'X_unsup_train.npy')
    X_train_normal = np.load(data_path)
    
    # 3. Huấn luyện thuật toán K-Means
    print("Đang huấn luyện mô hình K-Means (Học không giám sát)...")
    # n_clusters=5: Gom các hành vi mạng bình thường thành 5 nhóm đặc trưng
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_train_normal)
    
    # 4. Tính toán Threshold (Ngưỡng dị thường)
    print("Đang tính toán Ngưỡng giới hạn (Threshold)...")
    # Tính khoảng cách từ mỗi điểm dữ liệu đến tâm cụm (centroid) gần nhất của nó
    distances = np.min(cdist(X_train_normal, kmeans.cluster_centers_, 'euclidean'), axis=1)
    
    # Lấy phân vị 99% để loại bỏ các điểm nhiễu (outliers) trong tập Normal
    threshold = np.percentile(distances, 95)
    print(f"-> Ngưỡng khoảng cách tối đa (Threshold) được thiết lập: {threshold:.4f}")
    
    # 5. Đóng gói và lưu trữ
    print("Đang lưu trữ mô hình cho Thành viên 5...")
    anomaly_model_package = {
        'model': kmeans,
        'threshold': threshold
    }
    
    save_path = os.path.join(current_dir, 'unsupervised_kmeans_model.pkl')
    joblib.dump(anomaly_model_package, save_path)
    print(f"-> Đã lưu thành công file: {save_path}")

if __name__ == "__main__":
    train_anomaly_detector()

    #threshold 27.1515
    