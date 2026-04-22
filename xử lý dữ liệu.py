import pandas as pd # trong đây có getdummies() để biến các cột chữ thành các cột nhị phân
import numpy as np # dùng để lưu numpy array
from sklearn.model_selection import train_test_split # hàm dùng để tách dữ 
from sklearn.preprocessing import StandardScaler # chuẩn hóa dữ liệu dựa trên phân bố thống kê: biến đổi mean = 0 và variance = 1

# Tải lên dữ liệu
data_from = pd.read_csv('ten_file')

# Biến các cột chữ thành các cột nhị phân
categorial_cols = ['cot_1','cot_2','cot_3',...]
data_from = pd.get_dummies(data_from, columns = categorial_cols)

# Tách features(X) ra khỏi target label(Y)
X = data_from.drop('cột chứa đáp án', axis = 1)
Y = data_from['cột chứa đáp án']

# Chia gói X và Y thành các gói train và test(train 70% test 30%) dùng kỹ thuật Hold-on
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

# Khởi tạo bộ chuẩn hóa (StandardScaler)
scaler = StandardScaler()

# Fit trên tập train: học các thông số từ tập train
scaler.fit(X_train)

#Transform trên cả 2 tập: áp dụng công thức đã học được lên cả 2 tập
X_train_sclaed = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Đầu ra là ma trận số thực NumPy
print(X_train_scaled[:5]) # cái này chỉ để kiểm tra dữ kiệu xem đúng ý chưa

# Chuyển ma trận numpy ngược lại thành bảng Pandas để lưu CSV
datafrom_final = pd.DataFrame(X_train_scaled)
datafrom_final.to_csv('X_train_preprocessed.csv', index=False)

# Hoặc lưu lại thành các file .npy
