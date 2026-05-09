import pandas as pd # trong đây có getdummies() để biến các cột chữ thành các cột nhị phân
import numpy as np # dùng để lưu numpy array
from sklearn.model_selection import train_test_split # hàm dùng để tách dữ 
from sklearn.preprocessing import StandardScaler # chuẩn hóa dữ liệu dựa trên phân bố thống kê: biến đổi mean = 0 và variance = 1

# Tải lên dữ liệu
df_unsup = pd.read_csv('Train_Unsupervised_Only_Normal.csv')
df_sup = pd.read_csv('Train_Supervised_Normal_Neptune.csv')
df_test = pd.read_csv('Test_ZeroDay_Scenario.csv')

# Đánh dấu nguồn trước khi gộp
df_unsup['source'] = 'unsup'
df_sup['source'] = 'sup'
df_test['source'] = 'test'

df_all = pd.concat([df_unsup, df_sup, df_test], ignore_index=True)

# Biến các cột chữ thành số (One-Hot Encoding)
# Sau bước này, tất cả dòng dữ liệu sẽ có số cột y hệt nhau
categorial_cols = ['protocol_type', 'service', 'flag']
df_all = pd.get_dummies(df_all, columns=categorial_cols)

# Tách dữ liệu trả về 3 tập như cũ
X_unsup = df_all[df_all['source'] == 'unsup'].drop(['source', 'label'], axis = 1)
Y_unsup = df_all[df_all['source'] == 'unsup']['label']

X_sup = df_all[df_all['source'] == 'sup'].drop(['source', 'label'], axis = 1)
Y_sup = df_all[df_all['source'] == 'sup']['label']

X_test = df_all[df_all['source'] == 'test'].drop(['source', 'label'], axis = 1)
Y_test = df_all[df_all['source'] == 'test']['label']

# Khởi tạo bộ chuẩn hóa (StandardScaler)
scaler = StandardScaler()
X_train_combined = pd.concat([X_unsup, X_sup])
scaler.fit(X_train_combined) # Học trung bình và độ lệch chuẩn chung

# Áp dụng chuẩn hóa đồng bộ cho các tập dữ liệu
X_unsup_scaled = scaler.transform(X_unsup)
X_sup_scaled = scaler.transform(X_sup)
X_test_scaled_for_unsup = scaler.transform(X_test) # Transform tập test để test mô hình Unsupervised
X_test_scaled_for_sup = scaler.transform(X_test) # Transform tập test để test mô hình Supervised

# Đầu ra là ma trận số thực NumPy
np.save('X_unsup_train.npy', X_unsup_scaled)
np.save('X_sup_train.npy', X_sup_scaled)
np.save('X_test_for_unsup.npy', X_test_scaled_for_unsup)
np.save('X_test_for_sup.npy', X_test_scaled_for_sup)

np.save('y_sup_train.npy', Y_sup.values)
np.save('y_test.npy', Y_test.values)