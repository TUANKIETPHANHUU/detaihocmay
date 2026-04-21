import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# --- 1. Cấu hình đường dẫn ---
# Thay đổi đường dẫn này cho đúng với máy của bạn
DATA_PATH = r"D:\stream_lit_DuDoanBenhTim\data\Heart Prediction Quantum Dataset.csv"
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- 2. Load và Tiền xử lý dữ liệu ---
data = pd.read_csv(DATA_PATH)

def assign_disease_level(row):
    if row['HeartDisease'] == 0:
        return 0  # Không bệnh
    
    high_risk_age = 55 if row['Gender'] == 1 else 65
    risk_factors = 0
    
    if row['BloodPressure'] >= 160: risk_factors += 2
    elif row['BloodPressure'] >= 140: risk_factors += 1
    
    if row['Cholesterol'] >= 240: risk_factors += 1
    if row['Age'] >= high_risk_age: risk_factors += 1
    
    if risk_factors <= 1: return 1  # Nhẹ
    elif risk_factors == 2: return 2  # Trung bình
    else: return 3  # Nặng

data['DiseaseLevel'] = data.apply(assign_disease_level, axis=1)

# Feature engineering
data['BP_Cholesterol'] = data['BloodPressure'] * data['Cholesterol']
data['Age_BP'] = data['Age'] * data['BloodPressure']

# Tách Features và Target
X = data.drop(['HeartDisease', 'DiseaseLevel'], axis=1)
y = data['DiseaseLevel']

# --- 3. Lựa chọn đặc trưng (Feature Selection) ---
selector = SelectKBest(score_func=f_classif, k=7)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()
print(f"Các đặc trưng được chọn: {selected_features}")

# --- 4. Chuẩn hóa và Cân bằng dữ liệu ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Cân bằng dữ liệu với SMOTE (mỗi nhóm 600 mẫu)
target_samples = 600
smote = SMOTE(random_state=42, sampling_strategy={0: target_samples, 1: target_samples, 2: target_samples, 3: target_samples})
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# --- 5. Huấn luyện mô hình Stacking ---
base_models = [
    ('lgbm', LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbose=-1)),
    ('catboost', CatBoostClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbose=0))
]

meta_model = LogisticRegression(max_iter=1000, C=10.0)

stacking_clf = StackingClassifier(
    estimators=base_models, 
    final_estimator=meta_model, 
    cv=3, 
    passthrough=True
)

print("Đang huấn luyện mô hình Stacking... Vui lòng đợi.")
stacking_clf.fit(X_train, y_train)

# --- 6. Lưu kết quả vào thư mục 'models' ---
joblib.dump(stacking_clf, os.path.join(MODEL_DIR, 'stacking_model.pkl'))
joblib.dump(selector, os.path.join(MODEL_DIR, 'feature_selector.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(selected_features, os.path.join(MODEL_DIR, 'selected_features.pkl'))

print("-" * 30)
print(f"Thành công! 4 file đã được lưu vào thư mục '{MODEL_DIR}':")
print(f"1. {os.path.join(MODEL_DIR, 'stacking_model.pkl')}")
print(f"2. {os.path.join(MODEL_DIR, 'feature_selector.pkl')}")
print(f"3. {os.path.join(MODEL_DIR, 'scaler.pkl')}")
print(f"4. {os.path.join(MODEL_DIR, 'selected_features.pkl')}")