# 🫀Dự đoán mức độ bệnh tim từ dữ liệu lâm sàng bằng mô hình học máy kết hợp stacking nhằm hỗ trợ phân loại nguy cơ và chẩn đoán sớm

## 📌 Giới thiệu dự án

Dự án này nhằm xây dựng mô hình học máy để **dự đoán mức độ nguy cơ bệnh tim** dựa trên các dữ liệu lâm sàng của bệnh nhân.  
Hệ thống hỗ trợ **phát hiện sớm và phân loại nguy cơ**, giúp bác sĩ đưa ra quyết định điều trị kịp thời và chính xác hơn.

---

## 📊 Nguồn dữ liệu (Data Source)

- Dataset: Heart Disease Dataset
- Nguồn: Kaggle / UCI Machine Learning Repository
- Link: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

### 📁 Quy mô dữ liệu:

- Hơn 1.000 mẫu dữ liệu
- Các thuộc tính chính:
  - Tuổi (Age)
  - Giới tính (Sex)
  - Huyết áp (Blood Pressure)
  - Cholesterol
  - Nhịp tim (Heart Rate)
  - Và các chỉ số y tế khác

---

## ⚙️ Phương pháp (Methodology)

### 🧹 Tiền xử lý dữ liệu

- Xử lý giá trị thiếu (Missing Values)
- Loại bỏ nhiễu và outliers
- Chuẩn hóa dữ liệu: `StandardScaler` / `MinMaxScaler`
- Mã hóa dữ liệu phân loại: `Label Encoding` / `One-hot Encoding`
- Giảm chiều dữ liệu: **LDA (Linear Discriminant Analysis)**

---

### 🤖 Mô hình sử dụng

#### 1. Mô hình cơ bản

- **SVM (Support Vector Machine) + LDA**
  - Hiệu quả với dữ liệu nhỏ và trung bình
  - Tăng khả năng phân tách lớp sau khi giảm chiều

#### 2. Mô hình nâng cao

- **Stacking Ensemble**
  - Base models:
    - LightGBM
    - CatBoost
  - Meta model: Logistic Regression (hoặc SVM)

📌 Lý do chọn:

- Tăng độ chính xác dự đoán
- Giảm overfitting
- Kết hợp ưu điểm nhiều mô hình mạnh

---

## 📈 Đánh giá mô hình

Các chỉ số đánh giá:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 💡 Giá trị thực tiễn

- Hỗ trợ bác sĩ phát hiện sớm bệnh tim
- Phân loại bệnh nhân theo mức độ nguy cơ
- Hỗ trợ quyết định điều trị chính xác hơn
- Giảm tỷ lệ biến chứng và tái nhập viện
- Ứng dụng trong hệ thống hỗ trợ quyết định y khoa (Clinical Decision Support System)

---

## 🛠️ Công nghệ sử dụng

- Python 🐍
- Scikit-learn
- Pandas, NumPy
- LightGBM
- CatBoost
- Matplotlib / Seaborn

---

## 🚀 Hướng phát triển

- Tích hợp giao diện web (Streamlit / Flask)
- Thêm dữ liệu lớn từ bệnh viện thực tế
- Áp dụng Deep Learning (Neural Networks)
- Triển khai API cho hệ thống y tế

---

## 👨‍💻 Tác giả

- Sinh viên: Phan Hữu Tuấn Kiệt
- Môn học:Học Máy Python

---

## 📌 Lưu ý

Dự án chỉ mang tính chất nghiên cứu và học tập, không thay thế chẩn đoán y khoa chuyên nghiệp.
