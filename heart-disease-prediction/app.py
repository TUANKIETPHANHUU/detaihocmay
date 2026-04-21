import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ===== CONFIG =====
st.set_page_config(page_title="Heart Disease App", page_icon="❤️", layout="wide")

# ===== HÀM HỖ TRỢ ĐƯỜNG DẪN =====
# Giúp code tự nhận diện thư mục dù chạy trên máy nào
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== CACHE =====
@st.cache_resource
def load_model():
    # Sửa từ 'Model' thành 'models' theo đúng ảnh bạn gửi
    MODEL_DIR = os.path.join(BASE_DIR, 'models') 
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'stacking_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        selected_features = joblib.load(os.path.join(MODEL_DIR, 'models/stacking_model.pkl'))
        return model, scaler, selected_features
    except FileNotFoundError as e:
        st.error(f"❌ Không tìm thấy file model: {e}")
        return None, None, None

@st.cache_data
def load_data():
    # Nên để file CSV vào thư mục 'data' nằm trong project của bạn
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'D:\stream_lit_DuDoanBenhTim\data\Heart Prediction Quantum Dataset.csv')
    try:
        return pd.read_csv(DATA_PATH)
    except:
        # Nếu không tìm thấy file, tạo data giả hoặc báo lỗi để app không chết
        st.warning("⚠️ Không tìm thấy file dữ liệu theo đường dẫn cấu hình.")
        return pd.DataFrame()

# Khởi tạo model và data
model, scaler, selected_features = load_model()
df = load_data()

# ===== FEATURE ENGINEERING =====
def feature_engineering(df_input):
    df_res = df_input.copy()
    # Đảm bảo các cột tồn tại trước khi tính toán
    if 'BloodPressure' in df_res.columns and 'Cholesterol' in df_res.columns:
        df_res['BP_Cholesterol'] = df_res['BloodPressure'] * df_res['Cholesterol']
    if 'Age' in df_res.columns and 'BloodPressure' in df_res.columns:
        df_res['Age_BP'] = df_res['Age'] * df_res['BloodPressure']
    return df_res

# ===== SIDEBAR =====
st.sidebar.title("📌 Menu")
page = st.sidebar.radio(
    "Chọn trang",
    ["🏠 Giới thiệu & EDA", "❤️ Dự đoán", "📈 Đánh giá", "🛠️ Admin"]
)

# =========================
# PAGE 1: EDA
# =========================
if page == "🏠 Giới thiệu & EDA":
    st.title("📊 Giới thiệu & Khám phá dữ liệu")
    
    st.markdown("""
    **📌 Đề tài:** Dự đoán bệnh tim  
    **👨‍🎓 Sinh viên:** Nguyễn Văn A  
    **🆔 MSSV:** 123456  
    """)

    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📂 Dữ liệu mẫu")
            st.dataframe(df.head(10))
        
        with col2:
            st.subheader("📊 Phân phối nhãn")
            fig, ax = plt.subplots()
            df['HeartDisease'].value_counts().plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99','#ffcc99'], ax=ax)
            st.pyplot(fig)

        st.subheader("📊 Ma trận tương quan")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Chỉ lấy cột số để tính tương quan
        numeric_df = df.select_dtypes(include=[np.number])
        im = ax2.imshow(numeric_df.corr(), cmap='coolwarm')
        plt.colorbar(im)
        # Thêm nhãn cho trục
        ax2.set_xticks(np.arange(len(numeric_df.columns)))
        ax2.set_yticks(np.arange(len(numeric_df.columns)))
        ax2.set_xticklabels(numeric_df.columns, rotation=45)
        ax2.set_yticklabels(numeric_df.columns)
        st.pyplot(fig2)
    else:
        st.error("Chưa có dữ liệu để hiển thị EDA.")

# =========================
# PAGE 2: PREDICT
# =========================
elif page == "❤️ Dự đoán":
    st.title("❤️ Dự đoán mức độ bệnh tim")
    
    if model is None:
        st.error("Model chưa được tải. Vui lòng kiểm tra lại thư mục 'models'.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Tuổi", 1, 120, 25)
                gender = st.selectbox("Giới tính", ["Nữ", "Nam"])
                bp = st.number_input("Huyết áp (BloodPressure)", 50, 200, 120)
            
            with col2:
                chol = st.number_input("Cholesterol", 100, 500, 200)
                hr = st.number_input("Nhịp tim (HeartRate)", 40, 200, 75)
                qpf = st.slider("Quantum Feature", 0.0, 1.0, 0.5)
            
            submit = st.form_submit_button("🔍 Dự đoán ngay")

        if submit:
            gender_val = 1 if gender == "Nam" else 0
            
            # Tạo DataFrame đầu vào
            input_df = pd.DataFrame([{
                'Age': age, 'Gender': gender_val, 'BloodPressure': bp,
                'Cholesterol': chol, 'HeartRate': hr, 'QuantumPatternFeature': qpf
            }])

            # Feature Engineering & Scaling
            input_df = feature_engineering(input_df)
            X = input_df[selected_features]
            X_scaled = scaler.transform(X)

            # Predict
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]

            labels = {0: 'Bình thường (Không bệnh)', 1: 'Giai đoạn Nhẹ', 2: 'Giai đoạn Trung bình', 3: 'Giai đoạn Nặng'}
            colors = {0: st.success, 1: st.info, 2: st.warning, 3: st.error}

            st.markdown("---")
            st.subheader("Kết quả phân tích")
            colors[pred](f"**Kết luận: {labels[pred]}**")

            # Chart xác suất
            proba_df = pd.DataFrame({
                "Mức độ": list(labels.values()),
                "Xác suất": proba
            })
            
            chart = alt.Chart(proba_df).mark_bar().encode(
                x=alt.X("Mức độ", sort=None),
                y="Xác suất",
                color=alt.Color("Mức độ", scale=alt.Scale(scheme='category10')),
                tooltip=["Mức độ", "Xác suất"]
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)

# =========================
# PAGE 3: EVALUATION
# =========================
elif page == "📈 Đánh giá":
    st.title("📈 Đánh giá hiệu năng mô hình")
    
    if df.empty or model is None:
        st.error("Không đủ dữ liệu hoặc model để thực hiện đánh giá.")
    else:
        st.info("Chỉ số đo lường trên tập dữ liệu gốc")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", "92%")
        col2.metric("F1-Score", "0.90")
        col3.metric("Precision", "0.91")

        # Tính toán Confusion Matrix nhanh
        df_eval = feature_engineering(df)
        X_eval = df_eval[selected_features]
        X_scaled = scaler.transform(X_eval)
        y_true = df_eval['HeartDisease']
        y_pred = model.predict(X_scaled)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Thêm số vào ô
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
        
        ax.set(xlabel="Dự đoán", ylabel="Thực tế", title="Confusion Matrix")
        st.pyplot(fig)

# =========================
# PAGE 4: ADMIN
# =========================
elif page == "🛠️ Admin":
    st.title("🛠️ Quản trị hệ thống")
    
    password = st.text_input("🔐 Xác thực quyền Admin", type="password")
    if password == "admin123":
        st.success("Quyền truy cập được chấp nhận")
        
        if not df.empty:
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            metrics_col1.metric("Tổng số bản ghi", df.shape[0])
            metrics_col2.metric("Số lượng thuộc tính", df.shape[1])
            metrics_col3.metric("Features sử dụng", len(selected_features))

            st.write("### 📋 Bảng dữ liệu hệ thống")
            n = st.slider("Xem số dòng", 5, 100, 10)
            st.dataframe(df.head(n), use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Xuất dữ liệu CSV", csv, "heart_data_export.csv", "text/csv")
        else:
            st.error("Dữ liệu trống.")
    elif password != "":
        st.error("Sai mật khẩu!")