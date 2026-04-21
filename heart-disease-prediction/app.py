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
# Lấy đường dẫn thư mục gốc của dự án (nơi chứa file app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== CACHE =====
@st.cache_resource
def load_model():
    # Đường dẫn tương đối: BASE_DIR -> thư mục 'models' -> tên file
    MODEL_DIR = os.path.join(BASE_DIR, 'models') 
    try:
        # ĐÃ SỬA: Loại bỏ hoàn toàn đường dẫn D:\...
        model = joblib.load(os.path.join(MODEL_DIR, 'stacking_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        selected_features = joblib.load(os.path.join(MODEL_DIR, 'selected_features.pkl'))
        return model, scaler, selected_features
    except Exception as e:
        st.error(f"❌ Lỗi tải mô hình: {e}")
        return None, None, None

@st.cache_data
def load_data():
    # ĐÃ SỬA: Đường dẫn tới file csv trong thư mục data
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'Heart Prediction Quantum Dataset.csv')
    try:
        if os.path.exists(DATA_PATH):
            return pd.read_csv(DATA_PATH)
        else:
            st.warning(f"⚠️ Không tìm thấy file dữ liệu tại: {DATA_PATH}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Lỗi đọc dữ liệu: {e}")
        return pd.DataFrame()

# Khởi tạo model và data
model, scaler, selected_features = load_model()
df = load_data()

# ===== FEATURE ENGINEERING =====
def feature_engineering(df_input):
    df_res = df_input.copy()
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
            # Đảm bảo dùng đúng tên cột target trong file csv của bạn (HeartDisease hoặc DiseaseLevel)
            target_col = 'HeartDisease' if 'HeartDisease' in df.columns else df.columns[-1]
            df[target_col].value_counts().plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99','#ffcc99'], ax=ax)
            st.pyplot(fig)

        st.subheader("📊 Ma trận tương quan")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            im = ax2.imshow(numeric_df.corr(), cmap='coolwarm')
            plt.colorbar(im)
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
    
    if model is None or scaler is None:
        st.error("Cảnh báo: File model (.pkl) chưa được tải đúng. Hãy kiểm tra thư mục 'models'.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Tuổi", 1, 120, 45)
                gender = st.selectbox("Giới tính", ["Nữ", "Nam"])
                bp = st.number_input("Huyết áp (BloodPressure)", 50, 200, 120)
            
            with col2:
                chol = st.number_input("Cholesterol", 100, 500, 200)
                hr = st.number_input("Nhịp tim (HeartRate)", 40, 200, 75)
                qpf = st.slider("Quantum Feature", 0.0, 1.0, 0.5)
            
            submit = st.form_submit_button("🔍 Dự đoán ngay")

        if submit:
            gender_val = 1 if gender == "Nam" else 0
            input_df = pd.DataFrame([{
                'Age': age, 'Gender': gender_val, 'BloodPressure': bp,
                'Cholesterol': chol, 'HeartRate': hr, 'QuantumPatternFeature': qpf
            }])

            input_df = feature_engineering(input_df)
            
            try:
                # Chỉ lấy các feature mà model yêu cầu
                X = input_df[selected_features]
                X_scaled = scaler.transform(X)

                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]

                labels = {0: 'Bình thường', 1: 'Nhẹ', 2: 'Trung bình', 3: 'Nặng'}
                
                st.markdown("---")
                st.subheader("Kết quả phân tích")
                if pred == 0: st.success(f"Kết luận: {labels[pred]}")
                elif pred == 1: st.info(f"Kết luận: {labels[pred]}")
                elif pred == 2: st.warning(f"Kết luận: {labels[pred]}")
                else: st.error(f"Kết luận: {labels[pred]}")

                # Biểu đồ xác suất
                proba_df = pd.DataFrame({"Mức độ": list(labels.values()), "Xác suất": proba})
                chart = alt.Chart(proba_df).mark_bar().encode(
                    x=alt.X("Mức độ", sort=None),
                    y="Xác suất",
                    color="Mức độ"
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi khi xử lý dữ liệu: {e}")

# =========================
# PAGE 3: EVALUATION (Giữ nguyên logic của bạn)
# =========================
elif page == "📈 Đánh giá":
    st.title("📈 Đánh giá hiệu năng mô hình")
    st.info("Chỉ số đo lường mô phỏng")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "92%")
    col2.metric("F1-Score", "0.90")
    col3.metric("Precision", "0.91")

# =========================
# PAGE 4: ADMIN
# =========================
elif page == "🛠️ Admin":
    st.title("🛠️ Quản trị hệ thống")
    password = st.text_input("🔐 Xác thực quyền Admin", type="password")
    if password == "admin123":
        st.success("Quyền truy cập được chấp nhận")
        if not df.empty:
            st.write(f"Tổng số bản ghi: {len(df)}")
            st.dataframe(df.head(20))