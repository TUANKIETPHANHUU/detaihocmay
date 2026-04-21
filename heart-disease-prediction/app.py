import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score
import seaborn as sns

# ===== 1. CẤU HÌNH TRANG =====
st.set_page_config(page_title="Hệ Thống Dự Đoán Bệnh Tim", page_icon="❤️", layout="wide")

# ===== 2. XỬ LÝ ĐƯỜNG DẪN THÔNG MINH =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_file_path(sub_folder, file_name):
    path1 = os.path.join(BASE_DIR, sub_folder, file_name)
    path2 = os.path.join(BASE_DIR, "..", sub_folder, file_name)
    if os.path.exists(path1):
        return path1
    return path2

# ===== 3. TẢI MÔ HÌNH & DỮ LIỆU (CACHE) =====
@st.cache_resource
def load_model_objects():
    try:
        m_path = get_file_path('models', 'stacking_model.pkl')
        s_path = get_file_path('models', 'scaler.pkl')
        f_path = get_file_path('models', 'selected_features.pkl')
        
        model = joblib.load(m_path)
        scaler = joblib.load(s_path)
        selected_features = joblib.load(f_path)
        return model, scaler, selected_features
    except Exception as e:
        st.error(f"❌ Không thể tải mô hình. Lỗi: {e}")
        return None, None, None

@st.cache_data
def load_dataset():
    data_path = get_file_path('data', 'Heart Prediction Quantum Dataset.csv')
    try:
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Không thể đọc file CSV: {e}")
        return pd.DataFrame()

model, scaler, selected_features = load_model_objects()
df = load_dataset()

# ===== 4. HÀM XỬ LÝ ĐẶC TRƯNG =====
def process_input(df_input):
    df_res = df_input.copy()
    if 'BloodPressure' in df_res.columns and 'Cholesterol' in df_res.columns:
        df_res['BP_Cholesterol'] = df_res['BloodPressure'] * df_res['Cholesterol']
    if 'Age' in df_res.columns and 'BloodPressure' in df_res.columns:
        df_res['Age_BP'] = df_res['Age'] * df_res['BloodPressure']
    return df_res

# ===== 5. GIAO DIỆN SIDEBAR =====
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
st.sidebar.title("MENU ĐIỀU HƯỚNG")
page = st.sidebar.radio("Chọn trang:", [
    "🏠 Trang 1: Giới thiệu & EDA", 
    "🚀 Trang 2: Triển khai mô hình", 
    "📈 Trang 3: Đánh giá & Hiệu năng"
])

# ==========================================
# TRANG 1: GIỚI THIỆU & EDA
# ==========================================
if page == "🏠 Trang 1: Giới thiệu & EDA":
    st.title("📊 Giới thiệu & Khám phá dữ liệu (EDA)")
    
    st.markdown("""
    ### 👨‍🎓 Thông tin đồ án
    - **Tên đề tài:** Ứng dụng Học máy dự đoán cấp độ rủi ro bệnh tim mạch.
    - **Sinh viên thực hiện:** Nguyễn Văn A
    - **MSSV:** 123456
    
    ### 💡 Giá trị thực tiễn
    Hệ thống giúp hỗ trợ các bác sĩ chẩn đoán nhanh mức độ nguy cơ mắc bệnh tim mạch dựa trên các chỉ số sinh học cơ bản (tuổi, huyết áp, cholesterol,...). Từ đó, đưa ra các quyết định can thiệp y tế kịp thời, giảm thiểu rủi ro tử vong và tối ưu hóa thời gian khám chữa bệnh cho bệnh nhân.
    """)
    
    st.divider()
    
    if not df.empty:
        st.subheader("📂 1. Dữ liệu thô (Raw Data)")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📊 2. Phân tích trực quan (EDA)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Biểu đồ phân phối nhãn bệnh (HeartDisease):**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(x='HeartDisease', data=df, palette='Set2', ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0) # Sửa lỗi warning hiển thị nhãn
            st.pyplot(fig1)
            
        with col2:
            st.write("**Ma trận tương quan giữa các đặc trưng (Correlation Matrix):**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax2)
            st.pyplot(fig2)
            
        st.markdown("""
        ### 📝 Nhận xét dữ liệu:
        - **Sự cân bằng:** Biểu đồ phân phối cho thấy tỷ lệ giữa nhóm "Có bệnh" và "Không bệnh" trong tập dữ liệu gốc.
        - **Tương quan:** Thông qua ma trận tương quan, ta thấy các chỉ số như `BloodPressure` (Huyết áp) và `Cholesterol` có mức độ tương quan nhất định với khả năng mắc bệnh tim.
        """)
    else:
        st.error("Chưa kết nối được với file dữ liệu CSV trong thư mục 'data'.")

# ==========================================
# TRANG 2: TRIỂN KHAI MÔ HÌNH
# ==========================================
elif page == "🚀 Trang 2: Triển khai mô hình":
    st.title("🚀 Triển khai mô hình dự đoán")
    
    if model is None:
        st.error("Lỗi hệ thống: Mô hình chưa được tải. Vui lòng kiểm tra file .pkl.")
    else:
        st.markdown("Vui lòng nhập các thông số sinh học của bệnh nhân để hệ thống tiến hành chẩn đoán:")
        
        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Tuổi", min_value=1, max_value=120, value=45)
                gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
                bp = st.number_input("Huyết áp tâm thu (BloodPressure)", min_value=50, max_value=250, value=120)
            with col2:
                chol = st.number_input("Chỉ số Cholesterol", min_value=100, max_value=600, value=200)
                hr = st.number_input("Nhịp tim (HeartRate)", min_value=40, max_value=220, value=75)
                qpf = st.number_input("Chỉ số Quantum Pattern (QPF)", min_value=0.0, max_value=50.0, value=8.5, step=0.1)
                
            submit_btn = st.form_submit_button("🔍 Tiến hành dự đoán")
            
        if submit_btn:
            # 1. Tiền xử lý input giống hệt lúc huấn luyện
            input_df = pd.DataFrame([{
                'Age': age, 'Gender': 1 if gender=="Nam" else 0,
                'BloodPressure': bp, 'Cholesterol': chol, 
                'HeartRate': hr, 'QuantumPatternFeature': qpf
            }])
            
            processed_input = process_input(input_df)
            X_input = processed_input[selected_features]
            X_scaled = scaler.transform(X_input)
            
            # 2. Dự đoán
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            # 3. Hiển thị kết quả rõ ràng
            st.divider()
            st.subheader("🎯 Kết quả chẩn đoán:")
            
            labels_map = {
                0: ("An toàn (Không có dấu hiệu bệnh)", "🟢"), 
                1: ("Nguy cơ Nhẹ (Cần theo dõi)", "🔵"), 
                2: ("Nguy cơ Trung bình (Cần khám chuyên sâu)", "🟠"), 
                3: ("Nguy cơ Nặng (Cần can thiệp y tế ngay)", "🔴")
            }
            
            # Xử lý trường hợp mô hình chỉ có 2 nhãn (0 và 1) hoặc nhiều nhãn
            if prediction in labels_map:
                result_text, icon = labels_map[prediction]
            else:
                result_text, icon = (f"Phân lớp {prediction}", "🔔")
                
            st.markdown(f"### {icon} Cấp độ dự báo: **{result_text}**")
            
            # Tạo DataFrame cho biểu đồ xác suất dựa trên số lượng class thực tế của mô hình
            num_classes = len(probabilities)
            class_names = [labels_map[i][0] if i in labels_map else f"Class {i}" for i in range(num_classes)]
            
            st.write("**Độ tin cậy (Xác suất cho từng cấp độ):**")
            prob_df = pd.DataFrame({
                "Trạng thái": class_names,
                "Xác suất (%)": np.round(probabilities * 100, 2)
            })
            
            chart = alt.Chart(prob_df).mark_bar().encode(
                x='Xác suất (%):Q',
                y=alt.Y('Trạng thái:N', sort=None),
                color='Trạng thái:N',
                tooltip=['Trạng thái', 'Xác suất (%)']
            ).properties(height=250)
            
            st.altair_chart(chart, use_container_width=True)

# ==========================================
# TRANG 3: ĐÁNH GIÁ & HIỆU NĂNG
# ==========================================
elif page == "📈 Trang 3: Đánh giá & Hiệu năng":
    st.title("📈 Đánh giá & Hiệu năng (Evaluation)")
    
    if model is None or df.empty or 'HeartDisease' not in df.columns:
        st.warning("⚠️ Không thể đánh giá. Vui lòng đảm bảo mô hình đã được tải và file dữ liệu có cột 'HeartDisease'.")
    else:
        st.markdown("""
        Dưới đây là các chỉ số hiệu năng được **tính toán trực tiếp** bằng cách cho mô hình dự đoán lại trên toàn bộ tập dữ liệu đã thu thập.
        """)
        
        with st.spinner("Đang tính toán các chỉ số thực tế..."):
            # Lấy X và y từ tập dữ liệu
            X_eval = df.drop(columns=['HeartDisease'])
            y_eval = df['HeartDisease']
            
            # Tiền xử lý dữ liệu giống như lúc train
            X_eval_processed = process_input(X_eval)
            
            # Lọc các cột đã được features selection
            # Xử lý lỗi nếu trong file CSV tên cột không khớp hoàn toàn
            missing_cols = [col for col in selected_features if col not in X_eval_processed.columns]
            if missing_cols:
                st.error(f"Lỗi: Thiếu các cột sau trong tập dữ liệu để đánh giá: {missing_cols}")
            else:
                X_eval_final = X_eval_processed[selected_features]
                X_eval_scaled = scaler.transform(X_eval_final)
                
                # Dự đoán
                y_pred = model.predict(X_eval_scaled)
                
                # Tính toán các metric thật
                acc = accuracy_score(y_eval, y_pred)
                # Dùng average='weighted' để hỗ trợ cả bài toán nhị phân (Binary) và đa lớp (Multiclass)
                prec = precision_score(y_eval, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_eval, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_eval, y_pred, average='weighted', zero_division=0)
                
                # Hiển thị Metric
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy (Độ chính xác)", f"{acc*100:.2f}%")
                col2.metric("F1-Score (Trọng số)", f"{f1:.4f}")
                col3.metric("Precision", f"{prec:.4f}")
                col4.metric("Recall", f"{rec:.4f}")
                
                st.divider()
                
                col_cm, col_text = st.columns([1.2, 1])
                with col_cm:
                    st.subheader("📊 Ma trận nhầm lẫn (Confusion Matrix) Thực tế")
                    # Vẽ Confusion Matrix thật
                    real_cm = confusion_matrix(y_eval, y_pred)
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                    sns.heatmap(real_cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_ylabel('Nhãn Thực Tế (True Label)')
                    ax_cm.set_xlabel('Nhãn Dự Đoán (Predicted Label)')
                    st.pyplot(fig_cm)
                    
                with col_text:
                    st.subheader("🔍 Phân tích sơ bộ")
                    st.markdown(f"""
                    **Dựa trên Ma trận nhầm lẫn thực tế của tập dữ liệu hiện tại:**
                    
                    - **Tổng số mẫu đã đánh giá:** {len(y_eval)} mẫu.
                    - Mô hình đạt độ chính xác tổng thể là **{acc*100:.2f}%**, cho thấy sự ổn định của phương pháp học máy kết hợp (Stacking).
                    - Các ô nằm trên **đường chéo chính** (từ góc trên trái xuống góc dưới phải) thể hiện số lượng ca dự đoán ĐÚNG.
                    - Các ô nằm **ngoài đường chéo chính** là những trường hợp dự đoán SAI (Dương tính giả hoặc Âm tính giả).
                    
                    *(Lưu ý: Kết quả trên được tính toán trên toàn bộ tập dữ liệu hiện có trong ứng dụng. Trong thực tế triển khai, mô hình nên được đánh giá trên một tập kiểm thử - Test set hoàn toàn độc lập để đánh giá độ tổng quát).*
                    """)