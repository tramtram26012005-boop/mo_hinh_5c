import streamlit as tf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import io

# 1. Cấu hình trang Streamlit đầu tiên
tf.set_page_config(
    layout="wide",
    page_title="Hệ thống Dự báo Rủi ro Khách hàng (PD)",
    page_icon="🔮"
)

# 2. Hàm nạp dữ liệu dùng chung có cache
@tf.cache_data
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        tf.error(f"Lỗi khi đọc file: {e}")
        return None

# Định nghĩa danh sách biến X từ notebook
FEATURES = [
    'TC1', 'TC2', 'TC3', 'TC4', 'TC5', 'NL1', 'NL2', 'NL3', 'NL4', 
    'DK1', 'DK2', 'DK3', 'DK4', 'DK5', 'V1', 'V2', 'V3', 'V4', 'V5', 
    'V6', 'TS1', 'TS2', 'TS3', 'TS4'
]
TARGET = 'PD'

# 3. SIDEBAR — VÙNG CẤU HÌNH
with tf.sidebar:
    tf.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu mẫu
    uploaded_file = tf.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Hãy tải file dữ liệu mẫu 5c.csv hoặc file có cấu trúc tương đương để huấn luyện mô hình."
    )
    
    tf.divider()
    tf.subheader("Tham số mô hình AI")
    tf.caption("Thuật toán: Logistic Regression")
    
    # Các siêu tham số cấu hình động
    c_value = tf.slider("Hệ số nghịch đảo điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1, help="Giá trị nhỏ hơn chỉ định mức độ điều hòa mạnh hơn.")
    max_iter = tf.slider("Số vòng lặp tối đa (max_iter)", min_value=50, max_value=1000, value=100, step=50, help="Số lượng vòng lặp tối đa cho các thuật toán tối ưu hóa hội tụ.")
    random_state = tf.number_input("Cấu hình ngẫu nhiên (random_state)", min_value=0, max_value=9999, value=42, step=1, help="Mầm số ngẫu nhiên để đảm bảo tính tái lập kết quả.")
    test_size = tf.slider("Tỷ lệ tập kiểm tra (test_size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05, help="Tỷ lệ chia tập dữ liệu cho việc kiểm định mô hình.")
    
    tf.divider()
    # Nút bấm kích hoạt huấn luyện duy nhất
    train_clicked = tf.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True, help="Bấm để bắt đầu trích xuất biến và huấn luyện thuật toán.")

# 4. HEADER — VÙNG ĐỊNH HƯỚNG
tf.title("🔮 Hệ thống Dự báo Rủi ro Khách hàng")
tf.caption("Ứng dụng quản trị rủi ro tự động hóa, chuyển đổi từ mô hình Logistic Regression tối ưu trên Notebook.")

if uploaded_file is None:
    tf.info("👋 Chào mừng bạn! Vui lòng tải lên file dữ liệu (.csv hoặc .xlsx) tại thanh Sidebar bên trái để bắt đầu sử dụng ứng dụng.")
    tf.stop()

# Đọc file dữ liệu
file_bytes = uploaded_file.read()
df = load_data(file_bytes, uploaded_file.name)

if df is None:
    tf.error("Không thể xử lý định dạng file dữ liệu. Vui lòng kiểm tra lại.")
    tf.stop()

tf.caption(f"📁 Đang sử dụng tệp dữ liệu: **{uploaded_file.name}**")
tf.divider()

# 5. KHỐI HUẤN LUYỆN (Chạy khi nhấn nút và lưu vào session_state)
if train_clicked:
    # Kiểm tra sự tồn tại của các biến yêu cầu
    missing_features = [col for col in FEATURES if col not in df.columns]
    if TARGET not in df.columns:
        tf.error(f"Mẫu dữ liệu thiếu cột mục tiêu: **{TARGET}**")
    elif missing_features:
        tf.error(f"Dữ liệu thiếu các cột thuộc tính sau: {missing_features}")
    else:
        with tf.spinner("Đang huấn luyện mô hình..."):
            X = df[FEATURES]
            y = df[TARGET]
            
            # Chia tập dữ liệu giống notebook
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Khởi tạo và khớp mô hình
            model = LogisticRegression(C=c_value, max_iter=max_iter, random_state=random_state)
            model.fit(X_train, y_train)
            
            # Đánh giá kết quả
            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Lưu trữ toàn bộ 3 thành phần cốt lõi vào session_state
            tf.session_state['trained_model'] = model
            tf.session_state['features_list'] = FEATURES
            tf.session_state['eval_results'] = {
                'y_test': y_test,
                'y_pred': y_pred,
                'y_probs': y_probs,
                'X_test': X_test
            }
            tf.success("🎉 Huấn luyện thành công mô hình! Chuyển qua các tab bên dưới để xem kết quả chi tiết.")

# 6. KHỐI TABS NỘI DUNG CHÍNH
tab1, tab2, tab3, tab4 = tf.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa biến", 
    "🎯 Kết quả & Kiểm định", 
    "🔮 Sử dụng mô hình"
])

# TAB 1: TỔNG QUAN DỮ LIỆU
with tab1:
    tf.subheader("Đặc điểm & Cấu trúc Tập Dữ liệu")
    
    col_m1, col_m2, col_m3 = tf.columns(3)
    col_m1.metric("Số lượng dòng", f"{df.shape[0]:,}")
    col_m2.metric("Số lượng cột", f"{df.shape[1]}")
    file_size_mb = len(file_bytes) / (1024 * 1024)
    col_m3.metric("Dung lượng file", f"{file_size_mb:.2f} MB")
    
    tf.markdown("##### 🔍 5 dòng dữ liệu thô đầu tiên")
    tf.dataframe(df.head(5), use_container_width=True)
    
    tf.markdown("##### 📐 Thống kê mô tả các biến mô hình (X & y)")
    # Chỉ mô tả tập các biến đưa vào mô hình theo quy tắc
    available_cols = [col for col in FEATURES + [TARGET] if col in df.columns]
    tf.dataframe(df[available_cols].describe(), use_container_width=True)

# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
with tab2:
    tf.subheader("Biểu đồ phân phối thành phần biến")
    
    # Xác định danh sách biến vẽ đồ thị (Ưu tiên biến Target trước)
    plot_cols = []
    if TARGET in df.columns:
        plot_cols.append(TARGET)
    plot_cols.extend([col for col in FEATURES if col in df.columns])
    
    # Nếu có quá nhiều biến (>4), cho người dùng tùy chọn chọn nhanh hiển thị
    if len(plot_cols) > 4:
        selected_plot_cols = tf.multiselect(
            "Lựa chọn các biến cần trực quan hóa (Mặc định hiển thị 4 biến đầu tiên):",
            options=plot_cols,
            default=plot_cols[:4]
        )
    else:
        selected_plot_cols = plot_cols

    if selected_plot_cols:
        # Bố trí dạng lưới lưới 2x2 cân đối dùng st.columns
        for i in range(0, len(selected_plot_cols), 2):
            cols = tf.columns(2)
            for j in range(2):
                if i + j < len(selected_plot_cols):
                    col_name = selected_plot_cols[i + j]
                    with cols[j]:
                        # Tự chọn loại biểu đồ theo đặc điểm phân phối lớp/số rời rạc
                        if df[col_name].nunique() <= 10 or col_name == TARGET:
                            counts = df[col_name].value_counts().reset_index()
                            counts.columns = [col_name, 'Số lượng']
                            fig = px.bar(counts, x=col_name, y='Số lượng', 
                                         title=f"Phân phối tần suất biến {col_name}",
                                         color=col_name, height=350)
                        else:
                            fig = px.histogram(df, x=col_name, 
                                               title=f"Biểu đồ Histogram biến {col_name}",
                                               marginal="box", height=350)
                        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                        tf.plotly_chart(fig, use_container_width=True)
    else:
        tf.warning("Vui lòng lựa chọn ít nhất một cột để hiển thị trực quan.")

# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
with tab3:
    tf.subheader("Đánh giá độ chính xác mô hình")
    
    if 'eval_results' not in tf.session_state:
        tf.info("💡 Hãy thiết lập tham số và nhấn nút **🚀 Huấn luyện Mô hình** ở thanh Sidebar để xem đánh giá kiểm định.")
    else:
        results = tf.session_state['eval_results']
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_probs = results['y_probs']
        
        # Chỉ tiêu dạng số metric
        acc = accuracy_score(y_test, y_pred)
        col_r1, col_r2 = tf.columns(2)
        col_r1.metric("Độ chính xác tổng thể (Accuracy)", f"{acc*100:.2f}%")
        
        # Classification report chi tiết
        tf.markdown("##### 📄 Bảng báo cáo chi tiết chỉ số định lượng (Classification Report)")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        tf.dataframe(report_df, use_container_width=True)
        
        # Biểu đồ Ma trận nhầm lẫn và ROC Curve
        col_g1, col_g2 = tf.columns(2)
        
        with col_g1:
            tf.markdown("##### 🧮 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True,
                               labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế"),
                               x=['Không Rủi Ro (0)', 'Có Rủi Ro (1)'],
                               y=['Không Rủi Ro (0)', 'Có Rủi Ro (1)'],
                               color_continuous_scale="Blues", height=380)
            tf.plotly_chart(fig_cm, use_container_width=True)
            
        with col_g2:
            tf.markdown("##### 📉 Biểu đồ đường cong ROC")
            if y_probs is not None:
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.4f})',
                                  labels=dict(x='Tỷ lệ báo động giả (FPR)', y='Tỷ lệ nhận dạng đúng (TPR)'),
                                  height=380)
                fig_roc.add_shape(type="line", line=dict(dash='dash', color='red'), x0=0, x1=1, y0=0, y1=1)
                tf.plotly_chart(fig_roc, use_container_width=True)
            else:
                tf.warning("Mô hình không hỗ trợ xuất xác suất kiểm định ROC Curve.")

# TAB 4: SỬ DỤNG MÔ HÌNH (DỰ BÁO)
with tab4:
    tf.subheader("Thực hiện dự báo trực tuyến cho dữ liệu mới")
    
    if 'trained_model' not in tf.session_state:
        tf.info("💡 Bạn cần huấn luyện mô hình thành công trước khi truy cập tính năng dự báo rủi ro này.")
    else:
        model = tf.session_state['trained_model']
        
        mode = tf.radio(
            "Phương thức nhập dữ liệu dự báo:",
            ["Nhập trực tiếp qua form", "Tải tệp danh sách hàng loạt (Cấu trúc mẫu X_test)"],
            horizontal=True
        )
        
        if mode == "Nhập trực tiếp qua form":
            tf.markdown("##### 🖊️ Điền thông số chi tiết của khách hàng:")
            
            # Tạo form tự động hóa dựa theo tập thuộc tính đã huấn luyện
            with tf.form("prediction_form"):
                form_cols = tf.columns(4) # Chia form thành 4 cột nhập liệu gọn gàng
                input_data = {}
                
                for idx, col_name in enumerate(FEATURES):
                    col_idx = idx % 4
                    with form_cols[col_idx]:
                        # Lấy min/max/median thực tế từ df huấn luyện để gán biên mặc định
                        min_val = float(df[col_name].min()) if col_name in df.columns else 1.0
                        max_val = float(df[col_name].max()) if col_name in df.columns else 5.0
                        med_val = float(df[col_name].median()) if col_name in df.columns else 3.0
                        
                        input_data[col_name] = tf.number_input(
                            f"{col_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=med_val,
                            step=1.0,
                            help=f"Giá trị từ {min_val} đến {max_val}."
                        )
                        
                submit_btn = tf.form_submit_button("🔮 Dự báo ngay", type="primary")
                
            if submit_btn:
                # Chuyển đổi dữ liệu nhập form thành DataFrame chuẩn cấu trúc
                X_new = pd.DataFrame([input_data])
                
                prediction = model.predict(X_new)[0]
                probabilities = model.predict_proba(X_new)[0] if hasattr(model, "predict_proba") else None
                
                tf.divider()
                tf.markdown("### 📢 Kết quả phân tích rủi ro:")
                
                col_res1, col_res2 = tf.columns(2)
                if prediction == 1:
                    col_res1.error("Kết luận: **CÓ RỦI RO (1)**")
                else:
                    col_res1.success("Kết luận: **KHÔNG CÓ RỦI RO (0)**")
                    
                if probabilities is not None:
                    col_res2.metric("Xác suất khách hàng KHÔNG RỦI RO (0)", f"{probabilities[0]*100:.2f}%")
                    tf.metric("Xác suất khách hàng CÓ RỦI RO (1)", f"{probabilities[1]*100:.2f}%")
                    
        elif mode == "Tải tệp danh sách hàng loạt (Cấu trúc mẫu X_test)":
            tf.markdown("##### 📁 Tải file chứa danh sách các thuộc tính biến X")
            bulk_file = tf.file_uploader("Tải lên file dữ liệu kiểm tra mới (.csv, .xlsx)", type=["csv", "xlsx"], key="bulk_pred")
            
            if bulk_file is not None:
                bulk_bytes = bulk_file.read()
                df_bulk = load_data(bulk_bytes, bulk_file.name)
                
                if df_bulk is not None:
                    # Kiểm tra trùng khớp schema
                    missing_cols = [col for col in FEATURES if col not in df_bulk.columns]
                    if missing_cols:
                        tf.error(f"Lỗi cấu trúc tệp! Thiếu các cột thuộc tính yêu cầu sau: {missing_cols}")
                    else:
                        X_bulk = df_bulk[FEATURES]
                        bulk_preds = model.predict(X_bulk)
                        
                        df_result = df_bulk.copy()
                        df_result['Dự_Báo_PD'] = bulk_preds
                        
                        if hasattr(model, "predict_proba"):
                            bulk_probs = model.predict_proba(X_bulk)
                            df_result['Xác_Suất_An_Toàn(0)'] = bulk_probs[:, 0]
                            df_result['Xác_Suất_Rủi_Ro(1)'] = bulk_probs[:, 1]
                            
                        tf.success(f"Đã hoàn thành dự báo cho {df_result.shape[0]} bản ghi dữ liệu.")
                        
                        # Hiển thị kết quả cuộn gọn
                        tf.dataframe(df_result, use_container_width=True)
                        
                        # Nút xuất file tải về
                        csv_data = df_result.to_csv(index=False, encoding="utf-8-sig")
                        tf.download_button(
                            label="📥 Tải xuống kết quả dự báo (CSV)",
                            data=csv_data,
                            file_name="ket_qua_du_bao_hang_loat.csv",
                            mime="text/csv"
                        )
