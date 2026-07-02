# 🔮 Ứng dụng Dự báo Rủi ro Khách hàng (Streamlit Web App)

Hệ thống ứng dụng được thiết kế nhằm mục đích chuyển hóa quy trình làm việc từ Notebook phân tích dữ liệu rủi ro khách hàng thành một nền tảng Web tương tác trực quan, giúp người quản trị dễ dàng đưa ra quyết định dựa trên thuật toán Học máy.

## 🗂️ Mô tả Mô hình và Dữ liệu mẫu
- **Thuật toán sử dụng**: Hồi quy Logistic (Logistic Regression) phân loại nhị phân.
- **Biến mục tiêu (`y`)**: `PD` (Xác định tình trạng rủi ro: `0` - Không rủi ro, `1` - Có rủi ro).
- **Tập biến đặc trưng đầu vào (`X`)**: Bao gồm 24 chỉ số đánh giá tổng hợp (`TC1` đến `TC5`, `NL1` đến `NL4`, `DK1` đến `DK5`, `V1` đến `V6`, `TS1` đến `TS4`).

---

## 🛠️ Hướng dẫn Cài đặt & Triển khai

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo bạn đã cài đặt phiên bản Python ổn định (Khuyến nghị bản Python 3.9 - 3.11).

### Bước 2: Cài đặt các thư viện bắt buộc
Mở terminal/cmd tại thư mục chứa mã nguồn ứng dụng và chạy dòng lệnh sau:
```bash
pip install -r requirements.txt
