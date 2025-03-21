# HyperparameterDLmodels
## Giới thiệu
HyperparameterDLmodels là một website đánh giá ảnh hưởng của các siêu tham số đến hiệu suất của mô hình mạng nơ-ron nhân tạo (Deep Learning). Dự án nhằm giúp người dùng trực quan hóa và so sánh các mô hình dựa trên sự thay đổi của siêu tham số, dự đoán con số dựa trên dữ liệu đầu vào.
## Tính năng chính
- Nhập dữ liệu đầu vào
- Dự đoán con số
- Hiển thị kết quả dự đoán
- Trực quan hóa hiệu suất mô hình bằng biểu đồ
- Hỗ trợ lưu trữ kết quả để phân tích sau này
## Công nghệ sử dụng
- Backend: Python (Django)
- Frontend: HTML, CSS, JavaScript
- Cơ sở dữ liệu: SQLite3
- Thư viện hỗ trợ: Chart.js
## Yêu cầu hệ thống
- Python 3.x
- Django
- Thư viện máy học: Chart.js
## Hướng dẫn cài đặt
### Chuẩn bị môi trường:
Cài đặt Django và các thư viện cần thiết. Mở terminal/cmd và chạy:
```
pip install django matplotlib pillow
```
## Truy cập HyperparameterDLmodels
Sau khi cài đặt xong, bạn có thể khởi chạy Django:
```
python manage.py runserver
```
Khi hoàn tất các bước trên, bạn có thể mở trình duyệt và truy cập:
```
http://127.0.0.1:8000/
```

