
# 📦 PHÂN TÍCH VÀ DỰ BÁO TRỄ HẠN GIAO HÀNG

### Đồ án môn học: Phân tích và tích hợp dữ liệu lớn

**Nhóm thực hiện:** Nhóm 6  
**Trường:** Đại học Phenikaa  
**Bộ dữ liệu:** DataCo Smart Supply Chain  
**Công nghệ:** Python, PySpark, Spark SQL, Spark MLlib, Google Colab, Streamlit, Plotly

---

## 1. Giới thiệu dự án

Trong lĩnh vực thương mại điện tử và quản lý chuỗi cung ứng, việc giao hàng đúng hạn đóng vai trò quan trọng đối với trải nghiệm khách hàng và hiệu quả hoạt động của doanh nghiệp.

Dự án **Phân tích và dự báo trễ hạn giao hàng** sử dụng bộ dữ liệu DataCo Smart Supply Chain để nghiên cứu tình trạng giao hàng, xác định các yếu tố liên quan đến nguy cơ giao trễ và xây dựng mô hình học máy hỗ trợ dự báo.

Hệ thống kết hợp xử lý dữ liệu bằng PySpark, phân tích bằng Spark SQL, huấn luyện mô hình Random Forest và xây dựng Dashboard trực quan hóa kết quả.

## 2. Mục tiêu dự án

- Thu thập, kiểm tra và làm sạch dữ liệu đơn hàng.
- Thiết kế và tích hợp dữ liệu từ nhiều bảng.
- Phân tích tỷ lệ giao trễ theo khu vực và quốc gia.
- Phân tích ảnh hưởng của phương thức vận chuyển.
- Phân tích xu hướng giao trễ theo thời gian.
- Phân tích đặc điểm sản phẩm và đơn hàng.
- Xây dựng mô hình dự báo nguy cơ giao trễ.
- Trực quan hóa dữ liệu và hỗ trợ dự báo đơn hàng mới.

## 3. Bộ dữ liệu

Dự án sử dụng bộ dữ liệu **DataCo Smart Supply Chain**.

| Thông tin | Giá trị |
|---|---|
| Số dòng dữ liệu ban đầu | 180.519 |
| Số cột ban đầu | 53 |
| Số cột sau tiền xử lý | 51 |
| Số đơn hàng duy nhất | 65.752 |
| Khoảng thời gian | 2015–2018 |
| Số đơn giao trễ | 36.048 |
| Số đơn không trễ | 29.704 |
| Tỷ lệ giao trễ | 54,82% |

**Lưu ý:** Mỗi dòng dữ liệu gốc biểu diễn một chi tiết sản phẩm trong đơn hàng. Một đơn hàng có thể chứa nhiều sản phẩm, vì vậy số dòng dữ liệu lớn hơn số đơn hàng duy nhất.

## 4. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| Google Colab | Môi trường thực hiện và thử nghiệm |
| Google Drive | Lưu trữ dữ liệu và kết quả |
| PySpark | Tiền xử lý và tích hợp dữ liệu |
| Spark SQL | Truy vấn và phân tích dữ liệu |
| Spark MLlib | Huấn luyện mô hình học máy |
| Random Forest | Dự báo nguy cơ giao trễ |
| Parquet | Lưu trữ dữ liệu đã xử lý |
| Pandas | Xử lý dữ liệu phục vụ trực quan hóa |
| Streamlit | Xây dựng giao diện Dashboard |
| Plotly | Trực quan hóa biểu đồ tương tác |

## 5. Quy trình thực hiện

Dự án được triển khai qua các giai đoạn:

1. Đọc dữ liệu CSV bằng PySpark.
2. Kiểm tra dữ liệu thiếu, kiểu dữ liệu và giá trị bất thường.
3. Chuẩn hóa tên cột và xử lý dữ liệu ngày tháng.
4. Làm sạch dữ liệu phục vụ phân tích.
5. Tách dữ liệu thành các bảng theo mô hình dữ liệu.
6. Kiểm tra khóa liên kết và tích hợp dữ liệu.
7. Lưu dữ liệu đã xử lý dưới định dạng Parquet.
8. Phân tích dữ liệu bằng Spark SQL.
9. Xây dựng và so sánh các mô hình học máy.
10. Đánh giá mô hình và triển khai Dashboard.

## 6. Thiết kế và tích hợp dữ liệu

Dữ liệu được tổ chức thành 6 bảng chính:

| Bảng | Số bản ghi | Nội dung |
|---|---:|---|
| Customers | 20.652 | Thông tin khách hàng |
| Products | 118 | Thông tin sản phẩm |
| Orders | 65.752 | Thông tin đơn hàng |
| Order Items | 180.519 | Chi tiết sản phẩm trong đơn |
| Shipping | 65.752 | Thông tin vận chuyển |
| Regions | 167 | Thông tin khu vực và quốc gia |

Sau khi kiểm tra quan hệ giữa các bảng, nhóm tạo bảng dữ liệu tích hợp phục vụ phân tích và lưu dưới định dạng Parquet.

## 7. Phân tích dữ liệu bằng Spark SQL

### 7.1. Phân tích theo khu vực

Thống kê số đơn hàng, số đơn giao trễ và tỷ lệ giao trễ theo từng khu vực.

### 7.2. Phân tích theo phương thức vận chuyển

| Phương thức | Tỷ lệ giao trễ |
|---|---:|
| First Class | 95,27% |
| Second Class | 76,72% |
| Same Day | 46,15% |
| Standard Class | 38,13% |

Kết quả cho thấy tỷ lệ giao trễ có sự khác biệt đáng kể giữa các phương thức vận chuyển.

### 7.3. Phân tích theo thời gian

Phân tích xu hướng giao trễ theo năm, tháng và phương thức vận chuyển nhằm nhận diện sự thay đổi theo thời gian.

### 7.4. Phân tích sản phẩm và đơn hàng

Phân tích đặc điểm đơn hàng, số lượng sản phẩm và giá trị đơn hàng để tìm hiểu mối liên hệ với tình trạng giao trễ.

## 8. Xây dựng mô hình dự báo

### 8.1. Bài toán dự báo

Bài toán được xây dựng dưới dạng phân loại nhị phân:

- **0:** Không giao trễ.
- **1:** Có nguy cơ giao trễ.

### 8.2. Chia dữ liệu

Dữ liệu được chia theo thứ tự thời gian để mô phỏng việc sử dụng dữ liệu quá khứ nhằm dự báo đơn hàng trong tương lai.

| Tập dữ liệu | Số đơn hàng |
|---|---:|
| Train | 41.763 |
| Validation | 10.319 |
| Test | 13.670 |

### 8.3. Thử nghiệm mô hình

Nhóm tiến hành thử nghiệm và so sánh các thuật toán:

- Logistic Regression
- Random Forest
- Gradient-Boosted Trees (GBT)

Đồng thời, nhóm thử nghiệm Random Forest với các tập đặc trưng khác nhau để lựa chọn phương án phù hợp.

### 8.4. Mô hình cuối cùng

Nhóm lựa chọn **Random Forest với 5 đặc trưng đầu vào**:

1. `shipping_mode` – Phương thức vận chuyển.
2. `order_region` – Khu vực.
3. `order_country` – Quốc gia.
4. `scheduled_shipping_days` – Số ngày vận chuyển dự kiến.
5. `order_month` – Tháng đặt hàng.

Phương án 5 đặc trưng giúp đơn giản hóa đầu vào, thuận tiện cho việc dự báo đơn hàng mới và vẫn duy trì kết quả đánh giá tương đương hoặc tốt hơn phương án nhiều đặc trưng.

## 9. Kết quả đánh giá mô hình

Mô hình Random Forest cuối cùng được đánh giá trên tập Test.

| Chỉ số | Kết quả |
|---|---:|
| Accuracy | 69,49% |
| Weighted F1-score | 0,6930 |
| ROC-AUC | 0,7345 |
| Precision lớp giao trễ | 80,21% |
| Recall lớp giao trễ | 59,16% |

Kết quả cho thấy mô hình có khả năng nhận diện nguy cơ giao trễ ở mức thử nghiệm. Tuy nhiên, mô hình vẫn bỏ sót một số đơn hàng thực tế giao trễ và cần được cải thiện trước khi ứng dụng thực tế.

## 10. Dashboard trực quan hóa

Dashboard được xây dựng bằng Streamlit và Plotly, hỗ trợ người dùng theo dõi và phân tích tình trạng giao hàng.

### Các chức năng chính

- **Tổng quan:** Thống kê số lượng đơn hàng và tỷ lệ giao trễ.
- **Khu vực:** Phân tích tỷ lệ giao trễ theo khu vực và quốc gia.
- **Vận chuyển:** So sánh hiệu quả các phương thức vận chuyển.
- **Thời gian:** Theo dõi xu hướng giao trễ theo thời gian.
- **Đơn hàng:** Phân tích các đặc điểm của đơn hàng.
- **Dự báo ML:** Hiển thị kết quả và chỉ số đánh giá mô hình.
- **Dự báo đơn hàng mới:** Nhập thông tin để dự báo nguy cơ giao trễ.

Dashboard hỗ trợ các bộ lọc tương tác và biểu đồ như biểu đồ cột, biểu đồ đường, Heatmap và đường cong ROC.

### Dự báo đơn hàng mới

Người dùng nhập 5 thông tin:

- Phương thức vận chuyển.
- Khu vực.
- Quốc gia.
- Số ngày vận chuyển dự kiến.
- Tháng đặt hàng.

Hệ thống sử dụng mô hình Random Forest đã huấn luyện để đưa ra nhãn dự báo và xác suất giao trễ.

## 11. Hướng dẫn sử dụng

### Bước 1: Chuẩn bị môi trường

Truy cập Google Colab và mở notebook của dự án.

### Bước 2: Kết nối Google Drive

Kết nối Google Drive để truy cập bộ dữ liệu và lưu kết quả xử lý.

### Bước 3: Chạy notebook

Thực hiện lần lượt các phần:

1. Cài đặt và khởi tạo PySpark.
2. Đọc và làm sạch dữ liệu.
3. Tích hợp và lưu dữ liệu Parquet.
4. Phân tích bằng Spark SQL.
5. Huấn luyện và đánh giá mô hình.
6. Xuất dữ liệu và mô hình phục vụ Dashboard.

### Bước 4: Chạy Dashboard

Sau khi chuẩn bị đầy đủ các file dữ liệu và mô hình đã huấn luyện, chạy ứng dụng Streamlit:

```bash
streamlit run app.py
```

Lưu ý: Cần cấu hình đúng đường dẫn tới các file dữ liệu và mô hình theo môi trường thực tế.

## 12. Hạn chế của dự án

- Bộ dữ liệu được thu thập trong giai đoạn 2015–2018, chưa phản ánh đầy đủ điều kiện vận chuyển hiện tại.
- Mô hình chưa phát hiện được tất cả các đơn hàng giao trễ.
- Chưa tích hợp dữ liệu thời tiết, giao thông và các yếu tố bên ngoài.
- Hệ thống hiện tập trung vào phân tích và dự báo thử nghiệm, chưa triển khai trong môi trường vận hành thực tế.

## 13. Hướng phát triển

Trong tương lai, nhóm có thể tiếp tục:

- Cập nhật dữ liệu đơn hàng mới.
- Bổ sung các đặc trưng có ý nghĩa đối với vận chuyển.
- Thử nghiệm thêm các thuật toán học máy.
- Cải thiện khả năng phát hiện đơn hàng giao trễ.
- Tích hợp dữ liệu thời tiết và giao thông.
- Phát triển hệ thống dự báo phục vụ hoạt động thực tế của doanh nghiệp.

## 14. Kết luận

Dự án đã xây dựng được quy trình phân tích và tích hợp dữ liệu từ bộ DataCo Smart Supply Chain, sử dụng PySpark và Spark SQL để xử lý dữ liệu, đồng thời ứng dụng Random Forest để dự báo nguy cơ giao trễ.

Bên cạnh kết quả mô hình, Dashboard giúp trực quan hóa thông tin và hỗ trợ người dùng phân tích tình trạng giao hàng theo nhiều góc nhìn.

Dự án là cơ sở để tiếp tục nghiên cứu và phát triển các giải pháp ứng dụng dữ liệu lớn và học máy trong quản lý chuỗi cung ứng.

---

**Nhóm 6 – Phân tích và tích hợp dữ liệu lớn**  
**Đại học Phenikaa**
