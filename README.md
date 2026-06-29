# Hệ thống Nhận Dạng và Trích Xuất Thông Tin Căn Cước Công Dân (CCCD)

* Dự án này cung cấp một pipline hoàn chỉnh từ đầu đến cuối để phát hiện, cắt và trích xuất thông tin văn bản (Số CCCD và Họ Tên) từ ảnh chụp mặt trước của Căn cước Công dân (CCCD) Việt Nam.
* Giải pháp kết hợp sức mạnh của mô hình **YOLOv8** (để định vị thẻ) và mạng **CRNN + CTC Loss** (để nhận dạng chữ - OCR).

---

## Tính Năng Chính

1. **Tiền xử lý tự động:** Chuyển đổi dữ liệu mask ảnh thành định dạng nhãn bounding box chuẩn của YOLO.
2. **Định vị & Cắt thẻ (Object Detection):** Sử dụng `YOLOv8n` siêu nhẹ để nhận diện vị trí thẻ CCCD trên ảnh gốc và cắt ra với độ chính xác cao.
3. **Phân vùng thông tin (Rule-based Cropping):** Chuẩn hóa thẻ về độ phân giải chuẩn (800x500) và dùng "tỷ lệ vàng" để tự động trích xuất vùng chứa **Số thẻ** và **Họ tên**.
4. **Nhận dạng Chữ (OCR):** Huấn luyện mô hình CRNN (CNN + RNN) kết hợp CTC Loss để đọc văn bản tiếng Việt từ các vùng ảnh đã bị cắt.
5. **End-to-End Pipeline:** Lớp `CCCDProcessor` tích hợp toàn bộ quy trình, nhận đầu vào là ảnh thô và trả về JSON chứa thông tin dạng text.

---

## Cấu Trúc Notebook (`CCCD_Front.ipynb`)

Notebook được chia thành 3 phần chính tương ứng với quy trình huấn luyện và kiểm thử:

### PHẦN 1: TÌM THẺ (YOLOv8)

* **Tiền xử lý Mask:** Quét thư mục chứa ảnh mask, dùng OpenCV tìm đường viền (contours) lớn nhất để suy ra Bounding Box của thẻ. Chuyển đổi tọa độ sang chuẩn `[class_id, center_x, center_y, width, height]` và sinh file `data_card.yaml`.
* **Tạo Symlink:** Trỏ liên kết (symlink) giữa ảnh gốc và ảnh train để tiết kiệm bộ nhớ trên nền tảng Kaggle.
* **Huấn luyện YOLOv8:** Cài đặt `ultralytics`, load pre-trained weights `yolov8n.pt` và tiến hành fine-tuning trong 50 epochs.
* **Kiểm thử Cắt thẻ:** Dùng trọng số tốt nhất (`best.pt`) dự đoán trên ảnh mới. Resize ảnh về `800x500` và cắt trường thông tin (Số CCCD, Họ tên) dựa trên tỷ lệ đã định sẵn. Trực quan hóa kết quả bằng Matplotlib.

### PHẦN 2: ĐỌC THẺ (CRNN - OCR)

* **Định nghĩa mạng CRNN:** Xây dựng mô hình với các lớp Tích chập (CNN) để trích xuất đặc trưng, theo sau là các lớp Tuần hoàn (RNN/GRU) để xử lý chuỗi và lớp Fully Connected.
* **Huấn luyện OCR:** Sử dụng hàm mất mát `CTCLoss` và bộ tối ưu `AdamW`. Quá trình giải mã áp dụng phương pháp *Greedy Decoding* (gộp các ký tự giống nhau liền kề và loại bỏ token `<BLANK>`).

### PHẦN 3: TÍCH HỢP THỰC TẾ (PIPELINE)

* Khởi tạo class `CCCDProcessor`.
* Nạp đồng thời 2 mô hình: Trọng số YOLO và Trọng số CRNN.
* Thực hiện toàn bộ quy trình Inference qua hàm `predict()` và trả về kết quả dưới định dạng JSON.

---

## Cài Đặt & Yêu Cầu Môi Trường

* Mã nguồn được viết và tối ưu hóa để chạy trên môi trường **Kaggle Notebook** hoặc **Google Colab** có hỗ trợ GPU.
* Các thư viện chính:
    * `torch` (PyTorch)
    * `ultralytics` (YOLOv8)
    * `opencv-python` (cv2)
    * `numpy`
    * `matplotlib`
    * `tqdm`
* Để cài đặt thủ công, bạn chạy lệnh:

```bash
pip install torch torchvision ultralytics opencv-python matplotlib numpy tqdm

```

---

## Thử Nghiệm Nhanh

* Với file trọng số đã được cập nhật, bạn có thể dễ dàng sử dụng Lớp `CCCDProcessor` được định nghĩa trong Phần 3 của notebook:

```python
from your_module import CCCDProcessor
import json

# 1. Cấu hình đường dẫn
yolo_path = 'runs/detect/train/weights/best.pt'
crnn_path = 'crnn_model_epoch_30.pth'
char2id_path = 'char2id_vietnamese.json'
img_test_path = 'test_image.jpg'

# 2. Khởi tạo Pipeline
processor = CCCDProcessor(yolo_path, crnn_path, char2id_path)

# 3. Trích xuất thông tin
result = processor.predict(img_test_path)
print(json.dumps(result, ensure_ascii=False, indent=2))

```

### Output Trả Về Mẫu:

```json
{
  "cccd_number": "15395321036",
  "full_name": "HOÀNG THỊ THẢO",
  "status": "success"
}

```

---

## Lưu Ý Về Đường Dẫn

* Trong mã nguồn, các đường dẫn dữ liệu hiện đang được trỏ cứng vào bộ nhớ của Kaggle Input:
    * Đường dẫn mask & ảnh: `/kaggle/input/datasets/...`
    * Thư mục output: `/kaggle/working/...`
* Trường hợp mang code về chạy trên máy ảo local (như Jupyter Lab trên PC), bạn **BẮT BUỘC PHẢI THAY ĐỔI** các biến `mask_dir`, `image_dir` và `test_image_path` sao cho trỏ đúng tới thư mục trên thiết bị của bạn trước khi chạy huấn luyện.

---

*Dự án là tài liệu thực hành lý tưởng cho quá trình làm chủ hệ thống Computer Vision ứng dụng trong công việc số hóa quy trình (eKYC / OCR).* Tác giả hi vọng mã nguồn mang lại kiến thức bổ ích cho bạn!
