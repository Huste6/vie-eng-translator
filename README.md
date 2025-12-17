# 🌐 Dự Án Dịch Thuật Tiếng Anh - Tiếng Việt với LSTM

Dự án Machine Learning dịch thuật từ Tiếng Anh sang Tiếng Việt sử dụng mạng LSTM (Long Short-Term Memory) với kiến trúc Encoder-Decoder.

## 📁 Cấu Trúc Dự Án

```
vie-eng-translator/
│
├── data.txt                    # Dataset Tiếng Anh - Tiếng Việt
├── train_model.py             # File huấn luyện mô hình
├── inference.py               # File dự đoán/dịch thuật
├── gui_translator.py          # Giao diện người dùng (Tkinter)
├── requirements.txt           # Thư viện cần thiết
├── README.md                  # Hướng dẫn
│
└── models/                    # Thư mục chứa mô hình đã train
    ├── vie_eng_translator.h5  # Mô hình đã huấn luyện
    └── training_data.pkl      # Dữ liệu training (metadata)
```

## 🎯 Tính Năng

✅ **Mô hình LSTM 3 lớp**: Sử dụng 3 lớp LSTM cho cả Encoder và Decoder  
✅ **Dropout**: Tránh overfitting với Dropout layers  
✅ **Teacher Forcing**: Cải thiện độ chính xác khi training  
✅ **One-hot Encoding**: Vector hóa ký tự  
✅ **GUI thân thiện**: Giao diện Tkinter đẹp mắt và dễ sử dụng  
✅ **Batch Processing**: Xử lý nhiều câu cùng lúc

## 🛠️ Kiến Trúc Mô Hình

### Encoder (3 lớp LSTM):
```
Input → LSTM1(256) → Dropout(0.2) → 
        LSTM2(256) → Dropout(0.2) → 
        LSTM3(256) → [state_h, state_c]
```

### Decoder (3 lớp LSTM):
```
Input + Encoder States → LSTM1(256) → Dropout(0.2) → 
                          LSTM2(256) → Dropout(0.2) → 
                          LSTM3(256) → Dense(softmax)
```

## 📋 Yêu Cầu Hệ Thống

- Python 3.8+
- TensorFlow 2.13+
- NumPy 1.24+
- Scikit-learn 1.3+
- Tkinter (có sẵn với Python)

## 🚀 Cài Đặt

### 1. Clone hoặc tải dự án
```bash
git clone <your-repo-url>
cd vie-eng-translator
```

### 2. Tạo và kích hoạt môi trường ảo (Virtual Environment)

**Trên Windows:**
```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt venv
venv\Scripts\activate
```

**Trên Linux/macOS:**
```bash
# Tạo virtual environment
python3 -m venv venv

# Kích hoạt venv
source venv/bin/activate
```

> 💡 **Lưu ý**: Sau khi kích hoạt, bạn sẽ thấy `(venv)` xuất hiện ở đầu dòng lệnh.

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị dataset
Đảm bảo file `data.txt` có format:
```
English sentence<TAB>Vietnamese sentence
```

Ví dụ:
```
Run!	Chạy!
Help!	Giúp tôi với!
Stop!	Dừng lại!
```

## 📚 Hướng Dẫn Sử Dụng

### Bước 1: Huấn Luyện Mô Hình
```bash
python train_model.py
```

Tham số có thể tùy chỉnh trong file:
- `num_samples`: Số lượng mẫu training (None = tất cả)
- `epochs`: Số epoch (mặc định: 100)
- `batch_size`: Kích thước batch (mặc định: 64)
- `latent_dim`: Số neurons trong LSTM (mặc định: 256)
- `validation_split`: Tỷ lệ validation (mặc định: 0.2)

**Output:**
- ✅ `models/vie_eng_translator.h5`
- ✅ `models/training_data.pkl`

### Bước 2: Kiểm Tra Mô Hình (Terminal)
```bash
python inference.py
```
Sẽ test với một số câu mẫu và hiển thị kết quả dịch.

### Bước 3: Chạy Giao Diện
```bash
python gui_translator.py
```

Tính năng GUI:
- ✨ Nhập văn bản tiếng Anh
- 🔄 Nhấn "Dịch" hoặc Ctrl+Enter
- 📝 Xem kết quả tiếng Việt
- 🗑️ Xóa và dịch lại

## 💡 Tips Tối Ưu

### 1. Tăng độ chính xác:
- Tăng số lượng dữ liệu training
- Tăng số epochs (200-300)
- Thử nghiệm với latent_dim khác (128, 512)
- Thêm nhiều lớp LSTM hơn

### 2. Giảm overfitting:
- Tăng Dropout rate (0.3-0.5)
- Tăng validation_split (0.25-0.3)
- Sử dụng Early Stopping

### 3. Tăng tốc training:
- Giảm batch_size nếu GPU bị hết RAM
- Sử dụng GPU thay vì CPU
- Giảm max_input_length và max_target_length

## 📊 Kết Quả Mong Đợi

### Với dataset nhỏ (~100 câu):
- Training Accuracy: 80-95%
- Validation Accuracy: 70-85%
- Training Time: 5-15 phút (tùy CPU/GPU)

### Với dataset lớn (10,000+ câu):
- Training Accuracy: 95-99%
- Validation Accuracy: 85-95%
- Training Time: 2-5 giờ (tùy hardware)

## 🔧 Troubleshooting

### Lỗi: "Out of Memory"
```python
# Trong train_model.py, giảm:
batch_size = 32  # thay vì 64
latent_dim = 128  # thay vì 256
```

### Lỗi: Model không load được
```bash
# Kiểm tra file tồn tại
ls models/
# Phải có: vie_eng_translator.h5 và training_data.pkl
```

### Lỗi: Dịch không chính xác
- Train thêm epochs
- Thêm dữ liệu training
- Kiểm tra dataset có đúng format không

## 📖 Giải Thích Code

### train_model.py
- `load_dataset()`: Đọc và parse dataset
- `vectorize_data()`: Chuyển text thành vector
- `build_model()`: Xây dựng mô hình LSTM 3 lớp
- `train_model()`: Huấn luyện mô hình
- `save_model_and_data()`: Lưu mô hình và metadata

### inference.py
- `Translator.__init__()`: Load mô hình đã train
- `_build_encoder_model()`: Tạo encoder để extract states
- `_build_decoder_model()`: Tạo decoder để generate output
- `translate()`: Dịch văn bản

### gui_translator.py
- Giao diện Tkinter với threading
- Load mô hình bất đồng bộ
- Hỗ trợ dịch nhiều câu

## 🎓 Kiến Thức Cần Thiết

- **LSTM**: Hiểu cách LSTM hoạt động
- **Seq2Seq**: Kiến trúc Encoder-Decoder
- **One-hot Encoding**: Vector hóa dữ liệu
- **Teacher Forcing**: Kỹ thuật training
- **Keras/TensorFlow**: API của TensorFlow

## 🤝 Đóng Góp

Mọi đóng góp đều được chào đón! Hãy tạo Pull Request hoặc Issue.

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu.

## 👨‍💻 Tác Giả

Dự án Machine Learning - Dịch Thuật Tiếng Anh - Tiếng Việt

---

**Chúc bạn thành công! 🎉**
