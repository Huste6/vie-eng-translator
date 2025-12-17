"""
Mục đích: Huấn luyện mô hình dịch thuật Tiếng Anh -> Tiếng Việt sử dụng LSTM
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os

# Khởi tạo các biến toàn cục
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

def load_dataset(filepath='data.txt', num_samples=None):
    """
    Đọc và xử lý dataset từ file
    """
    global input_texts, target_texts, input_characters, target_characters
    
    print("Đang đọc dataset...")
    with open(filepath, 'r', encoding='utf-8') as f:
        rows = f.read().split('\n')
    
    # Lấy num_samples dòng đầu hoặc toàn bộ nếu None
    if num_samples:
        rows = rows[:num_samples]
    else:
        rows = rows[:-1] if rows[-1] == '' else rows
    
    for row in rows:
        if '\t' not in row:
            continue
        # Split English và Vietnamese
        input_text, target_text = row.split('\t')
        
        # Thêm ký tự bắt đầu và kết thúc cho target (Vietnamese)
        target_text = '\t' + target_text + '\n'
        
        input_texts.append(input_text.lower())
        target_texts.append(target_text.lower())
        
        # Lấy tất cả các ký tự
        input_characters.update(list(input_text.lower()))
        target_characters.update(list(target_text.lower()))
    
    # Sắp xếp và lấy thông tin
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    
    num_en_chars = len(input_characters)
    num_vie_chars = len(target_characters)
    max_input_length = max([len(txt) for txt in input_texts])
    max_target_length = max([len(txt) for txt in target_texts])
    
    print(f"Số lượng mẫu: {len(input_texts)}")
    print(f"Số ký tự tiếng Anh: {num_en_chars}")
    print(f"Số ký tự tiếng Việt: {num_vie_chars}")
    print(f"Độ dài tối đa câu tiếng Anh: {max_input_length}")
    print(f"Độ dài tối đa câu tiếng Việt: {max_target_length}")
    
    return (num_en_chars, num_vie_chars, max_input_length, max_target_length)

def vectorize_data(input_texts, target_texts, input_characters, target_characters, 
                   max_input_length, max_target_length):
    """
    Chuyển đổi text thành vector (One-hot encoding)
    """
    print("\nĐang vector hóa dữ liệu...")
    
    num_samples = len(input_texts)
    num_en_chars = len(input_characters)
    num_vie_chars = len(target_characters)
    
    # Tạo dictionary để map ký tự -> index
    input_char_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_char_index = dict([(char, i) for i, char in enumerate(target_characters)])
    
    # Khởi tạo ma trận 3D
    encoder_input_data = np.zeros(
        (num_samples, max_input_length, num_en_chars), dtype='float32')
    decoder_input_data = np.zeros(
        (num_samples, max_target_length, num_vie_chars), dtype='float32')
    decoder_target_data = np.zeros(
        (num_samples, max_target_length, num_vie_chars), dtype='float32')
    
    # Fill dữ liệu
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_char_index[char]] = 1.0
        
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_char_index[char]] = 1.0
            # decoder_target_data không có ký tự đầu tiên (\t)
            if t > 0:
                decoder_target_data[i, t - 1, target_char_index[char]] = 1.0
    
    print("Vector hóa hoàn tất!")
    return encoder_input_data, decoder_input_data, decoder_target_data, input_char_index, target_char_index

def build_model(num_en_chars, num_vie_chars, latent_dim=256):
    """
    Xây dựng mô hình Seq2Seq với 3 lớp LSTM
    """
    print("\nĐang xây dựng mô hình...")
    
    # ========== ENCODER ==========
    encoder_inputs = Input(shape=(None, num_en_chars), name='encoder_input')
    
    # LSTM Layer 1
    encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, 
                         name='encoder_lstm_1')
    encoder_outputs1, _, _ = encoder_lstm1(encoder_inputs)
    encoder_dropout1 = Dropout(0.2)(encoder_outputs1)
    
    # LSTM Layer 2
    encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True,
                         name='encoder_lstm_2')
    encoder_outputs2, _, _ = encoder_lstm2(encoder_dropout1)
    encoder_dropout2 = Dropout(0.2)(encoder_outputs2)
    
    # LSTM Layer 3 (Final)
    encoder_lstm3 = LSTM(latent_dim, return_state=True, name='encoder_lstm_3')
    encoder_outputs3, state_h, state_c = encoder_lstm3(encoder_dropout2)
    
    # Chỉ lấy states từ lớp cuối
    encoder_states = [state_h, state_c]
    
    # ========== DECODER ==========
    decoder_inputs = Input(shape=(None, num_vie_chars), name='decoder_input')
    
    # LSTM Layer 1
    decoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True,
                         name='decoder_lstm_1')
    decoder_outputs1, _, _ = decoder_lstm1(decoder_inputs, initial_state=encoder_states)
    decoder_dropout1 = Dropout(0.2)(decoder_outputs1)
    
    # LSTM Layer 2
    decoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True,
                         name='decoder_lstm_2')
    decoder_outputs2, _, _ = decoder_lstm2(decoder_dropout1)
    decoder_dropout2 = Dropout(0.2)(decoder_outputs2)
    
    # LSTM Layer 3
    decoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True,
                         name='decoder_lstm_3')
    decoder_outputs3, _, _ = decoder_lstm3(decoder_dropout2)
    
    # Dense Layer (Output)
    decoder_dense = Dense(num_vie_chars, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs3)
    
    # Tạo model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    print("Mô hình đã được xây dựng!")
    return model

def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data,
                epochs=100, batch_size=64, validation_split=0.2):
    """
    Huấn luyện mô hình
    """
    print("\nBắt đầu huấn luyện mô hình...")
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )
    
    print("\nHuấn luyện hoàn tất!")
    return history

def save_model_and_data(model, input_characters, target_characters, 
                        max_input_length, max_target_length,
                        num_en_chars, num_vie_chars,
                        input_char_index, target_char_index):
    """
    Lưu mô hình và các thông tin cần thiết
    """
    print("\nĐang lưu mô hình và dữ liệu...")
    
    # Tạo thư mục nếu chưa có
    os.makedirs('models', exist_ok=True)
    
    # Lưu model
    model.save('models/vie_eng_translator.h5')
    
    # Lưu các thông tin khác
    training_data = {
        'input_characters': input_characters,
        'target_characters': target_characters,
        'max_input_length': max_input_length,
        'max_target_length': max_target_length,
        'num_en_chars': num_en_chars,
        'num_vie_chars': num_vie_chars,
        'input_char_index': input_char_index,
        'target_char_index': target_char_index
    }
    
    with open('models/training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    
    print("Đã lưu mô hình và dữ liệu!")
    print("- models/vie_eng_translator.h5")
    print("- models/training_data.pkl")

def main():
    """
    Hàm chính để chạy toàn bộ quá trình training
    """
    # 1. Load dataset
    num_en_chars, num_vie_chars, max_input_length, max_target_length = load_dataset(
        filepath='data.txt', 
        num_samples=None  # None = lấy tất cả
    )
    
    # 2. Vectorize data
    encoder_input_data, decoder_input_data, decoder_target_data, input_char_index, target_char_index = vectorize_data(
        input_texts, target_texts, input_characters, target_characters,
        max_input_length, max_target_length
    )
    
    # 3. Build model
    model = build_model(num_en_chars, num_vie_chars, latent_dim=256)
    
    # Hiển thị summary
    model.summary()
    
    # 4. Train model
    history = train_model(
        model, encoder_input_data, decoder_input_data, decoder_target_data,
        epochs=100,  # Có thể điều chỉnh
        batch_size=64,
        validation_split=0.2
    )
    
    # 5. Save model và data
    save_model_and_data(
        model, input_characters, target_characters,
        max_input_length, max_target_length,
        num_en_chars, num_vie_chars,
        input_char_index, target_char_index
    )
    
    print("\n✅ Hoàn tất toàn bộ quá trình training!")

if __name__ == "__main__":
    main()
    