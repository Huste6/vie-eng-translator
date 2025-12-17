"""
Mục đích: Load mô hình đã train và thực hiện dịch thuật
"""

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import numpy as np
import pickle

class Translator:
    def __init__(self, model_path='models/vie_eng_translator.h5', 
                 data_path='models/training_data.pkl'):
        """
        Khởi tạo translator với mô hình đã train
        """
        print("Đang load mô hình...")
        
        # Load training data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.input_characters = data['input_characters']
        self.target_characters = data['target_characters']
        self.max_input_length = data['max_input_length']
        self.max_target_length = data['max_target_length']
        self.num_en_chars = data['num_en_chars']
        self.num_vie_chars = data['num_vie_chars']
        self.input_char_index = data['input_char_index']
        self.target_char_index = data['target_char_index']
        
        # Tạo reverse mapping cho target
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_char_index.items()
        )
        
        # Load model đã train
        self.model = load_model(model_path)
        
        # Tạo encoder model
        self.encoder_model = self._build_encoder_model()
        
        # Tạo decoder model
        self.decoder_model = self._build_decoder_model()
        
        print("✅ Load mô hình hoàn tất!")
    
    def _build_encoder_model(self):
        """
        Xây dựng encoder model từ model đã train
        """
        # Lấy encoder input
        encoder_inputs = self.model.input[0]
        
        # Lấy outputs từ encoder LSTM cuối cùng (layer thứ 5 - encoder_lstm_3)
        # Layer 0: encoder_input
        # Layer 1: encoder_lstm_1
        # Layer 2: dropout
        # Layer 3: encoder_lstm_2
        # Layer 4: dropout
        # Layer 5: encoder_lstm_3
        _, state_h, state_c = self.model.layers[5].output
        
        encoder_states = [state_h, state_c]
        encoder_model = Model(encoder_inputs, encoder_states)
        
        return encoder_model
    
    def _build_decoder_model(self):
        """
        Xây dựng decoder model từ model đã train
        """
        # Decoder inputs
        decoder_inputs = self.model.input[1]
        
        # State inputs
        decoder_state_input_h = Input(shape=(256,))
        decoder_state_input_c = Input(shape=(256,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        # Lấy các LSTM layers từ model
        # Layer 7: decoder_lstm_1
        # Layer 9: decoder_lstm_2
        # Layer 11: decoder_lstm_3
        # Layer 12: decoder_dense
        
        decoder_lstm1 = self.model.layers[7]
        decoder_lstm2 = self.model.layers[9]
        decoder_lstm3 = self.model.layers[11]
        decoder_dense = self.model.layers[12]
        
        # Chạy qua các layers
        decoder_outputs, state_h1, state_c1 = decoder_lstm1(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_outputs, state_h2, state_c2 = decoder_lstm2(decoder_outputs)
        decoder_outputs, state_h3, state_c3 = decoder_lstm3(decoder_outputs)
        
        decoder_states = [state_h3, state_c3]
        decoder_outputs = decoder_dense(decoder_outputs)
        
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )
        
        return decoder_model
    
    def encode_input(self, input_text):
        """
        Encode input text thành vector
        """
        input_text = input_text.lower()
        encoder_input_data = np.zeros(
            (1, self.max_input_length, self.num_en_chars),
            dtype='float32'
        )
        
        for t, char in enumerate(input_text):
            if char in self.input_char_index:
                encoder_input_data[0, t, self.input_char_index[char]] = 1.0
        
        return encoder_input_data
    
    def decode_sequence(self, input_seq):
        """
        Decode sequence từ input
        """
        # Encode input và lấy states
        states_value = self.encoder_model.predict(input_seq, verbose=0)
        
        # Tạo target sequence với ký tự bắt đầu
        target_seq = np.zeros((1, 1, self.num_vie_chars))
        target_seq[0, 0, self.target_char_index['\t']] = 1.0
        
        # Decoded sentence
        decoded_sentence = ''
        stop_condition = False
        
        while not stop_condition:
            # Predict tiếp theo
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value, verbose=0
            )
            
            # Lấy ký tự có xác suất cao nhất
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            
            decoded_sentence += sampled_char
            
            # Điều kiện dừng
            if (sampled_char == '\n' or 
                len(decoded_sentence) > self.max_target_length):
                stop_condition = True
            
            # Update target sequence
            target_seq = np.zeros((1, 1, self.num_vie_chars))
            target_seq[0, 0, sampled_token_index] = 1.0
            
            # Update states
            states_value = [h, c]
        
        return decoded_sentence.strip('\n\t')
    
    def translate(self, text):
        """
        Dịch text từ Tiếng Anh sang Tiếng Việt
        """
        encoded_input = self.encode_input(text)
        translation = self.decode_sequence(encoded_input)
        return translation

def main():
    """
    Test translator
    """
    # Khởi tạo translator
    translator = Translator()
    
    # Test một vài câu
    test_sentences = [
        "Run!",
        "Help!",
        "Stop!",
        "Hello!",
        "Thank you.",
        "Good luck!",
        "I'm sorry.",
        "Welcome."
    ]
    
    print("\n" + "="*50)
    print("KIỂM TRA DỊCH THUẬT")
    print("="*50)
    
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"EN: {sentence}")
        print(f"VI: {translation}")
        print("-" * 50)

if __name__ == "__main__":
    main()