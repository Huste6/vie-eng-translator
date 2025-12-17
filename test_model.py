"""
Mục đích: Test nhanh mô hình với input từ người dùng
"""

from inference import Translator
import sys

def test_interactive():
    """
    Chế độ test tương tác
    """
    print("="*60)
    print("🌐 CHƯƠNG TRÌNH DỊCH TIẾNG ANH - TIẾNG VIỆT")
    print("="*60)
    print("Đang khởi tạo mô hình...")
    
    try:
        translator = Translator()
        print("✅ Mô hình đã sẵn sàng!\n")
    except Exception as e:
        print(f"❌ Lỗi khi load mô hình: {e}")
        return
    
    print("Nhập 'quit' hoặc 'exit' để thoát")
    print("="*60 + "\n")
    
    while True:
        # Nhập câu tiếng Anh
        english_text = input("🇬🇧 English: ").strip()
        
        # Kiểm tra thoát
        if english_text.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Tạm biệt!")
            break
        
        # Kiểm tra rỗng
        if not english_text:
            print("⚠️  Vui lòng nhập câu tiếng Anh!\n")
            continue
        
        # Dịch
        try:
            vietnamese_text = translator.translate(english_text)
            print(f"🇻🇳 Vietnamese: {vietnamese_text}\n")
        except Exception as e:
            print(f"❌ Lỗi khi dịch: {e}\n")

def test_batch(sentences):
    """
    Test với một batch câu
    """
    print("="*60)
    print("🌐 TEST BATCH TRANSLATION")
    print("="*60)
    print("Đang khởi tạo mô hình...\n")
    
    try:
        translator = Translator()
        print("✅ Mô hình đã sẵn sàng!\n")
    except Exception as e:
        print(f"❌ Lỗi khi load mô hình: {e}")
        return
    
    print("="*60)
    print("KẾT QUẢ DỊCH THUẬT")
    print("="*60 + "\n")
    
    for i, sentence in enumerate(sentences, 1):
        try:
            translation = translator.translate(sentence)
            print(f"[{i}] EN: {sentence}")
            print(f"    VI: {translation}")
            print("-" * 60)
        except Exception as e:
            print(f"[{i}] EN: {sentence}")
            print(f"    ❌ Lỗi: {e}")
            print("-" * 60)

def main():
    """
    Hàm chính
    """
    # Kiểm tra arguments
    if len(sys.argv) > 1:
        # Nếu có argument, dịch trực tiếp
        sentence = ' '.join(sys.argv[1:])
        
        try:
            translator = Translator()
            translation = translator.translate(sentence)
            print(f"EN: {sentence}")
            print(f"VI: {translation}")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
    else:
        # Menu lựa chọn
        print("\n" + "="*60)
        print("CHỌN CHẾ ĐỘ TEST")
        print("="*60)
        print("1. Chế độ tương tác (nhập từng câu)")
        print("2. Test với batch câu mẫu")
        print("3. Thoát")
        print("="*60)
        
        choice = input("\nLựa chọn của bạn (1/2/3): ").strip()
        
        if choice == '1':
            test_interactive()
        elif choice == '2':
            # Các câu mẫu để test
            test_sentences = [
                "Run!",
                "Help!",
                "Stop!",
                "Wait!",
                "Hello!",
                "Thank you.",
                "Good luck!",
                "I'm sorry.",
                "Be careful.",
                "Come here.",
                "Don't worry.",
                "I'm tired.",
                "See you later.",
                "What's your name?",
                "How are you?"
            ]
            test_batch(test_sentences)
        elif choice == '3':
            print("\n👋 Tạm biệt!")
        else:
            print("⚠️  Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()