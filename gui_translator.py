"""
Mục đích: Tạo giao diện người dùng cho ứng dụng dịch thuật
"""

from tkinter import *
from tkinter import ttk, messagebox
from inference import Translator
import threading

# Màu sắc
BG_COLOR = "#2C3E50"
BG_GRAY = "#34495E"
TEXT_COLOR = "#ECF0F1"
BUTTON_COLOR = "#3498DB"
BUTTON_HOVER = "#2980B9"
ACCENT_COLOR = "#E74C3C"
FONT = "Segoe UI 11"
FONT_BOLD = "Segoe UI 12 bold"
FONT_TITLE = "Segoe UI 16 bold"

class TranslatorGUI:
    def __init__(self):
        self.window = Tk()
        self.translator = None
        self.loading = False
        self.setup_window()
        self.create_widgets()
        self.load_model_async()
    
    def setup_window(self):
        """
        Thiết lập cửa sổ chính
        """
        self.window.title("Dịch Tiếng Anh - Tiếng Việt")
        self.window.geometry("700x600")
        self.window.resizable(False, False)
        self.window.configure(bg=BG_COLOR)
        
        # Center window
        self.center_window()
    
    def center_window(self):
        """
        Căn giữa cửa sổ
        """
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """
        Tạo các widget cho giao diện
        """
        # Header
        header_frame = Frame(self.window, bg=BG_COLOR)
        header_frame.pack(fill=X, pady=(20, 10))
        
        title_label = Label(
            header_frame,
            text="🌐 Dịch Tiếng Anh - Tiếng Việt",
            font=FONT_TITLE,
            bg=BG_COLOR,
            fg=TEXT_COLOR
        )
        title_label.pack()
        
        subtitle_label = Label(
            header_frame,
            text="Sử dụng LSTM Neural Network",
            font="Segoe UI 9",
            bg=BG_COLOR,
            fg=BG_GRAY
        )
        subtitle_label.pack()
        
        # Main container
        main_frame = Frame(self.window, bg=BG_COLOR)
        main_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)
        
        # Input section
        input_frame = Frame(main_frame, bg=BG_COLOR)
        input_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        Label(
            input_frame,
            text="English (Input)",
            font=FONT_BOLD,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            anchor=W
        ).pack(fill=X, pady=(0, 5))
        
        # Input text area with scrollbar
        input_text_frame = Frame(input_frame, bg=BG_GRAY)
        input_text_frame.pack(fill=BOTH, expand=True)
        
        self.input_text = Text(
            input_text_frame,
            height=8,
            font=FONT,
            bg="#ECF0F1",
            fg="#2C3E50",
            wrap=WORD,
            padx=10,
            pady=10,
            relief=FLAT
        )
        self.input_text.pack(side=LEFT, fill=BOTH, expand=True)
        
        input_scrollbar = Scrollbar(input_text_frame, command=self.input_text.yview)
        input_scrollbar.pack(side=RIGHT, fill=Y)
        self.input_text.config(yscrollcommand=input_scrollbar.set)
        
        # Button frame
        button_frame = Frame(main_frame, bg=BG_COLOR)
        button_frame.pack(fill=X, pady=10)
        
        self.translate_btn = Button(
            button_frame,
            text="Dịch ➜",
            font=FONT_BOLD,
            bg=BUTTON_COLOR,
            fg="white",
            activebackground=BUTTON_HOVER,
            activeforeground="white",
            relief=FLAT,
            cursor="hand2",
            padx=30,
            pady=10,
            command=self.translate_text
        )
        self.translate_btn.pack(side=LEFT, expand=True, fill=X, padx=(0, 5))
        
        clear_btn = Button(
            button_frame,
            text="Xóa",
            font=FONT_BOLD,
            bg=ACCENT_COLOR,
            fg="white",
            activebackground="#C0392B",
            activeforeground="white",
            relief=FLAT,
            cursor="hand2",
            padx=30,
            pady=10,
            command=self.clear_text
        )
        clear_btn.pack(side=LEFT, expand=True, fill=X, padx=(5, 0))
        
        # Output section
        output_frame = Frame(main_frame, bg=BG_COLOR)
        output_frame.pack(fill=BOTH, expand=True)
        
        Label(
            output_frame,
            text="Vietnamese (Output)",
            font=FONT_BOLD,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            anchor=W
        ).pack(fill=X, pady=(0, 5))
        
        # Output text area with scrollbar
        output_text_frame = Frame(output_frame, bg=BG_GRAY)
        output_text_frame.pack(fill=BOTH, expand=True)
        
        self.output_text = Text(
            output_text_frame,
            height=8,
            font=FONT,
            bg="#D5DBDB",
            fg="#2C3E50",
            wrap=WORD,
            padx=10,
            pady=10,
            relief=FLAT,
            state=DISABLED
        )
        self.output_text.pack(side=LEFT, fill=BOTH, expand=True)
        
        output_scrollbar = Scrollbar(output_text_frame, command=self.output_text.yview)
        output_scrollbar.pack(side=RIGHT, fill=Y)
        self.output_text.config(yscrollcommand=output_scrollbar.set)
        
        # Status bar
        self.status_label = Label(
            self.window,
            text="Đang load mô hình...",
            font="Segoe UI 9",
            bg=BG_GRAY,
            fg=TEXT_COLOR,
            anchor=W,
            padx=10,
            pady=5
        )
        self.status_label.pack(fill=X, side=BOTTOM)
        
        # Bind Enter key
        self.input_text.bind('<Control-Return>', lambda e: self.translate_text())
    
    def load_model_async(self):
        """
        Load mô hình trong background thread
        """
        def load():
            try:
                self.translator = Translator()
                self.window.after(0, self.on_model_loaded)
            except Exception as e:
                self.window.after(0, lambda: self.on_model_error(str(e)))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def on_model_loaded(self):
        """
        Callback khi model load xong
        """
        self.status_label.config(text="✅ Sẵn sàng dịch thuật! (Ctrl+Enter để dịch nhanh)")
        self.translate_btn.config(state=NORMAL)
        self.input_text.focus()
    
    def on_model_error(self, error):
        """
        Callback khi có lỗi load model
        """
        self.status_label.config(text=f"❌ Lỗi: {error}")
        messagebox.showerror("Lỗi", f"Không thể load mô hình:\n{error}")
        self.translate_btn.config(state=DISABLED)
    
    def translate_text(self):
        """
        Thực hiện dịch văn bản
        """
        if not self.translator:
            messagebox.showwarning("Cảnh báo", "Mô hình chưa được load!")
            return
        
        # Lấy text input
        input_text = self.input_text.get("1.0", END).strip()
        
        if not input_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập văn bản cần dịch!")
            return
        
        # Disable button khi đang dịch
        self.translate_btn.config(state=DISABLED)
        self.status_label.config(text="🔄 Đang dịch...")
        
        def translate():
            try:
                # Dịch từng câu
                sentences = input_text.split('\n')
                translations = []
                
                for sentence in sentences:
                    if sentence.strip():
                        translated = self.translator.translate(sentence.strip())
                        translations.append(translated)
                    else:
                        translations.append('')
                
                result = '\n'.join(translations)
                
                # Update UI
                self.window.after(0, lambda: self.display_translation(result))
            except Exception as e:
                self.window.after(0, lambda: self.on_translation_error(str(e)))
        
        # Chạy trong thread riêng
        thread = threading.Thread(target=translate, daemon=True)
        thread.start()
    
    def display_translation(self, translation):
        """
        Hiển thị kết quả dịch
        """
        self.output_text.config(state=NORMAL)
        self.output_text.delete("1.0", END)
        self.output_text.insert("1.0", translation)
        self.output_text.config(state=DISABLED)
        
        self.status_label.config(text="✅ Dịch hoàn tất!")
        self.translate_btn.config(state=NORMAL)
    
    def on_translation_error(self, error):
        """
        Xử lý lỗi khi dịch
        """
        messagebox.showerror("Lỗi", f"Lỗi khi dịch:\n{error}")
        self.status_label.config(text="❌ Lỗi khi dịch!")
        self.translate_btn.config(state=NORMAL)
    
    def clear_text(self):
        """
        Xóa toàn bộ text
        """
        self.input_text.delete("1.0", END)
        self.output_text.config(state=NORMAL)
        self.output_text.delete("1.0", END)
        self.output_text.config(state=DISABLED)
        self.status_label.config(text="✅ Đã xóa!")
        self.input_text.focus()
    
    def run(self):
        """
        Chạy ứng dụng
        """
        self.window.mainloop()

def main():
    app = TranslatorGUI()
    app.run()

if __name__ == "__main__":
    main()
    