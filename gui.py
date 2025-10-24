import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

class SimpleSegmentationApp:
    def __init__(self, root, segmentator):
        self.root = root
        self.segmentator = segmentator
        self.current_image = None
        self.current_background = "blue"
        self.custom_background = None
        self.effects = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        self.root.title("AI Human Segmentation - PRO")
        self.root.geometry("1200x700")
        
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - управление
        control_frame = ttk.LabelFrame(main_frame, text="Управление", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        control_frame.pack_propagate(False)
        
        # Правая панель - изображения
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Создаем элементы управления
        self.create_control_panel(control_frame)
        self.create_image_panel(image_frame)
    
    def create_control_panel(self, parent):
        # Загрузка изображения
        ttk.Button(parent, text="📁 Загрузить изображение", 
                  command=self.load_image).pack(fill=tk.X, pady=5)
        
        ttk.Button(parent, text="📷 Открыть камеру",
                  command=self.open_camera).pack(fill=tk.X, pady=5)
        
        # Выбор фона
        bg_frame = ttk.LabelFrame(parent, text="Фон")
        bg_frame.pack(fill=tk.X, pady=10)
        
        backgrounds = [
            ("🔵 Синий", "blue"),
            ("🟢 Зеленый", "green"), 
            ("🌈 Градиент", "gradient"),
            ("⚫ Черный", "black"),
            ("🌀 Размытие", "blur")
        ]
        
        for text, bg_type in backgrounds:
            ttk.Button(bg_frame, text=text,
                      command=lambda bt=bg_type: self.set_background(bt)).pack(fill=tk.X, pady=2)
        
        # Панель редактирования фона
        editor_frame = ttk.LabelFrame(parent, text="🎨 Редактор фона")
        editor_frame.pack(fill=tk.X, pady=10)
        
        # Слайдеры
        self.create_slider(editor_frame, "Яркость:", "brightness", 0.5, 1.5, 1.0)
        self.create_slider(editor_frame, "Контраст:", "contrast", 0.5, 2.0, 1.0)
        self.create_slider(editor_frame, "Оттенок:", "hue", -30, 30, 0)
        self.create_slider(editor_frame, "Размытие:", "blur", 0, 5, 0)
        
        # Пресеты
        preset_frame = ttk.Frame(editor_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        
        presets = [
            ("🔵 DION Blue", "dion_blue"),
            ("⚫ DION Dark", "dion_dark"),
            ("🏢 DION Modern", "dion_modern")
        ]
        
        for text, preset in presets:
            ttk.Button(preset_frame, text=text,
                      command=lambda p=preset: self.apply_preset(p)).pack(side=tk.LEFT, padx=2)
    
    def create_slider(self, parent, label, effect, min_val, max_val, default):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label).pack(anchor=tk.W)
        
        var = tk.DoubleVar(value=default)
        scale = ttk.Scale(frame, variable=var, from_=min_val, to=max_val,
                         command=lambda v, e=effect: self.on_slider_change(e, float(v)))
        scale.pack(fill=tk.X)
        
        # Сохраняем ссылку на переменную
        setattr(self, f"{effect}_var", var)
    
    def create_image_panel(self, parent):
        # Оригинал
        orig_frame = ttk.LabelFrame(parent, text="Оригинал")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        
        self.orig_canvas = tk.Canvas(orig_frame, bg='#333')
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Результат
        result_frame = ttk.LabelFrame(parent, text="Результат")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0))
        
        self.result_canvas = tk.Canvas(result_frame, bg='#333')
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Заглушки
        self.show_placeholder(self.orig_canvas, "Загрузите изображение")
        self.show_placeholder(self.result_canvas, "Результат появится здесь")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.current_image = file_path
            image = Image.open(file_path)
            self.display_image(image, self.orig_canvas)
            self.process_current_image()
    
    def open_camera(self):
        print("📷 Режим камеры (реализовать)")
    
    def set_background(self, bg_type):
        self.current_background = bg_type
        self.process_current_image()
    
    def on_slider_change(self, effect, value):
        self.effects[effect] = value
        self.process_current_image()
    
    def apply_preset(self, preset_name):
        self.effects['preset'] = preset_name
        self.process_current_image()
    
    def process_current_image(self):
        if not self.current_image:
            return
            
        def process():
            try:
                mask, result = self.segmentator.process_image(
                    self.current_image, 
                    self.current_background,
                    effects=self.effects
                )
                
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                result_pil = Image.fromarray(result_bgr)
                self.display_image(result_pil, self.result_canvas)
                
            except Exception as e:
                print(f"Ошибка обработки: {e}")
        
        threading.Thread(target=process, daemon=True).start()
    
    def display_image(self, image, canvas):
        canvas.delete("all")
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1:
            self.root.after(100, lambda: self.display_image(image, canvas))
            return
        
        # Масштабируем изображение
        img_width, img_height = image.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_size = (int(img_width*ratio), int(img_height*ratio))
        
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        
        x = (canvas_width - new_size[0]) // 2
        y = (canvas_height - new_size[1]) // 2
        
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        canvas.image = photo
    
    def show_placeholder(self, canvas, text):
        canvas.delete("all")
        canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2,
                          text=text, fill="white", font=("Arial", 12))