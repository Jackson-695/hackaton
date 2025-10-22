import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import threading
import time

class AdvancedSegmentationApp:
    """Продвинутый графический интерфейс с AI функциями"""
    
    def __init__(self, root, segmentator):
        self.root = root
        self.segmentator = segmentator
        self.current_image = None
        self.custom_background = None
        self.background_generator = None
        self.style_classifier = None
        
        # Инициализация AI компонентов в отдельном потоке
        self.setup_ai_components()
        self.setup_ui()
        
    def setup_ai_components(self):
        """Инициализация AI компонентов в фоне"""
        def load_ai_components():
            try:
                from background_generator import BackgroundGenerator
                from style_classifier import StyleClassifier
                self.background_generator = BackgroundGenerator()
                self.style_classifier = StyleClassifier()
                print("✅ AI компоненты загружены")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить AI компоненты: {e}")
        
        # Запускаем в отдельном потоке чтобы не блокировать GUI
        ai_thread = threading.Thread(target=load_ai_components, daemon=True)
        ai_thread.start()
    
    def setup_ui(self):
        """Настройка продвинутого интерфейса"""
        self.root.title("🧠 AI Human Segmentation - PRO EDITION")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")
        
        # Стили
        self.setup_styles()
        
        # Создание интерфейса
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
        # Запуск анимации загрузки
        self.animate_loading()
    
    def setup_styles(self):
        """Настройка стилей интерфейса"""
        self.style = ttk.Style()
        self.style.configure("Custom.TFrame", background="#2d2d30")
        self.style.configure("Title.TLabel", 
                           background="#1e1e1e", 
                           foreground="#ffffff",
                           font=("Arial", 16, "bold"))
        self.style.configure("Subtitle.TLabel",
                           background="#1e1e1e",
                           foreground="#cccccc", 
                           font=("Arial", 10))
        
        # Стили для кнопок
        self.style.configure("Primary.TButton",
                           background="#007acc",
                           foreground="white",
                           font=("Arial", 10, "bold"),
                           padding=(15, 8))
        
        self.style.configure("Success.TButton",
                           background="#107c10", 
                           foreground="white",
                           font=("Arial", 10, "bold"),
                           padding=(15, 8))
        
        self.style.configure("Warning.TButton",
                           background="#d83b01",
                           foreground="white", 
                           font=("Arial", 10, "bold"),
                           padding=(15, 8))
    
    def create_header(self):
        """Создание заголовка"""
        header_frame = ttk.Frame(self.root, style="Custom.TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Заголовок с иконкой
        title_frame = ttk.Frame(header_frame, style="Custom.TFrame")
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(title_frame, 
                               text="🧠 AI HUMAN SEGMENTATION PRO", 
                               style="Title.TLabel")
        title_label.pack(side=tk.LEFT)
        
        # Версия и статус
        version_label = ttk.Label(title_frame,
                                 text="v2.0 | Neural Network Powered",
                                 style="Subtitle.TLabel")
        version_label.pack(side=tk.RIGHT)
        
        # Подзаголовок с фичами
        features = "🎯 Real-time Segmentation • 🎨 AI Background Generation • 👔 Style Detection • ⚡ GPU Accelerated"
        features_label = ttk.Label(header_frame,
                                  text=features,
                                  style="Subtitle.TLabel")
        features_label.pack(fill=tk.X, pady=(5, 0))
    
    def create_main_content(self):
        """Создание основного контента"""
        main_frame = ttk.Frame(self.root, style="Custom.TFrame")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Левая панель - управление
        self.create_control_panel(main_frame)
        
        # Правая панель - изображения
        self.create_image_panel(main_frame)
    
    def create_control_panel(self, parent):
        """Панель управления"""
        control_frame = ttk.LabelFrame(parent, 
                                      text="🎛️ CONTROL PANEL",
                                      style="Custom.TFrame",
                                      padding=15)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.config(width=350)
        
        # Секция загрузки
        self.create_load_section(control_frame)
        
        # Секция фонов
        self.create_background_section(control_frame)
        
        # Секция AI функций
        self.create_ai_section(control_frame)
        
        # Секция настроек
        self.create_settings_section(control_frame)
    
    def create_load_section(self, parent):
        """Секция загрузки изображений"""
        load_frame = ttk.LabelFrame(parent, text="📁 LOAD IMAGE", padding=10)
        load_frame.pack(fill=tk.X, pady=(0, 15))
        
        btn_load = ttk.Button(load_frame, 
                             text="📁 CHOOSE IMAGE", 
                             command=self.load_image,
                             style="Primary.TButton")
        btn_load.pack(fill=tk.X, pady=5)
        
        btn_camera = ttk.Button(load_frame,
                               text="📷 OPEN CAMERA",
                               command=self.open_camera,
                               style="Primary.TButton")
        btn_camera.pack(fill=tk.X, pady=5)
        
        btn_demo = ttk.Button(load_frame,
                             text="🎨 LOAD DEMO",
                             command=self.load_demo,
                             style="Success.TButton")
        btn_demo.pack(fill=tk.X, pady=5)
    
    def create_background_section(self, parent):
        """Секция выбора фона"""
        bg_frame = ttk.LabelFrame(parent, text="🎨 BACKGROUND", padding=10)
        bg_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Быстрые пресеты фонов
        presets_frame = ttk.Frame(bg_frame)
        presets_frame.pack(fill=tk.X, pady=5)
        
        backgrounds = [
            ("🔵 Blue", "blue"),
            ("🟢 Green", "green"), 
            ("🌈 Gradient", "gradient"),
            ("⚫ Black", "black"),
            ("🌀 Blur", "blur")
        ]
        
        for text, bg_type in backgrounds:
            btn = ttk.Button(presets_frame,
                           text=text,
                           command=lambda bt=bg_type: self.process_image(bt),
                           style="Primary.TButton")
            btn.pack(fill=tk.X, pady=2)
        
        # Кастомный фон
        custom_frame = ttk.Frame(bg_frame)
        custom_frame.pack(fill=tk.X, pady=(10, 0))
        
        btn_custom_bg = ttk.Button(custom_frame,
                                  text="🖼️ CUSTOM BACKGROUND",
                                  command=self.load_custom_background,
                                  style="Warning.TButton")
        btn_custom_bg.pack(fill=tk.X)
    
    def create_ai_section(self, parent):
        """Секция AI функций"""
        ai_frame = ttk.LabelFrame(parent, text="🤖 AI FUNCTIONS", padding=10)
        ai_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Генерация фона по промпту
        prompt_frame = ttk.Frame(ai_frame)
        prompt_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(prompt_frame, text="AI Prompt:").pack(anchor=tk.W)
        
        self.prompt_var = tk.StringVar(value="modern office background")
        prompt_entry = ttk.Entry(prompt_frame, textvariable=self.prompt_var, width=30)
        prompt_entry.pack(fill=tk.X, pady=2)
        
        btn_generate = ttk.Button(prompt_frame,
                                text="🎨 GENERATE BACKGROUND",
                                command=self.generate_background_from_prompt,
                                style="Success.TButton")
        btn_generate.pack(fill=tk.X, pady=2)
        
        # Автоопределение стиля
        btn_auto_style = ttk.Button(ai_frame,
                                  text="👔 AUTO DETECT STYLE",
                                  command=self.auto_detect_style,
                                  style="Primary.TButton")
        btn_auto_style.pack(fill=tk.X, pady=5)
        
        # Пакетная обработка
        btn_batch = ttk.Button(ai_frame,
                             text="📦 BATCH PROCESSING",
                             command=self.batch_processing,
                             style="Warning.TButton")
        btn_batch.pack(fill=tk.X, pady=5)
    
    def create_settings_section(self, parent):
        """Секция настроек"""
        settings_frame = ttk.LabelFrame(parent, text="⚙️ SETTINGS", padding=10)
        settings_frame.pack(fill=tk.X)
        
        # Качество обработки
        quality_frame = ttk.Frame(settings_frame)
        quality_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(quality_frame, text="Quality:").pack(side=tk.LEFT)
        
        self.quality_var = tk.StringVar(value="balanced")
        quality_combo = ttk.Combobox(quality_frame, 
                                    textvariable=self.quality_var,
                                    values=["fast", "balanced", "high"],
                                    state="readonly",
                                    width=12)
        quality_combo.pack(side=tk.RIGHT)
        quality_combo.bind('<<ComboboxSelected>>', self.on_quality_change)
        
        # Автосохранение
        self.auto_save_var = tk.BooleanVar(value=True)
        auto_save_cb = ttk.Checkbutton(settings_frame,
                                      text="Auto-save results",
                                      variable=self.auto_save_var)
        auto_save_cb.pack(anchor=tk.W, pady=2)
        
        # Показ маски
        self.show_mask_var = tk.BooleanVar(value=False)
        show_mask_cb = ttk.Checkbutton(settings_frame,
                                      text="Show segmentation mask",
                                      variable=self.show_mask_var)
        show_mask_cb.pack(anchor=tk.W, pady=2)
    
    def create_image_panel(self, parent):
        """Панель отображения изображений"""
        image_frame = ttk.Frame(parent, style="Custom.TFrame")
        image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Оригинальное изображение
        self.original_frame = ttk.LabelFrame(image_frame, 
                                           text="📸 ORIGINAL IMAGE",
                                           padding=10)
        self.original_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))
        
        self.original_canvas = tk.Canvas(self.original_frame, 
                                        bg="#2d2d30", 
                                        highlightthickness=0)
        self.original_canvas.pack(expand=True, fill=tk.BOTH)
        
        # Результат
        self.result_frame = ttk.LabelFrame(image_frame, 
                                         text="🎯 AI RESULT", 
                                         padding=10)
        self.result_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(5, 0))
        
        self.result_canvas = tk.Canvas(self.result_frame, 
                                      bg="#2d2d30", 
                                      highlightthickness=0)
        self.result_canvas.pack(expand=True, fill=tk.BOTH)
        
        # Добавляем подписи по умолчанию
        self.show_placeholder(self.original_canvas, "Load an image to start")
        self.show_placeholder(self.result_canvas, "AI result will appear here")
    
    def create_status_bar(self):
        """Создание статус-бара"""
        status_frame = ttk.Frame(self.root, style="Custom.TFrame", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        # Статус AI
        self.ai_status_var = tk.StringVar(value="🟢 AI Systems: Initializing...")
        ai_status_label = ttk.Label(status_frame, 
                                   textvariable=self.ai_status_var,
                                   style="Subtitle.TLabel")
        ai_status_label.pack(side=tk.LEFT, padx=10)
        
        # Прогресс-бар
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(status_frame, 
                                      variable=self.progress_var,
                                      mode='determinate')
        progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Статус обработки
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, 
                                textvariable=self.status_var,
                                style="Subtitle.TLabel")
        status_label.pack(side=tk.RIGHT, padx=10)
    
    def animate_loading(self):
        """Анимация загрузки"""
        def update_loading():
            for i in range(101):
                self.progress_var.set(i)
                self.ai_status_var.set(f"🟡 AI Systems: Loading... {i}%")
                time.sleep(0.02)
                self.root.update()
            
            self.ai_status_var.set("🟢 AI Systems: Ready!")
            self.progress_var.set(0)
        
        loading_thread = threading.Thread(target=update_loading, daemon=True)
        loading_thread.start()
    
    def show_placeholder(self, canvas, text):
        """Показ заглушки на canvas"""
        canvas.delete("all")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width > 1 and height > 1:  # Если canvas уже отрисован
            canvas.create_text(width//2, height//2, 
                             text=text, 
                             fill="#666666",
                             font=("Arial", 12),
                             justify=tk.CENTER)
    
    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.status_var.set("📁 Loading image...")
                self.root.update()
                
                self.current_image = file_path
                image = Image.open(file_path)
                
                # Показываем изображение
                self.display_image_on_canvas(image, self.original_canvas)
                self.status_var.set(f"✅ Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Cannot load image: {e}")
                self.status_var.set("❌ Load failed")
    
    def load_custom_background(self):
        """Загрузка кастомного фона"""
        file_path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                self.custom_background = file_path
                self.status_var.set(f"🎨 Custom background loaded")
                
                # Если есть загруженное изображение, сразу обрабатываем
                if self.current_image:
                    self.process_image("custom")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Cannot load background: {e}")
    
    def open_camera(self):
        """Открытие камеры"""
        try:
            from video_processor import SmartVideoProcessor
            
            model_path = 'improved_model.pth' if os.path.exists('improved_model.pth') else None
            processor = SmartVideoProcessor(model_path=model_path)
            
            self.status_var.set("📹 Starting camera...")
            
            # Запуск в отдельном потоке
            def run_camera():
                processor.start_webcam()
            
            camera_thread = threading.Thread(target=run_camera, daemon=True)
            camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open camera: {e}")
    
    def load_demo(self):
        """Загрузка демо примера"""
        try:
            from model import create_demo_image
            import cv2
            
            self.status_var.set("🎨 Creating demo...")
            self.root.update()
            
            # Создаем демо изображение
            demo_img = create_demo_image()
            cv2.imwrite("demo_human.jpg", demo_img)
            
            self.current_image = "demo_human.jpg"
            image = Image.open("demo_human.jpg")
            self.display_image_on_canvas(image, self.original_canvas)
            self.status_var.set("🎨 Demo loaded - Choose background!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Demo failed: {e}")
            self.status_var.set("❌ Demo failed")
    
    def process_image(self, background_type):
        """Обработка изображения"""
        if not hasattr(self, 'current_image'):
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        # Запуск обработки в отдельном потоке
        def process_thread():
            try:
                self.status_var.set("🧠 AI Processing...")
                self.progress_var.set(30)
                self.root.update()
                
                # Обработка AI
                custom_bg = self.custom_background if background_type == "custom" else None
                mask, result = self.segmentator.process_image(
                    self.current_image, background_type, custom_bg
                )
                
                self.progress_var.set(70)
                
                # Конвертация для отображения
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                result_pil = Image.fromarray(result_bgr)
                
                # Показ результата
                self.display_image_on_canvas(result_pil, self.result_canvas)
                
                # Показ маски если включено
                if self.show_mask_var.get():
                    mask_pil = Image.fromarray(mask)
                    self.show_mask_preview(mask_pil)
                
                self.progress_var.set(100)
                self.status_var.set(f"✅ Processed with {background_type} background")
                
                # Автосохранение
                if self.auto_save_var.get():
                    self.auto_save_result(result_bgr, background_type)
                
                self.progress_var.set(0)
                
            except Exception as e:
                self.status_var.set("❌ Processing failed")
                messagebox.showerror("Error", f"AI processing failed: {e}")
                self.progress_var.set(0)
        
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
    
    def generate_background_from_prompt(self):
        """Генерация фона по текстовому промпту"""
        if not hasattr(self, 'current_image'):
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt for background generation")
            return
        
        if self.background_generator is None:
            messagebox.showwarning("Warning", "Background generator is still loading...")
            return
        
        def generate_thread():
            try:
                self.status_var.set("🎨 Generating background...")
                self.progress_var.set(20)
                self.root.update()
                
                # Генерация фона
                background = self.background_generator.generate_from_prompt(prompt, size=(512, 512))
                
                self.progress_var.set(50)
                
                # Конвертация для OpenCV
                background_cv = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)
                
                # Обработка с сгенерированным фоном
                mask, result = self.segmentator.process_image(
                    self.current_image, "custom", custom_background=background_cv
                )
                
                self.progress_var.set(80)
                
                # Показ результата
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                result_pil = Image.fromarray(result_bgr)
                self.display_image_on_canvas(result_pil, self.result_canvas)
                
                self.progress_var.set(100)
                self.status_var.set(f"✅ Generated: {prompt}")
                
                # Сохранение
                if self.auto_save_var.get():
                    cv2.imwrite(f"outputs/generated_{prompt[:20]}.jpg", result_bgr)
                
                self.progress_var.set(0)
                
            except Exception as e:
                self.status_var.set("❌ Generation failed")
                messagebox.showerror("Error", f"Background generation failed: {e}")
                self.progress_var.set(0)
        
        thread = threading.Thread(target=generate_thread, daemon=True)
        thread.start()
    
    def auto_detect_style(self):
        """Автоматическое определение стиля и подбор фона"""
        if not hasattr(self, 'current_image'):
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.style_classifier is None:
            messagebox.showwarning("Warning", "Style classifier is still loading...")
            return
        
        def detect_style_thread():
            try:
                self.status_var.set("👔 Analyzing style...")
                self.progress_var.set(30)
                self.root.update()
                
                # Определение стиля
                recommended_bg, style, confidence = self.style_classifier.get_recommended_background(
                    self.current_image
                )
                
                self.progress_var.set(60)
                
                # Маппинг на типы фонов
                bg_mapping = {
                    "офис": "gradient",
                    "кафе": "green", 
                    "спортзал": "blue",
                    "торжественный": "gradient",
                    "город": "black",
                    "стандартный": "blue"
                }
                
                background_type = bg_mapping.get(recommended_bg, "blue")
                
                # Обработка с подобранным фоном
                mask, result = self.segmentator.process_image(self.current_image, background_type)
                
                self.progress_var.set(90)
                
                # Показ результата
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                result_pil = Image.fromarray(result_bgr)
                self.display_image_on_canvas(result_pil, self.result_canvas)
                
                self.progress_var.set(100)
                self.status_var.set(f"✅ Style: {style} → {recommended_bg} (confidence: {confidence:.2f})")
                
                self.progress_var.set(0)
                
            except Exception as e:
                self.status_var.set("❌ Style detection failed")
                messagebox.showerror("Error", f"Style detection failed: {e}")
                self.progress_var.set(0)
        
        thread = threading.Thread(target=detect_style_thread, daemon=True)
        thread.start()
    
    def batch_processing(self):
        """Пакетная обработка изображений"""
        files = filedialog.askopenfilenames(
            title="Select Multiple Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not files:
            return
        
        def batch_process_thread():
            try:
                total = len(files)
                for i, file_path in enumerate(files):
                    self.status_var.set(f"📦 Processing {i+1}/{total}...")
                    self.progress_var.set((i / total) * 100)
                    self.root.update()
                    
                    # Обработка каждого файла
                    mask, result = self.segmentator.process_image(file_path, "gradient")
                    
                    # Сохранение
                    base_name = os.path.basename(file_path).split('.')[0]
                    result_path = f"outputs/batch_{base_name}_result.jpg"
                    cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                
                self.progress_var.set(100)
                self.status_var.set(f"✅ Batch processed {total} images")
                self.progress_var.set(0)
                
                messagebox.showinfo("Complete", f"Successfully processed {total} images!")
                
            except Exception as e:
                self.status_var.set("❌ Batch processing failed")
                messagebox.showerror("Error", f"Batch processing failed: {e}")
                self.progress_var.set(0)
        
        thread = threading.Thread(target=batch_process_thread, daemon=True)
        thread.start()
    
    def display_image_on_canvas(self, image, canvas):
        """Отображение изображения на canvas с сохранением пропорций"""
        def update_display():
            canvas.delete("all")
            
            # Получаем размеры canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Если canvas еще не отрисован, пробуем снова через 100ms
                self.root.after(100, lambda: self.display_image_on_canvas(image, canvas))
                return
            
            # Ресайз изображения с сохранением пропорций
            img_width, img_height = image.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))
            
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            
            # Центрируем изображение
            x = (canvas_width - new_size[0]) // 2
            y = (canvas_height - new_size[1]) // 2
            
            canvas.create_image(x, y, anchor=tk.NW, image=photo)
            canvas.image = photo  # Сохраняем ссылку
        
        # Запускаем в основном потоке GUI
        self.root.after(0, update_display)
    
    def show_mask_preview(self, mask_image):
        """Показ превью маски"""
        # Создаем отдельное окно для маски
        mask_window = tk.Toplevel(self.root)
        mask_window.title("Segmentation Mask Preview")
        mask_window.geometry("400x500")
        
        mask_canvas = tk.Canvas(mask_window, bg="#2d2d30")
        mask_canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.display_image_on_canvas(mask_image, mask_canvas)
    
    def auto_save_result(self, result_image, background_type):
        """Автосохранение результата"""
        try:
            os.makedirs("outputs", exist_ok=True)
            
            if hasattr(self, 'current_image'):
                base_name = os.path.basename(self.current_image).split('.')[0]
                filename = f"outputs/{base_name}_{background_type}.jpg"
            else:
                filename = f"outputs/result_{background_type}_{int(time.time())}.jpg"
            
            cv2.imwrite(filename, result_image)
            print(f"💾 Auto-saved: {filename}")
            
        except Exception as e:
            print(f"⚠️ Auto-save failed: {e}")
    
    def on_quality_change(self, event):
        """Обработчик изменения качества"""
        quality = self.quality_var.get()
        self.status_var.set(f"⚙️ Quality setting: {quality}")
        
        # Здесь можно добавить логику изменения параметров обработки
        # в зависимости от выбранного качества

# Адаптер для совместимости со старым кодом
class SimpleSegmentationApp(AdvancedSegmentationApp):
    """Совместимость со старым кодом"""
    pass

def main():
    """Тестовый запуск"""
    root = tk.Tk()
    
    # Создаем заглушку сегментатора для теста
    class DummySegmentator:
        def process_image(self, image, background, custom_background=None):
            # Возвращаем тестовое изображение
            test_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            cv2.rectangle(test_img, (50, 50), (150, 250), (100, 100, 100), -1)
            mask = np.zeros((300, 200), dtype=np.uint8)
            cv2.rectangle(mask, (50, 50), (150, 250), 255, -1)
            return mask, test_img
    
    segmentator = DummySegmentator()
    app = AdvancedSegmentationApp(root, segmentator)
    root.mainloop()

if __name__ == "__main__":
    main()