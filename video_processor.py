import cv2
import torch
import numpy as np
import time
from model import HumanSegmentator

class VideoProcessor:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎥 Инициализация видео-процессора на {self.device}")
        
        # Загружаем улучшенную модель
        self.segmentator = HumanSegmentator()
        if model_path and os.path.exists(model_path):
            self.segmentator.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✅ Загружена улучшенная модель")
        
        # Оптимизации для скорости
        self.optimize_for_speed()
        
        self.cap = None
        self.is_processing = False
        self.current_background = "blue"
        self.custom_background = None
        
    def optimize_for_speed(self):
        """Оптимизация модели для максимальной скорости"""
        # TorchScript компиляция
        try:
            self.segmentator.model = torch.jit.script(self.segmentator.model)
            print("✅ Модель скомпилирована с TorchScript")
        except:
            print("⚠️ TorchScript компиляция не удалась, используем стандартный режим")
        
        # Половинная точность для GPU
        if self.device.type == 'cuda':
            self.segmentator.model = self.segmentator.model.half()
            print("✅ Используется половинная точность (FP16)")
        
        # Включение режима оценки
        self.segmentator.model.eval()
        
    def start_webcam(self, camera_id=0, output_size=(1280, 720)):
        """Запуск обработки с веб-камеры"""
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Не удалось открыть камеру {camera_id}")
            return False
        
        # Настройки камеры для лучшей производительности
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, output_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, output_size[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_processing = True
        print("🎥 Веб-камера запущена. Нажми 'q' для выхода, 'b' для смены фона")
        
        fps_counter = 0
        fps_time = time.time()
        
        while self.is_processing:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Измерение FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
                print(f"📊 FPS: {fps}")
            
            # Обработка кадра
            processed_frame = self.process_frame(frame)
            
            # Отображение FPS на кадре
            cv2.putText(processed_frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"Background: {self.current_background}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('AI Human Segmentation - Video', processed_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                self.cycle_background()
            elif key == ord('c'):
                self.use_custom_background()
        
        self.stop_webcam()
    
    def process_frame(self, frame):
        """Обработка одного кадра с оптимизацией"""
        try:
            with torch.no_grad():  # Отключаем градиенты для скорости
                # Быстрая обработка
                mask, result = self.segmentator.process_image(frame, self.current_background)
                return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"❌ Ошибка обработки кадра: {e}")
            return frame
    
    def cycle_background(self):
        """Смена фона"""
        backgrounds = ["blue", "green", "gradient", "black", "blur"]
        current_idx = backgrounds.index(self.current_background) if self.current_background in backgrounds else 0
        self.current_background = backgrounds[(current_idx + 1) % len(backgrounds)]
        print(f"🎨 Сменен фон на: {self.current_background}")
    
    def use_custom_background(self):
        """Использование кастомного фона"""
        # Можно добавить загрузку своего фона
        print("🖼️ Функция кастомного фона (реализовать загрузку изображения)")
    
    def stop_webcam(self):
        """Остановка веб-камеры"""
        self.is_processing = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 Видео-обработка остановлена")

def main():
    processor = VideoProcessor(model_path='improved_model.pth')
    processor.start_webcam()

if __name__ == "__main__":
    main()