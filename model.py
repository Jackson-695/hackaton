import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import os

class HumanSegmentator:
    """Улучшенный сегментатор с поддержкой разных моделей"""
    
    def __init__(self, model_path=None, model_type="improved"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Инициализация сегментатора на {self.device}")
        
        # Выбор размера изображения в зависимости от модели
        if model_type == "improved" and model_path:
            self.image_size = (512, 512)  # Высокое качество
            self.model = self._load_improved_model(model_path)
        else:
            self.image_size = (256, 256)  # Баланс скорость/качество
            self.model = self._load_basic_model()
        
        self.model.to(self.device)
        self.model.eval()
        
        # Оптимизации для скорости
        self._optimize_model()
    
    def _load_basic_model(self):
        """Базовая модель с предобученными весами"""
        print("✅ Загрузка базовой модели (ResNet18)")
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            classes=1,
            activation="sigmoid"
        )
        return model
    
    def _load_improved_model(self, model_path):
        """Загрузка улучшенной модели"""
        if not os.path.exists(model_path):
            print("⚠️ Улучшенная модель не найдена, используем базовую")
            return self._load_basic_model()
        
        print("✅ Загрузка улучшенной модели (ResNet34)")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # Без импейнет весов, т.к. свои
            classes=1,
            activation="sigmoid"
        )
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("🎯 Улучшенная модель успешно загружена")
        except Exception as e:
            print(f"❌ Ошибка загрузки улучшенной модели: {e}")
            return self._load_basic_model()
        
        return model
    
    def _optimize_model(self):
        """Оптимизация модели для скорости"""
        # TorchScript компиляция для ускорения
        try:
            example_input = torch.randn(1, 3, *self.image_size).to(self.device)
            self.model = torch.jit.trace(self.model, example_input)
            print("⚡ Модель оптимизирована с TorchScript")
        except Exception as e:
            print(f"⚠️ TorchScript оптимизация не удалась: {e}")
        
        # Половинная точность для GPU
        if self.device.type == 'cuda':
            self.model = self.model.half()
            print("🔧 Используется половинная точность (FP16)")
    
    def process_image(self, image_path_or_array, background="blue", custom_background=None):
        """
        Основной метод обработки изображения
        
        Args:
            image_path_or_array: путь к изображению или numpy array
            background: тип фона ("blue", "green", "gradient", "black", "blur")
            custom_background: кастомное изображение фона (опционально)
        
        Returns:
            mask: бинарная маска сегментации
            result: изображение с замененным фоном
        """
        # Загрузка и подготовка изображения
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path_or_array}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_array
        
        original_size = image.shape[:2]
        
        # Сегментация
        processed_image = self._preprocess_image(image)
        mask = self._predict_mask(processed_image)
        final_mask = self._postprocess_mask(mask, original_size)
        
        # Замена фона
        result = self._apply_background(image, final_mask, background, custom_background)
        
        return final_mask, result
    
    def _preprocess_image(self, image):
        """Препроцессинг изображения для нейросети"""
        image_resized = cv2.resize(image, self.image_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Конвертация в тензор
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0)
        
        # Нормализация ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor.float() / 255.0 - mean) / std
        
        # Конвертация в половинную точность если на GPU
        if self.device.type == 'cuda':
            image_tensor = image_tensor.half()
        else:
            image_tensor = image_tensor.float()
            
        return image_tensor.to(self.device)
    
    def _predict_mask(self, image_tensor):
        """Предсказание маски нейросетью"""
        with torch.no_grad():
            output = self.model(image_tensor)
            mask = torch.sigmoid(output)  # Активация сигмоидой
            mask = mask.squeeze().cpu().numpy()
        return mask
    
    def _postprocess_mask(self, mask, original_size):
        """Постобработка и улучшение маски"""
        # Ресайз до оригинального размера
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]))
        
        # Бинаризация
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # Морфологические операции для улучшения качества
        kernel = np.ones((5, 5), np.uint8)
        
        # Закрытие (заполнение маленьких дырок)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Открытие (удаление маленьких шумов)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Размытие границ для плавности
        binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
        binary_mask = (binary_mask > 128).astype(np.uint8) * 255
        
        return binary_mask
    
    def _apply_background(self, image, mask, background_type, custom_background=None):
        """Применение нового фона к изображению"""
        if custom_background is not None:
            # Используем кастомный фон
            if isinstance(custom_background, str):
                background = cv2.imread(custom_background)
                background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            else:
                background = custom_background
            
            # Ресайз фона под размер изображения
            if background.shape[:2] != image.shape[:2]:
                background = cv2.resize(background, (image.shape[1], image.shape[0]))
        else:
            # Генерация фона по типу
            background = self._generate_background(image.shape, background_type)
        
        # Создаем плавную маску для краев
        smooth_mask = mask.astype(np.float32) / 255.0
        smooth_mask = cv2.GaussianBlur(smooth_mask, (7, 7), 0)
        
        # Применяем маску с плавными краями
        if len(smooth_mask.shape) == 2:
            smooth_mask = np.stack([smooth_mask] * 3, axis=2)
        
        result = image * smooth_mask + background * (1 - smooth_mask)
        return result.astype(np.uint8)
    
    def _generate_background(self, image_shape, background_type):
        """Генерация фона по типу"""
        h, w = image_shape[:2]
        
        if background_type == "blue":
            background = np.full((h, w, 3), [255, 0, 0], dtype=np.uint8)  # Синий в RGB
        elif background_type == "green":
            background = np.full((h, w, 3), [0, 255, 0], dtype=np.uint8)  # Зеленый
        elif background_type == "black":
            background = np.zeros((h, w, 3), dtype=np.uint8)  # Черный
        elif background_type == "blur":
            # Размытая версия оригинального изображения как фон
            background = cv2.GaussianBlur(np.zeros((h, w, 3)), (51, 51), 0)
        elif background_type == "gradient":
            background = self._create_gradient_background((h, w))
        else:
            # Градиент по умолчанию
            background = self._create_gradient_background((h, w))
        
        return background
    
    def _create_gradient_background(self, size):
        """Создание красивого градиентного фона"""
        h, w = size
        background = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(h):
            # Синий -> Фиолетовый градиент
            progress = i / h
            blue = int(255 * (1 - progress))
            red = int(255 * progress)
            green = int(128 * abs(np.sin(progress * 3.14)))
            background[i, :] = [blue, green, red]
        
        return background
    
    def batch_process(self, image_paths, background="blue", output_dir="outputs"):
        """Пакетная обработка изображений"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                mask, result = self.process_image(image_path, background)
                
                # Сохранение результатов
                base_name = os.path.basename(image_path).split('.')[0]
                result_path = os.path.join(output_dir, f"{base_name}_result.jpg")
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                
                cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                cv2.imwrite(mask_path, mask)
                
                results.append({
                    'input': image_path,
                    'result': result_path,
                    'mask': mask_path
                })
                
                print(f"✅ Обработано {i+1}/{len(image_paths)}: {base_name}")
                
            except Exception as e:
                print(f"❌ Ошибка обработки {image_path}: {e}")
        
        return results

# Функция для быстрого тестирования
def test_model():
    """Быстрый тест модели"""
    print("🧪 Тестирование модели...")
    
    # Создаем тестовое изображение
    test_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_img, (50, 50), (150, 250), (100, 100, 100), -1)
    
    # Обработка
    segmentator = HumanSegmentator()
    mask, result = segmentator.process_image(test_img, "gradient")
    
    # Сохранение результатов
    cv2.imwrite("test_input.jpg", test_img)
    cv2.imwrite("test_mask.jpg", mask)
    cv2.imwrite("test_result.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    print("✅ Тест завершен! Проверь файлы test_*.jpg")

if __name__ == "__main__":
    test_model()