import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import os
from background_editor import BackgroundEditor

class HumanSegmentator:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Инициализация сегментатора на {self.device}")
        
        # Загрузка модели
        if model_path and os.path.exists(model_path):
            self.model = self.load_improved_model(model_path)
            self.image_size = (512, 512)
        else:
            self.model = self.load_basic_model() 
            self.image_size = (256, 256)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Инициализация редактора фонов
        self.background_editor = BackgroundEditor()
    
    def load_basic_model(self):
        print("✅ Загрузка базовой модели")
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet", 
            classes=1,
            activation="sigmoid"
        )
        return model
    
    def load_improved_model(self, model_path):
        print("✅ Загрузка улучшенной модели")
        model = smp.Unet(
            encoder_name="resnet34", 
            encoder_weights=None,
            classes=1,
            activation="sigmoid"
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def process_image(self, image_path_or_array, background_type="blue", custom_background=None, effects=None):
        """Основной метод обработки с поддержкой эффектов"""
        # Загрузка изображения
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_array
        
        original_size = image.shape[:2]
        
        # Сегментация
        processed = self.preprocess_image(image)
        mask = self.predict_mask(processed)
        final_mask = self.postprocess_mask(mask, original_size)
        
        # Применение фона с эффектами
        result = self.apply_background_with_effects(image, final_mask, background_type, custom_background, effects)
        
        return final_mask, result
    
    def preprocess_image(self, image):
        """Подготовка изображения для нейросети"""
        image_resized = cv2.resize(image, self.image_size)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0)
        
        # Нормализация
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor.float() / 255.0 - mean) / std
        
        return image_tensor.to(self.device)
    
    def predict_mask(self, image_tensor):
        """Предсказание маски"""
        with torch.no_grad():
            output = self.model(image_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
        return mask
    
    def postprocess_mask(self, mask, original_size):
        """Постобработка маски"""
        mask_resized = cv2.resize(mask, (original_size[1], original_size[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
        
        # Улучшение качества маски
        kernel = np.ones((5,5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.GaussianBlur(binary_mask, (5,5), 0)
        
        return (binary_mask > 128).astype(np.uint8) * 255
    
    def apply_background_with_effects(self, image, mask, background_type, custom_background, effects):
        """Применение фона с эффектами"""
        # Получаем фон
        if custom_background is not None:
            if isinstance(custom_background, str):
                background = cv2.imread(custom_background)
                background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            else:
                background = custom_background
            
            if background.shape[:2] != image.shape[:2]:
                background = cv2.resize(background, (image.shape[1], image.shape[0]))
        else:
            background = self.generate_background(image.shape, background_type)
        
        # Применяем эффекты к фону
        if effects:
            background = self.background_editor.apply_effects(background, effects)
        
        # Плавное смешивание
        smooth_mask = mask.astype(np.float32) / 255.0
        smooth_mask = cv2.GaussianBlur(smooth_mask, (7,7), 0)
        smooth_mask = np.stack([smooth_mask]*3, axis=2)
        
        result = image * smooth_mask + background * (1 - smooth_mask)
        return result.astype(np.uint8)
    
    def generate_background(self, image_shape, background_type):
        """Генерация базового фона"""
        h, w = image_shape[:2]
        
        if background_type == "blue":
            return np.full((h,w,3), [255,0,0], dtype=np.uint8)
        elif background_type == "green":
            return np.full((h,w,3), [0,255,0], dtype=np.uint8)
        elif background_type == "black":
            return np.zeros((h,w,3), dtype=np.uint8)
        elif background_type == "blur":
            return cv2.GaussianBlur(np.zeros((h,w,3)), (51,51), 0)
        elif background_type == "gradient":
            background = np.zeros((h,w,3), dtype=np.uint8)
            for i in range(h):
                progress = i / h
                blue = int(255 * (1-progress))
                red = int(255 * progress)
                background[i,:] = [blue, 128, red]
            return background
        else:
            return np.full((h,w,3), [255,0,0], dtype=np.uint8)  # Синий по умолчанию