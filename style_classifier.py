import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

class StyleClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model()
        self.transform = self._get_transform()
        
        # Стили одежды
        self.style_labels = [
            "деловой", "повседневный", "спортивный", "формальный", "уличный"
        ]
        
        # Соответствие стилей фонам
        self.style_to_background = {
            "деловой": "офис",
            "повседневный": "кафе", 
            "спортивный": "спортзал",
            "формальный": "торжественный",
            "уличный": "город"
        }
    
    def _create_model(self):
        """Создание модели классификации стиля"""
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(self.style_labels))
        
        # Загружаем предобученные веса
        try:
            model.load_state_dict(torch.load('style_classifier.pth', map_location=self.device))
            print("✅ Загружен классификатор стилей")
        except:
            print("⚠️ Используется базовый классификатор стилей")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        """Трансформации для изображения"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_style(self, image):
        """Предсказание стиля одежды на изображении"""
        if isinstance(image, str):
            # Загрузка из файла
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Конвертация из OpenCV
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Препроцессинг
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        predicted_style = self.style_labels[predicted_idx]
        
        return predicted_style, confidence
    
    def get_recommended_background(self, image):
        """Рекомендация фона на основе стиля"""
        style, confidence = self.predict_style(image)
        
        if confidence < 0.5:
            return "стандартный", style, confidence
        
        recommended_bg = self.style_to_background.get(style, "стандартный")
        return recommended_bg, style, confidence

# Интеграция с видео-процессором
class SmartVideoProcessor(VideoProcessor):
    def __init__(self, model_path=None):
        super().__init__(model_path)
        self.style_classifier = StyleClassifier()
        self.last_style_check = 0
        self.style_check_interval = 5  # секунд
    
    def process_frame_smart(self, frame):
        """Умная обработка кадра с авто-подбором фона"""
        current_time = time.time()
        
        # Периодически проверяем стиль
        if current_time - self.last_style_check > self.style_check_interval:
            recommended_bg, style, confidence = self.style_classifier.get_recommended_background(frame)
            
            if confidence > 0.6:  # Достаточно высокая уверенность
                self.current_background = self.map_background_type(recommended_bg)
                print(f"🎨 Автоподбор: {style} -> {recommended_bg} (уверенность: {confidence:.2f})")
            
            self.last_style_check = current_time
        
        # Обычная обработка
        return self.process_frame(frame)
    
    def map_background_type(self, background_name):
        """Сопоставление названия фона с типом"""
        background_mapping = {
            "офис": "gradient",
            "кафе": "green", 
            "спортзал": "blue",
            "торжественный": "gradient",
            "город": "black",
            "стандартный": "blue"
        }
        return background_mapping.get(background_name, "blue")

# Быстрое обучение классификатора
def train_quick_classifier():
    """Быстрое обучение классификатора стилей"""
# минимальная модель для демо
    print("🔧 Создание базового классификатора стилей...")
    
    # Сохраняем базовую модель
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 5)
    torch.save(model.state_dict(), 'style_classifier.pth')
    print("✅ Базовый классификатор создан")