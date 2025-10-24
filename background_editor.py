import cv2
import numpy as np

class BackgroundEditor:
    def __init__(self):
        print("🎨 Инициализация редактора фонов")
        
    def apply_effects(self, background, effects):
        """Применяет эффекты к фону"""
        result = background.copy()
        
        # Блюр
        if 'blur' in effects and effects['blur'] > 0:
            blur_val = int(effects['blur'] * 10)
            if blur_val % 2 == 0: blur_val += 1
            result = cv2.GaussianBlur(result, (blur_val, blur_val), 0)
        
        # Сдвиг оттенка
        if 'hue' in effects and effects['hue'] != 0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            hsv[:,:,0] = (hsv[:,:,0] + effects['hue']) % 180
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Яркость
        if 'brightness' in effects and effects['brightness'] != 1.0:
            result = cv2.convertScaleAbs(result, alpha=effects['brightness'], beta=0)
        
        # Контраст
        if 'contrast' in effects and effects['contrast'] != 1.0:
            result = cv2.convertScaleAbs(result, alpha=effects['contrast'], beta=128*(1-effects['contrast']))
        
        # Корпоративные пресеты
        if 'preset' in effects:
            result = self.apply_preset(result, effects['preset'])
            
        return result
    
    def apply_preset(self, image, preset_name):
        """Применяет готовые пресеты"""
        if preset_name == "dion_blue":
            # Синий корпоративный стиль DION
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,0] = (hsv[:,:,0] - 10) % 180  # Сдвиг в синюю область
            hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.2, 0, 255)  # Увеличиваем насыщенность
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif preset_name == "dion_dark":
            # Темный профессиональный стиль
            img = cv2.convertScaleAbs(image, alpha=0.7, beta=0)
            return cv2.GaussianBlur(img, (3, 3), 0)
            
        elif preset_name == "dion_modern":
            # Современный стиль с теплым оттенком
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,0] = (hsv[:,:,0] + 15) % 180  # Теплый сдвиг
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        return image