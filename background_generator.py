import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import cv2
import numpy as np
import os

class BackgroundGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.load_model()
    
    def load_model(self):
        """Загрузка модели Stable Diffusion"""
        print("🔄 Загрузка Stable Diffusion...")
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            if self.device == "cuda":
                self.pipe = self.pipe.to("cuda")
                # Оптимизация для скорости
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipe.scheduler.config
                )
                self.pipe.enable_attention_slicing()
            
            print("✅ Stable Diffusion загружен!")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки Stable Diffusion: {e}")
            print("🔄 Используем резервный метод...")
            self.pipe = None
    
    def generate_from_prompt(self, prompt, size=(512, 512), num_inference_steps=20):
        """Генерация фона по текстовому описанию"""
        if self.pipe is None:
            return self._generate_fallback_background(size)
        
        try:
            # Генерация изображения
            image = self.pipe(
                prompt,
                height=size[0],
                width=size[1],
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5
            ).images[0]
            
            return image
            
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            return self._generate_fallback_background(size)
    
    def _generate_fallback_background(self, size):
        """Резервный метод генерации фона"""
        # Создаем градиентный фон
        background = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        for i in range(size[0]):
            # Красивый градиент
            r = int(128 + 127 * np.sin(i * 0.02))
            g = int(128 + 127 * np.sin(i * 0.03 + 2))
            b = int(128 + 127 * np.sin(i * 0.04 + 4))
            background[i, :] = [b, g, r]
        
        return Image.fromarray(background)
    
    def generate_for_style(self, style_type):
        """Генерация фона по типу стиля"""
        style_prompts = {
            "деловой": "professional office background, modern workspace, clean and professional",
            "повседневный": "coffee shop interior, comfortable casual background",
            "спортивный": "modern gym interior, fitness center background",
            "формальный": "elegant event hall, luxury venue background",
            "уличный": "urban city street, modern architecture background",
            "природа": "beautiful natural landscape, park background"
        }
        
        prompt = style_prompts.get(style_type, "beautiful abstract background")
        return self.generate_from_prompt(prompt)

# Интеграция с основным приложением
def add_background_generation_to_gui():
    """Добавляем функционал генерации в GUI"""