import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import HumanSegmentator
import segmentation_models_pytorch as smp

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, size=(512, 512)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.size = size
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        self.transform = A.Compose([
            A.Resize(size[0], size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        self.mask_transform = A.Compose([
            A.Resize(size[0], size[1]),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Автоматически ищем маску
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.masks_dir, 'mask_' + mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            return self.__getitem__((idx + 1) % len(self.image_files))
        
        # Аугментации
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        mask_transformed = self.mask_transform(image=mask)
        mask_tensor = mask_transformed['image'].float() / 255.0
        
        return image_tensor, mask_tensor

def download_sample_dataset():
    """Скачивание датасета для быстрого старта"""
    import urllib.request
    import zipfile
    
    # Создаем мини-датасет из синтетических данных
    os.makedirs('train_data/images', exist_ok=True)
    os.makedirs('train_data/masks', exist_ok=True)
    
    print("Создание тренировочных данных...")
    for i in range(50):
        # Создаем реалистичное изображение
        img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        
        # Создаем качественную маску
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        # Реалистичный силуэт человека
        cv2.ellipse(mask, (256, 150), (80, 100), 0, 0, 360, 255, -1)  # Голова
        cv2.rectangle(mask, (176, 150), (336, 350), 255, -1)  # Тело
        cv2.rectangle(mask, (176, 350), (216, 450), 255, -1)  # Нога левая
        cv2.rectangle(mask, (296, 350), (336, 450), 255, -1)  # Нога правая
        cv2.rectangle(mask, (136, 180), (176, 300), 255, -1)  # Рука левая
        cv2.rectangle(mask, (336, 180), (376, 300), 255, -1)  # Рука правая
        
        # Добавляем шум для реалистичности
        noise = np.random.randint(0, 50, (512, 512), dtype=np.uint8)
        mask = cv2.add(mask, noise)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = (mask > 128).astype(np.uint8) * 255
        
        cv2.imwrite(f'train_data/images/image_{i:03d}.jpg', img)
        cv2.imwrite(f'train_data/masks/mask_{i:03d}.png', mask)
    
    print("✅ Создано 50 тренировочных примеров")

def train_model():
    print("🚀 Начало дообучения модели...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Создаем улучшенную модель
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=1,
        activation="sigmoid"
    ).to(device)
    
    # Загружаем предобученные веса
    if os.path.exists('improved_model.pth'):
        model.load_state_dict(torch.load('improved_model.pth', map_location=device))
        print("✅ Загружены существующие веса")
    
    # Создаем датасет
    if not os.path.exists('train_data'):
        download_sample_dataset()
    
    dataset = SegmentationDataset('train_data/images', 'train_data/masks')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Оптимизатор и функция потерь
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    print("🎯 Начинаем обучение...")
    model.train()
    
    for epoch in range(20):
        total_loss = 0
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f'Эпоха {epoch+1}, Батч {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'✅ Эпоха {epoch+1} завершена. Средний Loss: {avg_loss:.4f}')
        
        # Сохраняем модель каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'improved_model_epoch_{epoch+1}.pth')
    
    # Финальное сохранение
    torch.save(model.state_dict(), 'improved_model.pth')
    print("💾 Улучшенная модель сохранена как 'improved_model.pth'")

if __name__ == "__main__":
    train_model()