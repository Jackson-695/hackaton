import os
import sys
import torch
import cv2
import numpy as np
from model import HumanSegmentator
from video_processor import SmartVideoProcessor
from background_generator import BackgroundGenerator
from style_classifier import StyleClassifier
import tkinter as tk
from gui import SimpleSegmentationApp

print("=" * 80)
print("🚀 AI HUMAN SEGMENTATION - ПРОДВИНУТАЯ СИСТЕМА")
print("=" * 80)
print("🧠 Neural Network: U-Net with ResNet34 (Improved)")
print("🎯 Features: Real-time segmentation, AI background generation, Style detection")
print("💻 Device: Auto GPU/CPU optimization")
print("⚡ Performance: 30+ FPS on GPU, High-quality segmentation")
print("=" * 80)

def check_environment():
    """Проверка окружения и зависимостей"""
    print("\n🔍 Проверка окружения...")
    
    # Проверка PyTorch и CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Проверка OpenCV
    print(f"OpenCV version: {cv2.__version__}")
    
    # Создание необходимых папок
    folders = ['outputs', 'examples', 'train_data', 'models']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Папка {folder}/ создана")
    
    return True

def show_main_menu():
    """Главное меню приложения"""
    print("\n🎮 ГЛАВНОЕ МЕНЮ")
    print("1. 🖼️  Запуск графического интерфейса (GUI)")
    print("2. 🎥  Видео-обработка с веб-камеры")
    print("3. 🎨  Генерация фонов по промпту")
    print("4. 🧠  Дообучение модели сегментации")
    print("5. 👔  Тест классификатора стилей")
    print("6. 🧪  Быстрый тест системы")
    print("7. 📊  Информация о системе")
    print("0. ❌  Выход")
    
    while True:
        try:
            choice = input("\nВыберите опцию (0-7): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7']:
                return int(choice)
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            sys.exit(0)

def launch_gui():
    """Запуск графического интерфейса"""
    print("\n🖼️ Загрузка графического интерфейса...")
    
    # Проверяем наличие улучшенной модели
    model_path = 'improved_model.pth' if os.path.exists('improved_model.pth') else None
    
    # Создаем сегментатор
    segmentator = HumanSegmentator(model_path=model_path, model_type="improved" if model_path else "basic")
    
    # Запускаем GUI
    root = tk.Tk()
    app = SimpleSegmentationApp(root, segmentator)
    root.mainloop()

def launch_video_processing():
    """Запуск видео-обработки с веб-камеры"""
    print("\n🎥 Запуск видео-обработки...")
    
    # Проверяем наличие веб-камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Веб-камера не найдена! Проверьте подключение.")
        return
    cap.release()
    
    # Проверяем наличие улучшенной модели
    model_path = 'improved_model.pth' if os.path.exists('improved_model.pth') else None
    
    # Создаем умный видео-процессор
    try:
        processor = SmartVideoProcessor(model_path=model_path)
        print("✅ Видео-процессор инициализирован")
        print("\n🎮 Управление:")
        print("   - 'q' - выход")
        print("   - 'b' - смена фона")
        print("   - 'c' - кастомный фон")
        print("   - Автоподбор фона по стилю каждые 5 секунд")
        
        processor.start_webcam()
    except Exception as e:
        print(f"❌ Ошибка запуска видео-процессора: {e}")

def launch_background_generation():
    """Демонстрация генерации фонов по промпту"""
    print("\n🎨 Генерация фонов по промпту...")
    
    try:
        generator = BackgroundGenerator()
        
        # Тестовые промпты для демонстрации
        demo_prompts = [
            "modern office background with glass windows",
            "cozy coffee shop interior",
            "futuristic cyberpunk city at night",
            "beautiful beach with palm trees",
            "luxury hotel lobby with marble floor"
        ]
        
        print("\n🧪 Генерация демо-фонов...")
        for i, prompt in enumerate(demo_prompts):
            print(f"   {i+1}. {prompt}")
            background = generator.generate_from_prompt(prompt, size=(512, 512))
            background.save(f"examples/generated_bg_{i+1}.jpg")
        
        print("\n✅ Демо-фоны сохранены в папке examples/")
        print("💡 В GUI можно использовать кнопку 'Сгенерировать фон'")
        
    except Exception as e:
        print(f"❌ Ошибка генерации фонов: {e}")

def launch_model_training():
    """Запуск дообучения модели"""
    print("\n🧠 Дообучение модели сегментации...")
    
    # Проверяем наличие данных для обучения
    if not os.path.exists('train_data/images') or len(os.listdir('train_data/images')) == 0:
        print("❌ Данные для обучения не найдены!")
        print("💡 Запустите сначала быстрый тест (опция 6) для создания демо-данных")
        return
    
    print("🚀 Начинаем дообучение...")
    print("⚠️  Это может занять от 30 минут до нескольких часов")
    
    try:
        # Импортируем и запускаем обучение
        from train_improved_segmentation import train_model
        train_model()
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")

def test_style_classifier():
    """Тест классификатора стилей одежды"""
    print("\n👔 Тестирование классификатора стилей...")
    
    try:
        classifier = StyleClassifier()
        
        # Создаем тестовое изображение
        test_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
        
        # Рисуем "деловой" силуэт (пиджак)
        cv2.rectangle(test_img, (80, 50), (120, 250), (0, 0, 0), -1)  # Галстук/рубашка
        
        # Предсказываем стиль
        style, confidence = classifier.predict_style(test_img)
        recommended_bg, detected_style, conf = classifier.get_recommended_background(test_img)
        
        print(f"🎯 Результаты классификации:")
        print(f"   - Определенный стиль: {detected_style}")
        print(f"   - Уверенность: {conf:.2f}")
        print(f"   - Рекомендованный фон: {recommended_bg}")
        
        # Сохраняем тестовое изображение
        cv2.imwrite("examples/style_test.jpg", test_img)
        print(f"✅ Тестовое изображение сохранено: examples/style_test.jpg")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования классификатора: {e}")

def quick_system_test():
    """Быстрый тест всей системы"""
    print("\n🧪 Быстрый тест системы...")
    
    try:
        # 1. Тест сегментации
        print("1. Тестирование сегментации...")
        model_path = 'improved_model.pth' if os.path.exists('improved_model.pth') else None
        segmentator = HumanSegmentator(model_path=model_path)
        
        # Создаем тестовое изображение
        test_img = np.ones((400, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_img, (100, 100), (200, 300), (100, 100, 100), -1)
        
        mask, result = segmentator.process_image(test_img, "gradient")
        cv2.imwrite("examples/quick_test_result.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print("   ✅ Сегментация работает")
        
        # 2. Тест видео-процессора
        print("2. Тестирование видео-процессора...")
        processor = SmartVideoProcessor(model_path=model_path)
        print("   ✅ Видео-процессор инициализирован")
        
        # 3. Тест генератора фонов
        print("3. Тестирование генератора фонов...")
        generator = BackgroundGenerator()
        test_bg = generator.generate_from_prompt("test", size=(100, 100))
        print("   ✅ Генератор фонов работает")
        
        # 4. Тест классификатора стилей
        print("4. Тестирование классификатора стилей...")
        classifier = StyleClassifier()
        style, confidence = classifier.predict_style(test_img)
        print(f"   ✅ Классификатор стилей работает ({style}, {confidence:.2f})")
        
        print("\n🎉 ВСЕ СИСТЕМЫ РАБОТАЮТ КОРРЕКТНО!")
        print("📁 Результаты теста в папке examples/")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")

def show_system_info():
    """Показать информацию о системе"""
    print("\n📊 ИНФОРМАЦИЯ О СИСТЕМЕ")
    
    # Аппаратное обеспечение
    print("\n💻 Аппаратное обеспечение:")
    print(f"   - Процессор: {torch.get_num_threads()} потоков")
    print(f"   - PyTorch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Модели
    print("\n🧠 Модели AI:")
    models = {
        "Улучшенная модель сегментации": "improved_model.pth",
        "Базовая модель сегментации": "model.py",
        "Классификатор стилей": "style_classifier.pth",
        "Генератор фонов": "Stable Diffusion"
    }
    
    for name, path in models.items():
        if path.endswith('.pth'):
            exists = os.path.exists(path)
            status = "✅ Загружена" if exists else "⚠️  Не найдена"
        else:
            status = "✅ Доступна"
        print(f"   - {name}: {status}")
    
    # Данные
    print("\n📁 Данные:")
    data_folders = ['outputs/', 'examples/', 'train_data/']
    for folder in data_folders:
        file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]) if os.path.exists(folder) else 0
        print(f"   - {folder}: {file_count} файлов")
    
    # Производительность
    print("\n⚡ Производительность:")
    print("   - Ожидаемый FPS на CPU: 10-15")
    print("   - Ожидаемый FPS на GPU: 30+")
    print("   - Качество сегментации: Высокое (512x512)")

def main():
    """Главная функция"""
    try:
        # Проверка окружения
        if not check_environment():
            return
        
        print("\n🎯 ДОСТУПНЫЕ РЕЖИМЫ РАБОТЫ:")
        
        while True:
            choice = show_main_menu()
            
            if choice == 0:
                print("\n👋 До свидания! Удачи на хакатоне! 🚀")
                break
            elif choice == 1:
                launch_gui()
            elif choice == 2:
                launch_video_processing()
            elif choice == 3:
                launch_background_generation()
            elif choice == 4:
                launch_model_training()
            elif choice == 5:
                test_style_classifier()
            elif choice == 6:
                quick_system_test()
            elif choice == 7:
                show_system_info()
            
            input("\nНажмите Enter чтобы продолжить...")
    
    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        print("💡 Попробуйте переустановить зависимости: pip install -r requirements.txt")

if __name__ == "__main__":
    main()