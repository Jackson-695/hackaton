import os
import tkinter as tk
from model import HumanSegmentator
from gui import SimpleSegmentationApp

def main():
    print("🚀 AI Human Segmentation - PRO")
    
    # Проверка окружения
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    # Загрузка модели
    model_path = 'improved_model.pth' if os.path.exists('improved_model.pth') else None
    segmentator = HumanSegmentator(model_path)
    
    # Запуск GUI
    root = tk.Tk()
    app = SimpleSegmentationApp(root, segmentator)
    root.mainloop()

if __name__ == "__main__":
    main()