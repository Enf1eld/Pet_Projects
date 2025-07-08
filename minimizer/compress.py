from PIL import Image
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

def resize_and_compress(input_path, output_path, target_size_kb=250, max_width=1920, max_height=1080, quality=85):
    """
    Изменяет размер и сжимает изображение
    :param input_path: Путь к исходному файлу
    :param output_path: Путь для сохранения
    :param target_size_kb: Целевой размер в КБ
    :param max_width: Максимальная ширина
    :param max_height: Максимальная высота
    :param quality: Начальное качество
    """
    try:
        with Image.open(input_path) as img:
            # Конвертируем в RGB если нужно
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Получаем текущие размеры
            width, height = img.size
            
            # Масштабируем с сохранением пропорций
            if width > max_width or height > max_height:
                ratio = min(max_width/width, max_height/height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Сохраняем с оптимизацией
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
            # Дополнительно сжимаем если нужно
            while os.path.getsize(output_path) > target_size_kb * 1024 and quality > 10:
                quality -= 5
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                
        return True, input_path, output_path
    except Exception as e:
        return False, input_path, str(e)

def process_images(input_dir='input', output_dir='output', max_workers=4):
    """
    Обрабатывает все JPEG изображения в папке
    """
    if not os.path.exists(input_dir):
        print(f"Ошибка: Папка {input_dir} не существует!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print("Нет изображений для обработки в папке input")
        return
    
    print(f"Найдено {len(files)} изображений для обработки...")
    
    lock = threading.Lock()
    processed = 0
    
    def callback(future):
        nonlocal processed
        success, input_path, result = future.result()
        
        with lock:
            processed += 1
            if success:
                original_size = os.path.getsize(input_path) / 1024
                compressed_size = os.path.getsize(result) / 1024
                with Image.open(result) as img:
                    new_width, new_height = img.size
                print(f"[{processed}/{len(files)}] {os.path.basename(input_path)}: "
                      f"{original_size:.1f}KB -> {compressed_size:.1f}KB | "
                      f"Размер: {new_width}x{new_height}")
            else:
                print(f"[{processed}/{len(files)}] Ошибка обработки {os.path.basename(input_path)}: {result}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            future = executor.submit(resize_and_compress, input_path, output_path)
            future.add_done_callback(callback)
            futures.append(future)
        
        for future in futures:
            future.result()
    
    print("\nОбработка завершена!")

if __name__ == "__main__":
    # Параметры по умолчанию
    input_folder = 'input'
    output_folder = 'output'
    threads = 4
    
    # Можно указать: python script.py input output потоки целевой_размер_КБ ширина высота
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    if len(sys.argv) > 3:
        threads = int(sys.argv[3])
    
    # Стандартные размеры (можно изменить)
    TARGET_SIZE_KB = 250
    MAX_WIDTH = 1920  # Full HD width
    MAX_HEIGHT = 1080  # Full HD height
    
    if len(sys.argv) > 4:
        TARGET_SIZE_KB = int(sys.argv[4])
    if len(sys.argv) > 5:
        MAX_WIDTH = int(sys.argv[5])
    if len(sys.argv) > 6:
        MAX_HEIGHT = int(sys.argv[6])
    
    print(f"Настройки: Размер {MAX_WIDTH}x{MAX_HEIGHT}, Целевой вес {TARGET_SIZE_KB}KB")
    process_images(input_folder, output_folder, threads)