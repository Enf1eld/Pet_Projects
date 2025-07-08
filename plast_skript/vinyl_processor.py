# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import glob
import math

def increase_contrast(image, alpha=1.0):
    """Увеличивает контраст изображения"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def detect_circles_hough(image, dp=1.2, min_dist=100, param1=100, param2=30, min_radius=10, max_radius=0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = increase_contrast(gray, alpha=1.4)
    gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Попробуем несколько наборов параметров для лучшего обнаружения
    param_sets = [
        {'dp': dp, 'param1': param1, 'param2': param2},  # Оригинальные параметры
        {'dp': 1.5, 'param1': 80, 'param2': 25},        # Более чувствительные
        {'dp': 1.0, 'param1': 120, 'param2': 35},       # Более строгие
        {'dp': 2.0, 'param1': 60, 'param2': 20},        # Очень чувствительные
    ]
    
    for params in param_sets:
        circles = cv2.HoughCircles(gray_blurred,
                                   cv2.HOUGH_GRADIENT,
                                   dp=params['dp'],
                                   minDist=min_dist,
                                   param1=params['param1'],
                                   param2=params['param2'],
                                   minRadius=min_radius,
                                   maxRadius=max_radius)
        if circles is not None:
            print(f"✓ Круги найдены с параметрами: dp={params['dp']}, param1={params['param1']}, param2={params['param2']}")
            return circles
    
    print("⚠ Не удалось найти круги ни с одним набором параметров")
    return None

def extract_circle_region(image, circle, output_size=1024):
    """Извлекает область вокруг круга и растягивает до нужного размера"""
    x, y, r = circle
    h, w = image.shape[:2]
    
    # Определяем область для извлечения - квадрат вокруг круга с небольшим запасом
    margin = int(r * 0.1)  # 10% запаса от радиуса
    crop_size = int(r * 2 + margin * 2)
    
    # Координаты левого верхнего угла области
    x1 = max(0, x - r - margin)
    y1 = max(0, y - r - margin)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    
    # Корректируем если вышли за границы
    if x2 - x1 < crop_size:
        x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size:
        y1 = max(0, y2 - crop_size)
    
    # Извлекаем область
    cropped = image[y1:y2, x1:x2]
    
    # Растягиваем до нужного размера
    resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    
    return resized

def extract_circle_crop_improved(image, size=1024):
    """Улучшенная функция извлечения круга с дополнительной предобработкой"""
    h, w = image.shape[:2]
    
    # Адаптивные параметры в зависимости от размера изображения
    min_dist = max(h, w) // 10  # Минимальное расстояние между кругами
    min_radius = min(h, w) // 20  # Минимальный радиус
    max_radius = min(h, w) // 2   # Максимальный радиус
    
    print(f"Поиск кругов с параметрами: min_dist={min_dist}, min_radius={min_radius}, max_radius={max_radius}")
    
    circles = detect_circles_hough(image,
                                  dp=1.2,
                                  min_dist=min_dist,
                                  param1=100,
                                  param2=30,
                                  min_radius=min_radius,
                                  max_radius=max_radius)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"Найдено кругов: {len(circles)}")
        
        # Выбираем наибольший круг
        largest_circle = max(circles, key=lambda c: c[2])
        print(f"Выбран круг: центр=({largest_circle[0]}, {largest_circle[1]}), радиус={largest_circle[2]}")
        
        # Проверяем, что можем извлечь достаточную область вокруг круга
        x, y, r = largest_circle
        margin = int(r * 0.1)
        
        if (x - r - margin >= 0 and y - r - margin >= 0 and 
            x + r + margin <= w and y + r + margin <= h):
            cropped = extract_circle_region(image, largest_circle, size)
            return cropped, True
        else:
            print("⚠ Недостаточно места вокруг найденного круга, используем максимально возможную область")
            # Используем максимально доступную область
            cropped = extract_circle_region(image, largest_circle, size)
            return cropped, True
    
    print("Использую резервный метод - квадратная обрезка по центру")
    # Резервный метод - квадратная обрезка
    min_side = min(h, w)
    x1 = (w - min_side) // 2
    y1 = (h - min_side) // 2
    cropped = image[y1:y1+min_side, x1:x1+min_side]
    resized = cv2.resize(cropped, (size, size))
    return resized, False

def extract_circle_crop(image, size=1024):
    """Оригинальная функция для совместимости"""
    result, _ = extract_circle_crop_improved(image, size)
    return result

def order_points(pts):
    """Orders points in order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

class ImageProcessor:
    def __init__(self):
        self.points = []  # Points for cover square
        self.disc_points = []  # Points on disc
        self.current_image = None
        self.display_image = None  # Масштабированное изображение для отображения
        self.clone = None
        self.image_index = 0
        self.image_paths = []
        self.output_folder = ""
        self.selected_point_idx = -1
        self.is_right_button_down = False
        self.mode = 'square'  # Mode: 'square' or 'disc'
        self.scale_factor = 1.0  # Коэффициент масштабирования
        self.max_display_size = 800  # Максимальный размер окна
    
    def calculate_scale_factor(self, image):
        """Вычисляет коэффициент масштабирования для отображения"""
        height, width = image.shape[:2]
        
        # Определяем максимальный размер экрана (можно настроить)
        max_width = self.max_display_size
        max_height = self.max_display_size
        
        # Вычисляем коэффициент масштабирования
        scale_w = max_width / width if width > max_width else 1.0
        scale_h = max_height / height if height > max_height else 1.0
        
        return min(scale_w, scale_h)
    
    def scale_point_to_original(self, x, y):
        """Преобразует координаты с отображаемого изображения в оригинальные"""
        return (int(x / self.scale_factor), int(y / self.scale_factor))
    
    def scale_point_to_display(self, x, y):
        """Преобразует координаты с оригинального изображения в отображаемые"""
        return (int(x * self.scale_factor), int(y * self.scale_factor))
    
    def click_event(self, event, x, y, flags, param):
        # Преобразуем координаты клика в координаты оригинального изображения
        orig_x, orig_y = self.scale_point_to_original(x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'square' and len(self.points) < 4:
                self.points.append((orig_x, orig_y))
            elif self.mode == 'disc' and len(self.disc_points) < 4:
                self.disc_points.append((orig_x, orig_y))
        
        # PKM - move point
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.is_right_button_down = True
            points_list = self.disc_points if self.mode == 'disc' else self.points
            for i, (px, py) in enumerate(points_list):
                # Сравниваем в координатах отображения
                display_px, display_py = self.scale_point_to_display(px, py)
                if abs(display_px - x) < 10 and abs(display_py - y) < 10:
                    self.selected_point_idx = i
                    break
        
        elif event == cv2.EVENT_MOUSEMOVE and self.is_right_button_down:
            if self.selected_point_idx >= 0:
                if self.mode == 'disc':
                    self.disc_points[self.selected_point_idx] = (orig_x, orig_y)
                else:
                    self.points[self.selected_point_idx] = (orig_x, orig_y)
        
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_right_button_down = False
            self.selected_point_idx = -1
        
        self.update_display()
    
    def find_square_from_disc(self):
        """Finds square by 4 points tangent to the middles of the sides"""
        if len(self.disc_points) < 4:
            return None
        
        # Преобразуем точки в numpy array
        points = np.array(self.disc_points, dtype=np.float32)
        
        # Find center as arithmetic mean of all points
        center = np.mean(points, axis=0)
        
        # Sort points by angle relative to center
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        # Calculate distances from center to each point
        distances = np.linalg.norm(sorted_points - center, axis=1)
        avg_distance = np.mean(distances)
        
        # For a square inscribed in a circle, distance from center to side middle
        # equals radius, and distance from center to corner equals radius * sqrt(2)
        # Therefore square side size = 2 * avg_distance
        square_half_size = avg_distance
        
        # Create square with center at found point
        square_points = np.array([
            [center[0] - square_half_size, center[1] - square_half_size],  # Top-left
            [center[0] + square_half_size, center[1] - square_half_size],  # Top-right
            [center[0] + square_half_size, center[1] + square_half_size],  # Bottom-right
            [center[0] - square_half_size, center[1] + square_half_size]   # Bottom-left
        ], dtype="float32")
        
        return square_points
    
    def update_display(self):
        self.clone = self.display_image.copy()
        
        # Draw current points depending on mode
        if self.mode == 'square':
            points = self.points
            color = (0, 255, 0)  # Green for square points
            text = "Mode: cover square (SPACE=save, Z=switch to disc)"
        else:
            points = self.disc_points
            color = (0, 0, 255)  # Red for disc points
            text = "Mode: disc points (SPACE=save, Z=switch to cover)"
        
        # Отображаем точки в масштабированных координатах
        for i, (x, y) in enumerate(points):
            display_x, display_y = self.scale_point_to_display(x, y)
            pt_color = (0, 0, 255) if (i == self.selected_point_idx and self.is_right_button_down) else color
            cv2.circle(self.clone, (display_x, display_y), 5, pt_color, -1)
            cv2.putText(self.clone, str(i+1), (display_x+10, display_y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, pt_color, 2)
        
        # If in disc mode and have 4 points, show predicted square
        if self.mode == 'disc' and len(self.disc_points) >= 4:
            square_points = self.find_square_from_disc()
            if square_points is not None:
                # Draw square в масштабированных координатах
                for i in range(4):
                    pt1 = self.scale_point_to_display(square_points[i][0], square_points[i][1])
                    pt2 = self.scale_point_to_display(square_points[(i+1)%4][0], square_points[(i+1)%4][1])
                    cv2.line(self.clone, pt1, pt2, (255, 255, 0), 2)
                
                # Show side middles for clarity
                center = np.mean(square_points, axis=0)
                center_display = self.scale_point_to_display(center[0], center[1])
                for i in range(4):
                    mid_point = (square_points[i] + square_points[(i+1)%4]) / 2
                    mid_display = self.scale_point_to_display(mid_point[0], mid_point[1])
                    cv2.circle(self.clone, mid_display, 3, (255, 255, 0), -1)
                    # Draw line from center to side middle
                    cv2.line(self.clone, center_display, mid_display, (255, 255, 0), 1)
        
        # Tips
        cv2.putText(self.clone, text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.clone, "LMB: add point", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.clone, "RMB: move point", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(self.clone, "ESC: skip image", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.mode == 'disc':
            cv2.putText(self.clone, "Mark points on disc edge", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Показываем информацию о масштабе
        if self.scale_factor != 1.0:
            scale_text = f"Scale: {self.scale_factor:.2f}"
            cv2.putText(self.clone, scale_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow("Vinyl Processor", self.clone)
    
    def process_single_image(self):
        self.points = []
        self.disc_points = []
        self.selected_point_idx = -1
        self.is_right_button_down = False
        self.mode = 'square'
        
        self.current_image = cv2.imread(self.image_paths[self.image_index])
        if self.current_image is None:
            print(f"⚠ Loading error: {self.image_paths[self.image_index]}")
            return False
        
        # Проверяем, является ли файл "_3" изображением (автоматическая обработка)
        filename = os.path.basename(self.image_paths[self.image_index])
        if "_3" in os.path.splitext(filename)[0]:
            print(f"🔍 Auto-processing _3 image: {filename}")
            print(f"Размер изображения: {self.current_image.shape}")
            
            cropped, circle_found = extract_circle_crop_improved(self.current_image)
            
            if circle_found:
                print("✅ Круг успешно найден и обработан")
            else:
                print("⚠ Круг не найден, использована квадратная обрезка")
            
            output_path = os.path.join(self.output_folder, filename)
            cv2.imwrite(output_path, cropped)
            print(f"✅ Auto-processed: {output_path}")
            return True
        
        # Вычисляем коэффициент масштабирования и создаем отображаемое изображение
        self.scale_factor = self.calculate_scale_factor(self.current_image)
        if self.scale_factor != 1.0:
            new_width = int(self.current_image.shape[1] * self.scale_factor)
            new_height = int(self.current_image.shape[0] * self.scale_factor)
            self.display_image = cv2.resize(self.current_image, (new_width, new_height))
            print(f"Image scaled by factor: {self.scale_factor:.2f} ({self.current_image.shape[1]}x{self.current_image.shape[0]} -> {new_width}x{new_height})")
        else:
            self.display_image = self.current_image.copy()
        
        # Создаем окно с возможностью изменения размера
        cv2.namedWindow("Vinyl Processor", cv2.WINDOW_NORMAL)
        
        # Устанавливаем размер окна
        window_height, window_width = self.display_image.shape[:2]
        cv2.resizeWindow("Vinyl Processor", window_width, window_height)
        
        cv2.setMouseCallback("Vinyl Processor", self.click_event)
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            # SPACE - save and go to next
            if key == 32:
                if self.mode == 'square' and len(self.points) == 4:
                    break
                elif self.mode == 'disc' and len(self.disc_points) >= 4:
                    self.points = self.find_square_from_disc()
                    if self.points is not None:
                        self.points = self.points.tolist()
                        self.mode = 'square'
                        self.update_display()
            
            # Z - switch mode (square/disc)
            elif key == ord('z'):
                self.mode = 'disc' if self.mode == 'square' else 'square'
                self.update_display()
            
            # ESC - skip current image
            elif key == 27:
                cv2.destroyWindow("Vinyl Processor")
                return False
        
        # Perspective correction (используем оригинальные координаты)
        rect = order_points(np.array(self.points, dtype="float32"))
        dst = np.array([[0, 0], [1023, 0], [1023, 1023], [0, 1023]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.current_image, M, (1024, 1024))
        
        # Save result
        output_path = os.path.join(self.output_folder, os.path.basename(self.image_paths[self.image_index]))
        cv2.imwrite(output_path, warped)
        print(f"✅ Processed: {output_path}")
        cv2.destroyWindow("Vinyl Processor")
        return True
    
    def process_folder(self, input_folder, output_folder):
        self.image_paths = sorted(glob.glob(os.path.join(input_folder, "*")))
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        total = len(self.image_paths)
        print(f"Found images: {total}")
        
        while self.image_index < total:
            filename = os.path.basename(self.image_paths[self.image_index])
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.JPG')):
                self.image_index += 1
                continue
                
            print(f"\nProcessing {self.image_index+1}/{total}: {filename}")
            
            # Для файлов без _3 показываем инструкции
            if "_3" not in os.path.splitext(filename)[0]:
                print("Instructions:")
                print("1. Default - cover square mode (green points)")
                print("2. Press Z to switch to disc mode (red points)")
                print("3. Mark 4 points on disc edge (tangent to side middles)")
                print("4. Press SPACE for automatic square detection")
                print("5. Press SPACE again to save") 
                print("6. ESC - skip current image")
            
            success = self.process_single_image()
            self.image_index += 1
            
            if not success:
                print(f"⏭ Skipped: {filename}")
        
        cv2.destroyAllWindows()
        print("\nProcessing completed!")

if __name__ == "__main__":
    processor = ImageProcessor()
    input_folder = "input_folder"
    output_folder = "output_folder"
    processor.process_folder(input_folder, output_folder)