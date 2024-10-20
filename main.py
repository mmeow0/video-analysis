import cv2
import json
import numpy as np
from ultralytics import YOLO

# Функция для конвертации numpy и тензоров в стандартные типы Python
def convert_to_standard_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()  # Преобразование numpy массивов в списки
    elif isinstance(data, np.generic):
        return data.item()  # Преобразование numpy типов (например, float32) в стандартные Python типы
    elif hasattr(data, 'cpu') and hasattr(data, 'numpy'):  # Если это тензор, преобразуем его в numpy и затем в список
        return data.cpu().numpy().tolist()
    return data

def extract_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    metadata = []

    # Переменные для детекции движения
    previous_frame = None
    movement_threshold = 10000  # Пороговое значение для движения

    # Загрузка YOLOv8 модели
    model = YOLO('yolov8n.pt')  # Используем версию nano для большей скорости

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Преобразуем кадр в чёрно-белый для упрощённой обработки
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        # Если предыдущий кадр существует, сравниваем для выявления движений
        if previous_frame is None:
            previous_frame = gray_frame
            continue

        frame_diff = cv2.absdiff(previous_frame, gray_frame)
        _, threshold_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        movement_score = threshold_diff.sum()

        # Обнаружение движения на основе порога
        is_moving = movement_score > movement_threshold
        movement_status = "moving" if is_moving else "static"

        # Обновляем предыдущий кадр для следующей итерации
        previous_frame = gray_frame

        # Выполняем детекцию объектов на кадре
        results = model(frame)
        
        objects = []
        for obj in results[0].boxes:
            objects.append({
                'object_id': obj.cls.item(),
                'confidence': round(float(obj.conf.item()), 2),
                'bbox': convert_to_standard_types(obj.xyxy[0])  # Преобразуем координаты объектов в стандартный тип
            })

        # Пример извлечения цветового профиля кадра
        avg_color_per_row = frame.mean(axis=0)
        avg_color = avg_color_per_row.mean(axis=0)
        
        # Преобразуем значения цветов из numpy типов в стандартные float
        avg_color = convert_to_standard_types(avg_color)
        
        # Добавим метаданные для каждого кадра
        metadata.append({
            'frame': frame_count,
            'avg_color': avg_color,  # Преобразованные значения цветов
            'movement_status': movement_status,
            'detected_objects': objects
        })

        frame_count += 1

    cap.release()
    
    # Записываем метаданные в файл с приведением типов
    with open('metadata_yolov8.json', 'w') as f:
        json.dump(convert_to_standard_types(metadata), f, indent=4)

    print("Метаданные успешно сохранены в metadata_yolov8.json")

if __name__ == "__main__":
    video_path = 'rickroll.mp4'
    extract_metadata(video_path)
