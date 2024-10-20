import cv2
import json
import numpy as np
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)

def convert_to_standard_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif hasattr(data, 'cpu') and hasattr(data, 'numpy'):
        return data.cpu().numpy().tolist()
    return data

def extract_metadata(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Не удалось открыть видеофайл: {video_path}")
            return

        frame_count = 0
        metadata = []
        previous_frame = None
        movement_threshold = 10000
        model = YOLO('yolov8n.pt')
        frame_skip = 24  # Пропускать 23 кадра

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            frame = cv2.resize(frame, (640, 480))  # Уменьшение разрешения для оптимизации
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            if previous_frame is None:
                previous_frame = gray_frame
                frame_count += 1
                continue

            frame_diff = cv2.absdiff(previous_frame, gray_frame)
            _, threshold_diff = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            movement_score = threshold_diff.sum()

            is_moving = movement_score > movement_threshold
            movement_status = "moving" if is_moving else "static"
            previous_frame = gray_frame

            results = model(frame)
            objects = []
            for obj in results[0].boxes:
                objects.append({
                    'object_id': obj.cls.item(),
                    'confidence': round(float(obj.conf.item()), 2),
                    'bbox': convert_to_standard_types(obj.xyxy[0])
                })

            avg_color_per_row = frame.mean(axis=0)
            avg_color = avg_color_per_row.mean(axis=0)
            avg_color = convert_to_standard_types(avg_color)
            
            metadata.append({
                'frame': frame_count,
                'avg_color': avg_color,
                'movement_status': movement_status,
                'detected_objects': objects
            })

            frame_count += 1

        cap.release()
        
        with open('metadata_yolov8.json', 'w') as f:
            json.dump(convert_to_standard_types(metadata), f, indent=4)

        logging.info("Метаданные успешно сохранены в metadata_yolov8.json")

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    video_path = 'rickroll.mp4'
    extract_metadata(video_path)
