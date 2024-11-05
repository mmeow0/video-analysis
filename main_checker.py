import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)

def convert_to_standard_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif hasattr(data, 'cpu') and hasattr(data, 'numpy'):
        return data.cpu().numpy().tolist()
    return data

def find_object(objects, target):
    for obj in objects:
        if obj['object_id'] == target['object_id']:
            return obj
    return None

def compare_frames(frame_a, frame_b) -> List[Dict[str, Any]]:
    errors = []

    detected_objects_a = frame_a['detected_objects']
    detected_objects_b = frame_b['detected_objects']
    
    new_objects_in_b = []

    for obj_b in detected_objects_b:
        match = find_object(detected_objects_a, obj_b)
        if not match:
            new_objects_in_b.append(obj_b)

    if new_objects_in_b:
        errors.append({
            'frame_b': frame_b['frame'],
            'error': 'Different detected_objects',
            'detected_new_object': [obj['object_id'] for obj in new_objects_in_b]
        })
    
    return errors

# Функция для сравнения двух JSON-файлов с допуском по кадрам
def compare_videos(metadata_a: List[Dict[str, Any]], metadata_b: List[Dict[str, Any]], frame_tolerance=10) -> List[Dict[str, Any]]:
    discrepancies = []
    frames_b = {frame['frame']: frame for frame in metadata_b}

    for frame_a in metadata_a:
        frame = frame_a['frame']
        errors_count = 0
        total_frames_checked = 0
        detailed_errors = []

        # Проверка ближайших кадров в диапазоне tolerance
        for delta in range(-frame_tolerance, frame_tolerance + 1):
            frame_b = frames_b.get(frame + delta)
            if frame_b:
                total_frames_checked += 1
                errors = compare_frames(frame_a, frame_b)
                if errors:
                    errors_count += 1
                    for error in errors:
                        compact_error = {
                            'frame_b': error['frame_b'],
                            'error': error['error'],
                            'detected_new_object': error.get('detected_new_object', [])
                        }
                        detailed_errors.append(compact_error)

        # Проверяем, если ошибки были во всех проверяемых кадрах
        if total_frames_checked > 0 and errors_count == total_frames_checked:
            discrepancies.append({
                'frame_a': frame,
                'errors': detailed_errors
            })

    return discrepancies

def create_metadata(video_path):
    try:
        # Инициализация модели и видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Не удалось открыть видеофайл: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        new_metadata = []
        previous_frame = None
        movement_threshold = 10000
        model = YOLO('yolov8n.pt')
        frame_skip = 2  # через фрейм пропускать

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
            
            new_metadata.append({
                'frame': frame_count,
                'avg_color': avg_color,
                'movement_status': movement_status,
                'detected_objects': objects,
                'fps': fps
            })

            frame_count += 1

        cap.release()

        return new_metadata

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        return []

if __name__ == "__main__":
    video_path = 'rickroll_bad.mp4'
    original_metadata_path = 'rickroll.json'
    output_discrepancy_path = 'discrepancies.json'

    if os.path.exists(original_metadata_path):
        with open(original_metadata_path, 'r') as f:
            original_metadata = json.load(f)
    else:
        logging.error(f"Оригинальный файл метаданных {original_metadata_path} не найден.")
        exit()

    new_metadata_filename = f"{os.path.splitext(video_path)[0]}_new_metadata.json"
    if not os.path.exists(new_metadata_filename):
        new_metadata = create_metadata(video_path)
        with open(new_metadata_filename, 'w') as f:
            json.dump(new_metadata, f, indent=4)
            logging.info(f"Метаданные сохранены в {new_metadata_filename}")
    else:
        with open(new_metadata_filename, 'r') as f:
            new_metadata = json.load(f)

    discrepancies = compare_videos(original_metadata, new_metadata)

    with open(output_discrepancy_path, 'w') as f:
        json.dump(discrepancies, f, indent=4)
        logging.info(f"Несоответствия сохранены в {output_discrepancy_path}")
