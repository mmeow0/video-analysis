import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

def convert_to_standard_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif hasattr(data, 'cpu') and hasattr(data, 'numpy'):
        return data.cpu().numpy().tolist()
    return data

def extract_features(data, max_objects=5):  # max_objects - максимальное количество объектов
    features = []
    for entry in data:
        object_features = []
        for obj in entry["detected_objects"][:max_objects]:  # Берем только до max_objects
            object_features.append(obj['object_id'])
            object_features.append(obj['confidence'])
        # Если меньше max_objects, заполняем пустыми значениями
        while len(object_features) < max_objects * 2:
            object_features.append(0)  # Или используйте 0, если это более целесообразно
        features.append(object_features)
    # logging.info(f"Frame {features}")
    return np.array(features, dtype=float)

def train_autoencoder(data):
    feature_dim = data.shape[1]
    autoencoder = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(feature_dim,)),  # Используем Input вместо input_shape
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(feature_dim, activation='sigmoid')
    ])
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
    autoencoder.fit(data, data, epochs=50, batch_size=16, shuffle=True)
    return autoencoder

def normalize_features(features):
    # Применяем нормализацию
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

def sort_detected_objects(objects):
    return sorted(objects, key=lambda x: (x['object_id'], x['confidence']))

def compare_detected_objects(original_objects, new_objects):
    # Сравниваем отсортированные объекты
    sorted_original = sort_detected_objects(original_objects)
    sorted_new = sort_detected_objects(new_objects)

    return sorted_original == sorted_new

# Функция для выявления аномалий
def detect_anomalies(original_metadata, new_metadata, autoencoder, threshold=5.0):
    original_features = extract_features(original_metadata)
    new_features = extract_features(new_metadata)

    if original_features.size == 0 or new_features.size == 0:
        logging.error("Нет доступных признаков для сравнения.")
        return []

    original_features = normalize_features(original_features)
    new_features = normalize_features(new_features)

    reconstructed = autoencoder.predict(new_features)
    mse = np.mean(np.power(new_features - reconstructed, 2), axis=1)

    discrepancies = []
    for i, error in enumerate(mse):
        if error > threshold:
            original_objects = original_metadata[i]['detected_objects']
            new_objects = new_metadata[i]['detected_objects']

            # Сравнение объектов
            objects_are_equal = compare_detected_objects(original_objects, new_objects)
            if not objects_are_equal:
                discrepancy = {
                    'frame': new_metadata[i]['frame'],
                    'original_detected_objects': original_objects,
                    'new_detected_objects': new_objects,
                    'error': f'Anomaly detected with MSE: {error}'
                }
                discrepancies.append(discrepancy)
    return discrepancies

def create_metadata(video_path):
    try:
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
    video_path = 'rickroll_fake.mp4'
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

    # Извлечение признаков для обучения автоэнкодера
    original_features = extract_features(original_metadata)
    logging.info(f"Original features shape: {original_features.shape}")

    if original_features.size == 0:
        logging.error("Нет доступных признаков для обучения автоэнкодера.")
        exit()

    # Обучение автоэнкодера
    autoencoder = train_autoencoder(original_features)

    # Выявление аномалий
    discrepancies = detect_anomalies(original_metadata, new_metadata, autoencoder)

    # Сохранение несоответствий
    with open(output_discrepancy_path, 'w') as f:
        json.dump(discrepancies, f, indent=4)
        logging.info(f"Несоответствия сохранены в {output_discrepancy_path}")
