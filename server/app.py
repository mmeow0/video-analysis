from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import os
import numpy as np
import json
import cv2
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


app = Flask(__name__, static_folder='../static', static_url_path='/static')  # Укажите правильный путь к статической папке
UPLOAD_FOLDER = 'uploads'  # Папка для загрузки в пределах static/uploads
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, UPLOAD_FOLDER)

latest_video = None
latest_metadata_path = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global latest_video

    video_file = request.files['video']
    if not video_file:
        return jsonify({'error': 'Файл не предоставлен'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)
    latest_video = video_path

    return jsonify({'video_path': url_for('static', filename=f'uploads/{video_file.filename}', _external=True)})

@app.route('/generate-metadata', methods=['POST'])
def generate_metadata():
    global latest_video, latest_metadata_path
    if not latest_video:
        return jsonify({'error': 'Видео не загружено'}), 400

    metadata, metadata_filename = extract_metadata(latest_video)
    latest_metadata_path = url_for('static', filename=f'uploads/{metadata_filename}', _external=True)  # Путь к метаданным

    return jsonify({'metadata': metadata, 'metadata_path': latest_metadata_path})

# Обработчик для загрузки существующего видео и метаданных
@app.route('/load-existing', methods=['GET'])
def load_existing():
    global latest_video, latest_metadata_path
    if not latest_video or not os.path.exists(latest_video):
        return jsonify({'error': 'Нет загруженного видео.'}), 400

    # Загружаем метаданные
    metadata_filename = f"{os.path.splitext(os.path.basename(latest_video))[0]}_metadata.json"
    metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_filename)

    if not os.path.exists(metadata_path):
        return jsonify({'error': 'Метаданные не найдены.'}), 400

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return jsonify({
        'video_path': url_for('static', filename=f'uploads/{os.path.basename(latest_video)}', _external=True),
        'metadata': metadata,
        'metadata_path': url_for('static', filename=f'uploads/{metadata_filename}', _external=True)
    })

# Функция создания метаданных
def extract_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Не удалось открыть видеофайл: {video_path}")
        return

    # Получаем количество кадров в секунду (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    metadata = []
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
            
        metadata.append({
            'frame': frame_count,
            'avg_color': avg_color,
            'movement_status': movement_status,
            'detected_objects': objects,
            'fps': fps
        })

        frame_count += 1
    cap.release()
    
    metadata_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_metadata.json"
    with open(os.path.join(app.config['UPLOAD_FOLDER'], metadata_filename), 'w') as f:
        json.dump(metadata, f)

    return metadata, metadata_filename

if __name__ == '__main__':
    app.run(port=5043, debug=True)
