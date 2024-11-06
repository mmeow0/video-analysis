from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import os
import numpy as np
import json
import cv2
from ultralytics import YOLO
import logging
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

names = {
    0: 'человек',
    1: 'велосипед',
    2: 'автомобиль',
    3: 'мотоцикл',
    4: 'самолет',
    5: 'автобус',
    6: 'поезд',
    7: 'грузовик',
    8: 'лодка',
    9: 'светофор',
    10: 'пожарный гидрант',
    11: 'знак стоп',
    12: 'паркомат',
    13: 'скамейка',
    14: 'птица',
    15: 'кот',
    16: 'собака',
    17: 'лошадь',
    18: 'овца',
    19: 'корова',
    20: 'слон',
    21: 'медведь',
    22: 'зебра',
    23: 'жираф',
    24: 'рюкзак',
    25: 'зонт',
    26: 'сумка',
    27: 'галстук',
    28: 'чемодан',
    29: 'фрисби',
    30: 'лыжи',
    31: 'сноуборд',
    32: 'спортивный мяч',
    33: 'воздушный змей',
    34: 'бейсбольная бита',
    35: 'бейсбольная перчатка',
    36: 'скейтборд',
    37: 'серфборд',
    38: 'теннисная ракетка',
    39: 'бутылка',
    40: 'бокал для вина',
    41: 'чашка',
    42: 'вилка',
    43: 'нож',
    44: 'ложка',
    45: 'миска',
    46: 'банан',
    47: 'яблоко',
    48: 'бутерброд',
    49: 'апельсин',
    50: 'брокколи',
    51: 'морковь',
    52: 'хот-дог',
    53: 'пицца',
    54: 'пончик',
    55: 'торт',
    56: 'стул',
    57: 'диван',
    58: 'горшечное растение',
    59: 'кровать',
    60: 'обеденный стол',
    61: 'туалет',
    62: 'телевизор',
    63: 'ноутбук',
    64: 'мышь',
    65: 'пульт',
    66: 'клавиатура',
    67: 'мобильный телефон',
    68: 'микроволновка',
    69: 'духовой шкаф',
    70: 'тостер',
    71: 'раковина',
    72: 'холодильник',
    73: 'книга',
    74: 'часы',
    75: 'ваза',
    76: 'ножницы',
    77: 'плюшевая игрушка',
    78: 'фен',
    79: 'зубная щетка'
}

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
    print(data)
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
def detect_anomalies(original_metadata, new_metadata, autoencoder, threshold=10.0):
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
                    'error_mse': error
                }
                discrepancies.append(discrepancy)
    return discrepancies

app = Flask(__name__, static_folder='../static', static_url_path='/static')
UPLOAD_FOLDER = 'uploads'  # Папка для загрузки в пределах static/uploads
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, UPLOAD_FOLDER)

uploaded_videos = {}
autoencoder = None  # Будем сохранять обученный автоэнкодер здесь
original_features = None  # Сохраняем признаки оригинальных метаданных

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video_file = request.files['video']
    if not video_file:
        return jsonify({'error': 'Файл не предоставлен'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # Проверка на существование метаданных
    metadata_filename = f"{os.path.splitext(video_file.filename)[0]}_metadata.json"
    metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_filename)

    uploaded_videos[video_file.filename] = {
        'video_path': video_path,
        'metadata_path': metadata_path,
        'metadata_exists': os.path.exists(metadata_path)
    }
    print(url_for('static', filename=f'uploads/{video_file.filename}', _external=True))
    return jsonify({
        'video_path': url_for('static', filename=f'uploads/{video_file.filename}', _external=True),
        'metadata_exists': uploaded_videos[video_file.filename]['metadata_exists']
    })

@app.route('/generate_metadata/<filename>', methods=['POST'])
def generate_metadata(filename=''):
    global autoencoder, original_features
    if filename not in uploaded_videos:
        return jsonify({'error': 'Нет загруженного видео'}), 400

    latest_video = uploaded_videos[filename]
    metadata, metadata_filename = extract_metadata(latest_video['video_path'])

    latest_video['metadata_path'] = url_for('static', filename=f'uploads/{metadata_filename}', _external=True)
    if original_features is None and autoencoder is None:
        original_features = extract_features(metadata)
        original_features = normalize_features(original_features)
        autoencoder = train_autoencoder(original_features)

    return jsonify({'metadata': metadata, 'metadata_path': latest_video['metadata_path']})

@app.route('/load_metadata/<filename>', methods=['GET'])
def load_existing(filename=''):
    global autoencoder, original_features

    if filename not in uploaded_videos:
        return jsonify({'error': 'Нет загруженного видео.'}), 400

    latest_video = uploaded_videos[filename]

    if not os.path.exists(latest_video['video_path']):
        return jsonify({'error': 'Нет загруженного видео.'}), 400

    if not os.path.exists(latest_video['metadata_path']):
        return jsonify({'error': 'Метаданные не найдены.'}), 400

    with open(latest_video['metadata_path'], 'r') as f:
        metadata = json.load(f)

    if original_features is None and autoencoder is None:
        original_features = extract_features(metadata)
        original_features = normalize_features(original_features)
        autoencoder = train_autoencoder(original_features)
    
    return jsonify({
        'video_path': url_for('static', filename=f'uploads/{filename}', _external=True),
        'metadata': metadata,
        'metadata_path': latest_video['metadata_path']
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    global autoencoder, original_features

    # Получаем имя файла метаданных из запроса
    data = request.get_json()
    video_source = data.get('filename_source')
    video_dest = data.get('filename_dest')
    if not video_source or not video_dest:
        return jsonify({'error': 'Имя файлов метаданных не предоставлено.'}), 400

    metadata_filename_source = f"{os.path.splitext(video_source)[0]}_metadata.json"
    metadata_filename_dest = f"{os.path.splitext(video_dest)[0]}_metadata.json"
    
    # Определяем путь к файлу метаданных
    metadata__source_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_filename_source)
    metadata_dest_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_filename_dest)
    
    # Проверяем существование файла метаданных
    if not os.path.exists(metadata__source_path) or not os.path.exists(metadata_dest_path):
        return jsonify({'error': 'Метаданные не найдены.'}), 404

    # Читаем содержимое файла
    with open(metadata__source_path, 'r') as f:
        try:
            original_metadata = json.load(f)  # Преобразуем файл в JSON
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Ошибка разбора JSON: {str(e)}'}), 400

    with open(metadata_dest_path, 'r') as f:
        try:
            dest_metadata = json.load(f)  # Преобразуем файл в JSON
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Ошибка разбора JSON: {str(e)}'}), 400

    # Выявление аномалий в новых метаданных
    discrepancies = detect_anomalies(original_metadata, dest_metadata, autoencoder)

    discrepancies_filename = f"{os.path.splitext(video_source)[0]}{os.path.splitext(video_dest)[0]}_discrepancies.json"
    discrepancies_path = os.path.join(app.config['UPLOAD_FOLDER'], discrepancies_filename)
    
    # Сохраняем список discrepancies в JSON файл
    with open(discrepancies_path, 'w') as f:
        json.dump(discrepancies, f, ensure_ascii=False, indent=4)

    # Возвращаем результат
    if not discrepancies:
        return jsonify({'message': 'Аномалии не обнаружены.'})
    else:
        return jsonify({
            'discrepancies': discrepancies,
            'discrepancies_path': url_for('static', filename=f'uploads/{discrepancies_filename}', _external=True),
            })


# Функция создания метаданных
def extract_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Не удалось открыть видеофайл: {video_path}")
        return None, None

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
                'object_name': names[obj.cls.item()],
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
