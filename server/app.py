from flask import Flask, request, render_template, redirect, url_for
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'  # Измените путь к папке загрузок
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    video_path = os.path.join('static', 'uploads', 'rickroll.mp4')
    metadata_path = os.path.join('static', 'uploads', 'metadata_yolov8.json')

    # Читаем метаданные из JSON файла
    with open(metadata_path) as f:
        metadata = json.load(f)

    return render_template('index.html', video_path=video_path, metadata=metadata)

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file provided', 400

    video_file = request.files['video']
    metadata_file = request.files['metadata']

    # Сохранение видео
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    # Сохранение метаданных
    metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_file.filename)
    metadata_file.save(metadata_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=5043)
