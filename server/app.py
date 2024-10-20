from flask import Flask, request, render_template, redirect, url_for
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

latest_video = None
latest_metadata = None

@app.route('/')
def index():
    global latest_video, latest_metadata

    if not latest_video or not latest_metadata:
        return 'No video or metadata uploaded yet', 400

    with open(latest_metadata) as f:
        metadata = json.load(f)

    return render_template('index.html', video_path=latest_video, metadata=metadata)

@app.route('/upload', methods=['POST'])
def upload():
    global latest_video, latest_metadata

    if 'video' not in request.files or 'metadata' not in request.files:
        return 'Video or metadata file not provided', 400

    video_file = request.files['video']
    metadata_file = request.files['metadata']

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)

    metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_file.filename)
    metadata_file.save(metadata_path)

    latest_video = os.path.join('static', 'uploads', video_file.filename)
    latest_metadata = os.path.join('static', 'uploads', metadata_file.filename)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=5043)
