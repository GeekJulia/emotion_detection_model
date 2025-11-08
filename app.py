# app.py
import os
import io
import sqlite3
from datetime import datetime
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
import torch.nn as nn
from torchvision import transforms

# -------- CONFIG --------
MODEL_PATH = "model.pth"
DB_PATH = "database.db"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (48, 48)

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MOOD_MESSAGES = {
    'Happy': "You are happy😂",
    'Sad': "You are sad😔",
    'Angry': "You seem angry😠",
    'Disgust': "You look disgusted🤢",
    'Fear': "You look scared😨",
    'Surprise': "You look surprised😲",
    'Neutral': "You are neutral🙂"
}

# -------- MODEL (same arch as training) --------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -------- UTILITIES --------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                predicted_emotion TEXT,
                confidence REAL
            )
        ''')
        conn.commit()
        conn.close()

def log_prediction(filename, emotion, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (timestamp, filename, predicted_emotion, confidence)
        VALUES (?, ?, ?, ?)
    ''', (datetime.utcnow().isoformat(), filename, emotion, float(confidence)))
    conn.commit()
    conn.close()

# -------- MODEL LOAD --------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def load_model(path=MODEL_PATH):
    model = SmallCNN(num_classes=len(EMOTION_LABELS)).to(DEVICE)
    if not os.path.exists(path):
        print(f"Warning: model file {path} not found. The app will run but predictions will be random (untrained).")
        return model
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

app = Flask(__name__, template_folder='templates')

model = load_model()
init_db()

# -------- ROUTES --------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({"error": "Invalid image"}), 400

    x = transform(image).unsqueeze(0).to(DEVICE)  # [1,1,48,48]
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        label = EMOTION_LABELS[idx]
        confidence = float(probs[idx])

    log_prediction(file.filename, label, confidence)
    mood_msg = MOOD_MESSAGES.get(label, f"You look {label}")

    return jsonify({
        "emotion": label,
        "confidence": confidence,
        "mood_message": mood_msg
    })

@app.route('/history', methods=['GET'])
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, timestamp, filename, predicted_emotion, confidence FROM predictions ORDER BY id DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "timestamp": r[1],
            "filename": r[2],
            "predicted_emotion": r[3],
            "confidence": r[4]
        })
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
