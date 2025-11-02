# app.py
import os
import io
import base64
import sqlite3
from datetime import datetime
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
# Use TFLite runtime for inference (low memory)
import tflite_runtime.interpreter as tflite
# Prefer MTCNN if available (stronger detector), otherwise fall back to OpenCV Haar cascade
try:
    from mtcnn import MTCNN
    mtcnn_detector = MTCNN()
except Exception:
    mtcnn_detector = None

try:
    import cv2
    # load Haar cascade; cv2.data.haarcascades is available when opencv-python is installed
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception:
    cv2 = None
    face_cascade = None

# ======================================================
# ✅ STEP 1: Model Configuration (updated to best_fer_model.keras)
# ======================================================
MODEL_PATH = "best_fer_model.tflite"  # TFLite model file
UPLOAD_FOLDER = os.path.join("static", "uploads")
DB_PATH = "database_sentiv.db"
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ======================================================
# ✅ STEP 2: FER2013 Emotion Order (matches training)
# ======================================================
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ======================================================
# ✅ STEP 1 (continued): Load model safely
# ======================================================
# Load TFLite model
if os.path.exists(MODEL_PATH):
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ Loaded TFLite model successfully.")
        print("Input details:", input_details)
        print("Output details:", output_details)
    except Exception as e:
        interpreter = None
        print(f"❌ Failed to load TFLite model: {e}")
else:
    interpreter = None
    print(f"❌ TFLite model file {MODEL_PATH} not found.")

# ======================================================
# Database Setup (unchanged)
# ======================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filename TEXT,
            emotion TEXT,
            confidence REAL,
            timestamp TEXT,
            mode TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_usage(name, filename, emotion, confidence, mode='upload'):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO usage_log (name, filename, emotion, confidence, timestamp, mode) VALUES (?, ?, ?, ?, ?, ?)',
              (name, filename, emotion, float(confidence), datetime.utcnow().isoformat(), mode))
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# ======================================================
# ✅ STEP 3: Ensure preprocessing matches training
# ======================================================
def crop_and_preprocess(img: Image.Image, save_debug: bool = False, debug_filename: str = None):
    """Attempt to detect and crop the largest face, fallback to center-crop.
    Then resize to 48x48, normalize and return shape (1,48,48,1).
    If save_debug=True and debug_filename provided, saves the 48x48 image to uploads.
    """
    pil_img = img
    # Prepare both gray and RGB numpy arrays for different detectors
    gray_np = np.array(pil_img.convert('L'))
    rgb_np = np.array(pil_img.convert('RGB'))

    # Try MTCNN first (if available)
    faces = []
    if 'mtcnn_detector' in globals() and mtcnn_detector is not None:
        try:
            mt_faces = mtcnn_detector.detect_faces(rgb_np)
            # mt_faces is a list of dicts with 'box' and 'confidence'
            if isinstance(mt_faces, list) and len(mt_faces) > 0:
                # convert to (x,y,w,h) tuples
                faces = [tuple(f['box']) for f in mt_faces if 'box' in f]
        except Exception:
            faces = []

    # If MTCNN found nothing, try Haar cascade on grayscale
    if len(faces) == 0 and face_cascade is not None:
        try:
            haar_faces = face_cascade.detectMultiScale(gray_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces = [tuple(f) for f in haar_faces]
        except Exception:
            faces = []

    if len(faces) == 0:
        # fallback: center-crop
        w, h = pil_img.size
        min_edge = min(w, h)
        left = (w - min_edge) // 2
        top = (h - min_edge) // 2
        cropped = pil_img.crop((left, top, left + min_edge, top + min_edge)).convert('L')
    else:
        # choose the largest detected face
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        # Some detectors may return negative coords; clamp and ensure within image
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)
        x2, y2 = x + w, y + h
        # Clamp to image dimensions
        W, H = pil_img.size
        x2 = min(x2, W)
        y2 = min(y2, H)
        cropped = pil_img.crop((x, y, x2, y2)).convert('L')

    cropped = cropped.resize((48, 48))
    arr = np.asarray(cropped).astype('float32') / 255.0

    if save_debug and debug_filename:
        try:
            debug_path = os.path.join(app.config['UPLOAD_FOLDER'], debug_filename)
            # Save scaled back to 0-255 so it's viewable
            Image.fromarray((arr * 255).astype('uint8')).save(debug_path)
        except Exception as e:
            print("Could not save debug image:", e)

    arr = arr.reshape(1, 48, 48, 1)
    return arr

def feedback_for_emotion(emotion):
    mapping = {
        'Happy': "Nice! Keep that smile — consider sharing your joy with someone today.",
        'Surprise': "Something caught you off guard — take a breath and stay curious.",
        'Neutral': "Feeling neutral is okay. Maybe try a short break or a walk to reset.",
        'Sad': "Sorry you're feeling down. Try reaching out to a friend, or do something small you enjoy.",
        'Angry': "Take a calming moment. Try deep breaths, walk away for a few minutes, or count to ten.",
        'Fear': "You seem uneasy. Slow breaths and grounding techniques can help. Reach out for support if needed.",
        'Disgust': "That feeling matters. Consider what specifically is causing it and whether you can avoid it or address it."
    }
    return mapping.get(emotion, "")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    name = request.form.get('name', 'Anonymous')
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path)
        # save a debug version of the preprocessed image that the model sees
        debug_filename = f"debug_{filename}"
        arr = crop_and_preprocess(img, save_debug=True, debug_filename=debug_filename)

        if interpreter is None:
            return jsonify({'error': 'Model not available on server'}), 500

        # TFLite inference
        interpreter.set_tensor(input_details[0]['index'], arr.astype(np.float32))
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
        print("Raw predictions:", preds)  # DEBUG
        idx = int(np.argmax(preds))
        emotion = EMOTIONS[idx]
        confidence = float(preds[0, idx])

        log_usage(name, filename, emotion, confidence, mode='upload')
        feedback = feedback_for_emotion(emotion)
        return jsonify({'emotion': emotion, 'confidence': confidence, 'feedback': feedback, 'filename': filename})
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

@app.route('/predict_live', methods=['POST'])
def predict_live():
    data = request.get_json(force=True)
    b64 = data.get('image_base64', '')
    name = data.get('name', 'Anonymous')
    if not b64:
        return jsonify({'error': 'No image provided'}), 400

    header, encoded = b64.split(',', 1) if ',' in b64 else ('', '')
    try:
        image_data = base64.b64decode(encoded)
    except Exception as e:
        return jsonify({'error': 'Invalid base64 image', 'details': str(e)}), 400

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_live.png"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(file_path, 'wb') as f:
        f.write(image_data)

    img = Image.open(io.BytesIO(image_data))
    debug_filename = f"debug_{filename}"
    arr = crop_and_preprocess(img, save_debug=True, debug_filename=debug_filename)

    if interpreter is None:
        return jsonify({'error': 'Model not available on server'}), 500

    interpreter.set_tensor(input_details[0]['index'], arr.astype(np.float32))
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    print("Raw predictions (live):", preds)  # DEBUG
    idx = int(np.argmax(preds))
    emotion = EMOTIONS[idx]
    confidence = float(preds[0, idx])

    log_usage(name, filename, emotion, confidence, mode='live')
    feedback = feedback_for_emotion(emotion)
    return jsonify({'emotion': emotion, 'confidence': confidence, 'feedback': feedback, 'filename': filename})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/stats')
def stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT emotion, COUNT(*) FROM usage_log GROUP BY emotion ORDER BY COUNT(*) DESC')
    rows = c.fetchall()
    conn.close()
    return jsonify({'counts': rows})

if __name__ == "__main__":
    # Read port from environment (Render provides PORT); default to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    # Enable debug only when FLASK_DEBUG or APP_DEBUG env var is set to '1'
    debug_env = os.environ.get("FLASK_DEBUG") or os.environ.get("APP_DEBUG")
    debug = True if str(debug_env) == '1' else False
    # Bind to 0.0.0.0 so Render (or other hosts) can route traffic
    app.run(host='0.0.0.0', port=port, debug=debug)
