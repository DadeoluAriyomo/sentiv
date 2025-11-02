from pathlib import Path
from PIL import Image
import os

# Import app to reuse model, EMOTIONS and crop_and_preprocess
import app

MODEL = getattr(app, 'model', None)
EMOTIONS = getattr(app, 'EMOTIONS', None)
if MODEL is None:
    print('Model not loaded in app.py; aborting.')
    raise SystemExit(1)
if EMOTIONS is None:
    EMOTIONS = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

p = Path('FER2013/test/happy')
if not p.exists():
    print('Test path', p, 'not found. Aborting.')
    raise SystemExit(1)

count = 0
for f in p.glob('*.jpg'):
    if count >= 10:
        break
    img = Image.open(f)
    debug_name = f"dbg_{f.name}"
    arr = app.crop_and_preprocess(img, save_debug=True, debug_filename=debug_name)
    preds = MODEL.predict(arr)
    idx = int(preds.argmax())
    print(f.name, '->', EMOTIONS[idx], f"(conf={preds[0,idx]:.4f})")
    count += 1

print('Saved debug images (48x48) to', os.path.join('static','uploads'))
