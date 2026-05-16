from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
from tensorflow import keras

# ==================== CONFIG ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
PHOTOS_FOLDER = os.path.join(BASE_DIR, 'photos')

print(f"📁 BASE_DIR: {BASE_DIR}")
print(f"📁 MODELS_FOLDER: {MODELS_FOLDER}")
print(f"📁 MODELS_FOLDER exists: {os.path.exists(MODELS_FOLDER)}")

if os.path.exists(MODELS_FOLDER):
    print(f"📁 Files in MODELS_FOLDER: {os.listdir(MODELS_FOLDER)}")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== CLASSIFIER ====================

class IronWaterClassifier:
    def __init__(self, object_type, model_path):
        self.object_type = object_type

        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded: {object_type} from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load {object_type}: {e}")
            self.model = None

        if object_type == 'orange':
            self.class_names = ['orange_clean', 'orange_iron_contaminated']
        elif object_type == 'banana':
            self.class_names = ['banana_clean', 'banana_iron_contaminated']
        elif object_type == 'egg':
            self.class_names = ['egg_clean', 'egg_iron_contaminated']

        self.condition_map = {
            'clean': 'Clean Water',
            'iron_contaminated': 'Iron Contaminated Water'
        }

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return np.array(img, dtype=np.float32) / 255.0

    def classify(self, img):
        if self.model is None:
            raise ValueError("Model not loaded")

        img = self.preprocess(img)
        img = np.expand_dims(img, axis=0)

        pred = self.model.predict(img, verbose=0)[0]
        idx = np.argmax(pred)
        confidence = float(pred[idx])

        class_name = self.class_names[idx]
        condition_key = "_".join(class_name.split('_')[1:])
        condition = self.condition_map.get(condition_key, condition_key)

        return {
            "condition": condition,
            "confidence": confidence
        }

# ==================== LOAD MODELS ====================

classifiers = {}

def load_models():
    print("\n" + "="*50)
    print("🔄 LOADING MODELS")
    print("="*50)
    
    model_files = {
        'orange': os.path.join(MODELS_FOLDER, 'orange_classifier.keras'),
        'banana': os.path.join(MODELS_FOLDER, 'banana_classifier.keras'),
        'egg': os.path.join(MODELS_FOLDER, 'egg_classifier.keras')
    }

    for obj, path in model_files.items():
        print(f"\n🔍 Checking {obj}...")
        print(f"   Path: {path}")
        print(f"   Exists: {os.path.exists(path)}")
        
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            print(f"   Size: {file_size} bytes")
            
            if file_size < 1000:  # Less than 1KB means corrupted/placeholder
                print(f"   ⚠️ File seems corrupted (too small)")
                continue
                
            classifiers[obj] = IronWaterClassifier(obj, path)
        else:
            print(f"   ❌ Model file not found: {path}")

    print("\n" + "="*50)
    print(f"✅ Loaded models: {list(classifiers.keys())}")
    print("="*50 + "\n")

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


# Serve photos
@app.route('/photos/<filename>')
def serve_photo(filename):
    if not os.path.exists(PHOTOS_FOLDER):
        return {'error': 'Photos folder not found'}, 404
    return send_from_directory(PHOTOS_FOLDER, filename)


@app.route('/api/classify-image', methods=['POST'])
def classify_image():
    try:
        object_type = request.form.get('object_type')
        print(f"\n🔍 Classification request for: {object_type}")

        if object_type not in classifiers:
            error_msg = f'Model not loaded: {object_type}. Available: {list(classifiers.keys())}'
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400

        if 'image' not in request.files:
            print("❌ No image uploaded")
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Invalid image file")
            return jsonify({'error': 'Invalid image file'}), 400

        result = classifiers[object_type].classify(img)
        print(f"✅ Classification result: {result}")

        return jsonify({
            'success': True,
            'result': result
        })  

    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 STARTING ORION APP")
    print("="*50)

    load_models()

    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Listen on all interfaces for Railway
    
    print(f"🌐 Server running on http://0.0.0.0:{port}")
    app.run(host=host, port=port, debug=False)
