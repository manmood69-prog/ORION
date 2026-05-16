from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import warnings
import urllib.request
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras

# ==================== CONFIG ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
PHOTOS_FOLDER = os.path.join(BASE_DIR, 'photos')

# Ensure models folder exists
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ==================== AUTO-DOWNLOAD MODELS ====================

# Direct Google Drive download links (converted from share links)
model_urls = {
    "orange_classifier.keras": "https://drive.google.com/uc?export=download&id=13GQSTqLghiw5_NmoTQBSlvFjhhgzEelg",
    "banana_classifier.keras": "https://drive.google.com/uc?export=download&id=1yhl_qkeoIwAzY2NZRpO0qQTgPCUqth38",
    "egg_classifier.keras": "https://drive.google.com/uc?export=download&id=1wtnWQ_MUBkpYvqqJZaZbBeWvmaCwH6Dk"
}

def download_models():
    """Auto-download models from Google Drive if they don't exist or are corrupted"""
    print("\n" + "="*60)
    print("📥 CHECKING MODELS...")
    print("="*60)
    
    for model_name, url in model_urls.items():
        model_path = os.path.join(MODELS_FOLDER, model_name)
        
        # Check if file exists and has reasonable size
        file_exists = os.path.exists(model_path)
        file_size = os.path.getsize(model_path) if file_exists else 0
        
        if file_exists and file_size > 1000000:  # More than 1MB = good
            print(f"✅ {model_name} exists ({file_size:,} bytes)")
        else:
            if file_exists:
                print(f"⚠️ {model_name} corrupted/small ({file_size:,} bytes), re-downloading...")
            else:
                print(f"📥 Downloading {model_name}...")
            
            try:
                urllib.request.urlretrieve(url, model_path)
                new_size = os.path.getsize(model_path)
                print(f"✅ {model_name} downloaded! ({new_size:,} bytes)")
            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")

# Download models on startup
download_models()

print(f"\n📁 BASE_DIR: {BASE_DIR}")
print(f"📁 MODELS_FOLDER: {MODELS_FOLDER}")
print(f"📁 MODELS_FOLDER exists: {os.path.exists(MODELS_FOLDER)}")

if os.path.exists(MODELS_FOLDER):
    files = os.listdir(MODELS_FOLDER)
    print(f"📁 Files in MODELS_FOLDER: {files}")
    for f in files:
        full_path = os.path.join(MODELS_FOLDER, f)
        size = os.path.getsize(full_path)
        print(f"   - {f}: {size:,} bytes")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== CLASSIFIER ====================

class IronWaterClassifier:
    def __init__(self, object_type, model_path):
        self.object_type = object_type
        self.model = None
        self.class_names = []

        # Load model with better error handling
        try:
            print(f"   Loading model from: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            file_size = os.path.getsize(model_path)
            print(f"   File size: {file_size:,} bytes")
            
            # Try multiple loading strategies
            loaded = False
            
            # Strategy 1: Standard load
            try:
                self.model = keras.models.load_model(model_path, compile=False)
                print(f"✅ Model loaded successfully (standard): {object_type}")
                loaded = True
            except Exception as e1:
                print(f"   Strategy 1 failed: {type(e1).__name__}")
                
                # Strategy 2: Load with custom_objects
                try:
                    self.model = keras.models.load_model(
                        model_path, 
                        custom_objects=None,
                        safe_mode=False
                    )
                    print(f"✅ Model loaded successfully (custom_objects): {object_type}")
                    loaded = True
                except Exception as e2:
                    print(f"   Strategy 2 failed: {type(e2).__name__}")
                    
                    # Strategy 3: Try with skip_mismatch
                    try:
                        self.model = keras.models.load_model(
                            model_path,
                            skip_mismatch=True
                        )
                        print(f"✅ Model loaded successfully (skip_mismatch): {object_type}")
                        loaded = True
                    except Exception as e3:
                        print(f"   Strategy 3 failed: {type(e3).__name__}")
                        raise e1
                        
        except Exception as e:
            print(f"❌ FAILED to load {object_type}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

        # Set class names
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
            raise ValueError(f"Model not loaded for {self.object_type}")

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
    print("\n" + "="*60)
    print("🔄 LOADING MODELS")
    print("="*60)
    
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
            print(f"   Size: {file_size:,} bytes")
            
            if file_size < 1000:
                print(f"   ⚠️ File seems corrupted (too small)")
                continue
                
            classifier = IronWaterClassifier(obj, path)
            if classifier.model is not None:
                classifiers[obj] = classifier
                print(f"   ✅ {obj} classifier ready")
            else:
                print(f"   ❌ {obj} classifier failed to load")
        else:
            print(f"   ❌ Model file not found: {path}")

    print("\n" + "="*60)
    if classifiers:
        print(f"✅ Successfully loaded models: {list(classifiers.keys())}")
    else:
        print(f"❌ NO MODELS LOADED! Check paths and file integrity.")
        print(f"⚠️ If model files are corrupted, you may need to re-upload them.")
    print("="*60 + "\n")

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
    print("\n" + "="*60)
    print("🚀 STARTING ORION APP")
    print("="*60)

    load_models()

    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'  # Listen on all interfaces for Railway
    
    print(f"🌐 Server running on http://0.0.0.0:{port}")
    app.run(host=host, port=port, debug=False)
