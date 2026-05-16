from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import zipfile
import tempfile
import base64
from tensorflow import keras
from tensorflow.keras import layers

# ==================== CONFIG ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== MODEL ARCHITECTURE ====================
# Must exactly match the architecture used during training.

IMG_SIZE = (256, 256)

def build_model(num_classes=2):
    model = keras.Sequential([
        layers.InputLayer(input_shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1./255),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def load_keras2_model(keras_path):
    """
    Load a Keras 2.x .keras file (ZIP containing model.weights.h5) into a
    freshly built Keras 3 model by extracting and loading just the weights.
    """
    # Strategy 1: standard load (works if versions match)
    try:
        model = keras.models.load_model(keras_path)
        print(f"   ✅ Standard load succeeded")
        return model
    except Exception as e:
        print(f"   ⚠️  Standard load failed ({e}), trying weight extraction...")

    # Strategy 2: extract model.weights.h5 from the ZIP and load into fresh model
    try:
        with zipfile.ZipFile(keras_path, 'r') as z:
            names = z.namelist()
            weight_files = [n for n in names if n.endswith('.h5')]
            if not weight_files:
                raise ValueError(f"No .h5 weight file found inside {keras_path}. Contents: {names}")

            weight_entry = weight_files[0]
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                tmp.write(z.read(weight_entry))
                tmp_path = tmp.name

        model = build_model(num_classes=2)
        model.build(input_shape=(None, *IMG_SIZE, 3))
        model.load_weights(tmp_path)
        os.unlink(tmp_path)
        print(f"   ✅ Weight extraction load succeeded")
        return model

    except Exception as e:
        print(f"   ⚠️  Weight extraction failed ({e}), trying by_name=True...")

    # Strategy 3: same but load weights by layer name (more lenient)
    try:
        with zipfile.ZipFile(keras_path, 'r') as z:
            weight_files = [n for n in z.namelist() if n.endswith('.h5')]
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                tmp.write(z.read(weight_files[0]))
                tmp_path = tmp.name

        model = build_model(num_classes=2)
        model.build(input_shape=(None, *IMG_SIZE, 3))
        model.load_weights(tmp_path, by_name=True, skip_mismatch=True)
        os.unlink(tmp_path)
        print(f"   ✅ by_name weight load succeeded")
        return model

    except Exception as e:
        raise RuntimeError(f"All loading strategies failed: {e}")


# ==================== CLASSIFIER ====================

class IronWaterClassifier:
    def __init__(self, object_type, model_path):
        self.object_type = object_type

        print(f"🔄 Loading model: {object_type}")
        try:
            self.model = load_keras2_model(model_path)
            print(f"✅ Model ready: {object_type}")
        except Exception as e:
            print(f"❌ Failed load {object_type}: {e}")
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

        # NOTE: Model has Rescaling(1/255) built in — do NOT divide by 255 here.
        return np.array(img, dtype=np.float32)

    def classify(self, img):
        if self.model is None:
            raise ValueError(f"Model '{self.object_type}' failed to load at startup")

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
    model_files = {
        'orange': os.path.join(MODELS_FOLDER, 'orange_classifier.keras'),
        'banana': os.path.join(MODELS_FOLDER, 'banana_classifier.keras'),
        'egg':    os.path.join(MODELS_FOLDER, 'egg_classifier.keras')
    }

    for obj, path in model_files.items():
        if os.path.exists(path):
            classifiers[obj] = IronWaterClassifier(obj, path)
        else:
            print(f"⚠️  Model file not found: {path}")

    loaded = [k for k, v in classifiers.items() if v.model is not None]
    failed = [k for k, v in classifiers.items() if v.model is None]
    print(f"✅ Models ready: {loaded}")
    if failed:
        print(f"❌ Models failed: {failed}")

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/api/classify-image', methods=['POST'])
def classify_image():
    try:
        object_type = request.form.get('object_type')
        print("Object:", object_type)

        if object_type not in classifiers:
            return jsonify({'error': f"Invalid object type '{object_type}'. Valid: {list(classifiers.keys())}"}), 400

        if classifiers[object_type].model is None:
            return jsonify({'error': f"Model '{object_type}' failed to load at startup"}), 503

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img_bytes = file.read()

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        result = classifiers[object_type].classify(img)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/classify-camera', methods=['POST'])
def classify_camera():
    """
    Classify an image captured from camera using base64 encoding
    """
    try:
        data = request.get_json()
        object_type = data.get('object_type')
        image_data = data.get('image')  # base64 encoded image

        print(f"🎥 Camera classification for: {object_type}")

        if object_type not in classifiers:
            return jsonify({'error': f"Invalid object type '{object_type}'. Valid: {list(classifiers.keys())}"}), 400

        if classifiers[object_type].model is None:
            return jsonify({'error': f"Model '{object_type}' failed to load at startup"}), 503

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        try:
            # Remove the data:image/...;base64, prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({'error': 'Invalid image data'}), 400

            result = classifiers[object_type].classify(img)

            return jsonify({
                'success': True,
                'result': result
            })

        except Exception as e:
            print(f"❌ Image decode error: {e}")
            return jsonify({'error': f"Failed to decode image: {str(e)}"}), 400

    except Exception as e:
        print("❌ CAMERA ERROR:", e)
        return jsonify({'error': str(e)}), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 STARTING APP")
    print("=" * 50)

    load_models()

    print("🌐 Open: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)    print("="*60)
    
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
        self.load_error = None

        # Load model with better error handling
        try:
            print(f"   Loading model from: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            file_size = os.path.getsize(model_path)
            print(f"   File size: {file_size:,} bytes")
            
            if file_size < 1000000:
                raise ValueError(f"Model file too small ({file_size:,} bytes) - likely corrupted")
            
            # Try standard load
            try:
                self.model = keras.models.load_model(model_path, compile=False)
                print(f"✅ Model loaded successfully: {object_type}")
            except Exception as e:
                print(f"⚠️ Model loading failed: {type(e).__name__}: {str(e)[:100]}")
                self.load_error = str(e)
                self.model = None
                        
        except Exception as e:
            print(f"❌ FAILED to load {object_type}: {type(e).__name__}: {e}")
            self.load_error = str(e)
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
            raise ValueError(f"Model not loaded for {self.object_type}. Error: {self.load_error}")

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
            
            classifier = IronWaterClassifier(obj, path)
            if classifier.model is not None:
                classifiers[obj] = classifier
                print(f"   ✅ {obj} classifier ready")
            else:
                print(f"   ⚠️ {obj} classifier loading failed (will return error on use)")
                classifiers[obj] = classifier  # Store even if failed, so we can return proper error
        else:
            print(f"   ❌ Model file not found: {path}")

    print("\n" + "="*60)
    if classifiers:
        loaded_count = sum(1 for c in classifiers.values() if c.model is not None)
        print(f"✅ Successfully loaded {loaded_count}/{len(classifiers)} models")
    else:
        print(f"⚠️ No classifiers initialized")
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
            error_msg = f'Model not found: {object_type}. Available: {list(classifiers.keys())}'
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 400

        classifier = classifiers[object_type]
        if classifier.model is None:
            error_msg = f'Model for {object_type} failed to load. Error: {classifier.load_error}'
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 503

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

        result = classifier.classify(img)
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
    print("⚠️  App will start even if models fail to load")
    app.run(host=host, port=port, debug=False)
