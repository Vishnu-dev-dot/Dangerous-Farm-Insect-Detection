import os
import numpy as np

# DJANGO / TF
from django.conf import settings

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
except Exception:
    load_model = None
    keras_image = None

# -----------------------------
# CONSTANTS (unchanged content)
# -----------------------------
CLASS_NAMES = [
    'Africanized Honey Bees (Killer Bees)',
    'Aphids',
    'Armyworms',
    'Brown Marmorated Stink Bugs',
    'Cabbage Loopers',
    'Citrus Canker',
    'Colorado Potato Beetles',
    'Corn Borers',
    'Corn Earworms',
    'Fall Armyworms',
    'Fruit Flies',
    'Spider Mites',
    'Thrips',
    'Tomato Hornworms',
    'Western Corn Rootworms'
]

PESTICIDE_MAP = {
    'Africanized Honey Bees (Killer Bees)': 'Caution: Do NOT spray near bees. Use mechanical controls or contact pest control professionals. If control is required, use targeted baits placed away from flowering plants.',
    'Aphids': 'Neem oil 2% foliar spray or Imidacloprid systemic (follow label). Typical: 2 ml neem oil / liter; repeat every 7-10 days as needed.',
    'Armyworms': 'Spinosad or Bacillus thuringiensis (Bt) spray. Example: Spinosad 0.5-1 ml/liter, follow label.',
    'Brown Marmorated Stink Bugs': 'Carbaryl or Pyrethroid sprays for heavy infestations (follow label). Use trap monitoring first.',
    'Cabbage Loopers': 'Bacillus thuringiensis (Bt) spray or Spinosad. Example: Bt as directed on product label.',
    'Citrus Canker': 'Cultural controls and copper-based bactericide sprays; follow agricultural extension guidance.',
    'Colorado Potato Beetles': 'Use systemic insecticides or spinosad; handpick adults/larvae. Example: Spinosad per label instructions.',
    'Corn Borers': 'Use Bacillus thuringiensis varieties or recommended pyrethroid sprays in heavy infestations; follow label.',
    'Corn Earworms': 'Pyrethroid or Bacillus thuringiensis (Bt) sprays; pheromone traps for monitoring.',
    'Fall Armyworms': 'Spinosad or lambda-cyhalothrin following label; apply early in infestation.',
    'Fruit Flies': 'Use bait traps, protein bait sprays or spinosad-based baiting; follow product label.',
    'Spider Mites': 'Use miticides such as abamectin or horticultural oils; example: horticultural oil spray, follow label.',
    'Thrips': 'Use spinosad or insecticidal soaps; consider thrips-specific control measures.',
    'Tomato Hornworms': 'Bacillus thuringiensis (Bt) or handpicking recommended; Bt per label.',
    'Western Corn Rootworms': 'Crop rotation, soil insecticides or seed treatments; follow integrated pest management practices.'
}

CONCENTRATION_MAP_ML_PER_LITER = {
    'Africanized Honey Bees (Killer Bees)': 0.0,
    'Aphids': 2.0,
    'Armyworms': 1.0,
    'Brown Marmorated Stink Bugs': 2.0,
    'Cabbage Loopers': 1.0,
    'Citrus Canker': 10.0,
    'Colorado Potato Beetles': 1.5,
    'Corn Borers': 1.0,
    'Corn Earworms': 1.0,
    'Fall Armyworms': 1.0,
    'Fruit Flies': 1.0,
    'Spider Mites': 2.0,
    'Thrips': 1.0,
    'Tomato Hornworms': 1.0,
    'Western Corn Rootworms': 2.0
}

MODEL_PATH = 'hybrid_model.keras'

# -----------------------------
# MODEL CACHE / INPUT SIZE
# -----------------------------
_model = None
_input_size = (224, 224)
_expects_two_inputs = False


def _detect_input(model_obj):
    """
    Detect expected input shape and whether the model is a hybrid that expects two inputs.
    """
    global _input_size, _expects_two_inputs
    shape = model_obj.input_shape
    if isinstance(shape, list):
        _expects_two_inputs = len(shape) == 2
        # Use the first input to infer size
        _, h, w, _ = shape[0]
    else:
        _expects_two_inputs = False
        _, h, w, _ = shape
    _input_size = (int(h) if h else 224, int(w) if w else 224)


def get_model():
    """Load the Keras model once and reuse it."""
    global _model
    if _model is not None:
        return _model
    if load_model is None:
        raise RuntimeError("TensorFlow is not available. Please install tensorflow.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    _model = load_model(MODEL_PATH)
    _detect_input(_model)
    return _model


# -----------------------------
# IMAGE PIPELINE
# -----------------------------
def _load_rgb(img_path):
    """
    Load image in RGB with keras utilities.
    (We don’t import PIL directly to keep your stack consistent.)
    """
    if keras_image is None:
        raise RuntimeError("Keras image utilities not available (tensorflow missing).")
    # ensure model is loaded to have _input_size
    get_model()
    img = keras_image.load_img(img_path, target_size=_input_size)
    x = keras_image.img_to_array(img).astype('float32') / 255.0
    return np.expand_dims(x, axis=0)


def predict_from_path(img_path):
    """
    Predict on a single image path. Returns:
    { 'label': str, 'confidence': float, 'pesticide': str }
    """
    model = get_model()
    x = _load_rgb(img_path)

    # Handle hybrid models that expect two inputs
    if _expects_two_inputs:
        preds = model.predict([x, x])
    else:
        preds = model.predict(x)

    # If model returns list, take the first output
    if isinstance(preds, list):
        preds = preds[0]

    preds = np.asarray(preds).ravel()
    idx = int(np.argmax(preds))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f'class_{idx}'
    confidence = float(preds[idx])
    pesticide = PESTICIDE_MAP.get(label, 'No recommendation available — update PESTICIDE_MAP')

    return {'label': label, 'confidence': confidence, 'pesticide': pesticide}


# -----------------------------
# PESTICIDE CALCULATIONS
# -----------------------------
def calculate_pesticide_for_area(area_sqft, insect_label):
    """
    Compute liters and total pesticide (ml) for a given area and class.
    Assumptions:
      - 1 L covers 1000 sq.ft (adjust as needed)
      - ml/L comes from CONCENTRATION_MAP_ML_PER_LITER
    """
    coverage_per_liter_sqft = 1000.0
    liters_needed = area_sqft / coverage_per_liter_sqft
    ml_per_liter = CONCENTRATION_MAP_ML_PER_LITER.get(insect_label, 2.0)
    pesticide_ml_total = liters_needed * ml_per_liter
    return {
        'area': area_sqft,
        'liters_needed': round(liters_needed, 4),
        'pesticide_ml_total': round(pesticide_ml_total, 4),
        'insect_label': insect_label,
        'ml_per_liter': ml_per_liter,
        'notes': 'Assumptions: 1 L covers 1000 sq.ft. Concentrations are example values—follow product label.'
    }


def calculate_all_pesticides(area_sqft):
    """
    Helper to build the full table for ALL classes for a given area.
    Returns a list of dicts compatible with your calculator table.
    """
    rows = []
    for label in CLASS_NAMES:
        rows.append(calculate_pesticide_for_area(area_sqft, label))
    return rows


# import os
# import numpy as np
# from django.conf import settings
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image as keras_image

# try:
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras.preprocessing import image as keras_image
# except Exception as e:
#     load_model = None
#     keras_image = None

# CLASS_NAMES = [
#  'Africanized Honey Bees (Killer Bees)',
#  'Aphids',
#  'Armyworms',
#  'Brown Marmorated Stink Bugs',
#  'Cabbage Loopers',
#  'Citrus Canker',
#  'Colorado Potato Beetles',
#  'Corn Borers',
#  'Corn Earworms',
#  'Fall Armyworms',
#  'Fruit Flies',
#  'Spider Mites',
#  'Thrips',
#  'Tomato Hornworms',
#  'Western Corn Rootworms'
# ]

# PESTICIDE_MAP = {
#  'Africanized Honey Bees (Killer Bees)': 'Caution: Do NOT spray near bees. Use mechanical controls or contact pest control professionals. If control is required, use targeted baits placed away from flowering plants.',
#  'Aphids': 'Neem oil 2% foliar spray or Imidacloprid systemic (follow label). Typical: 2 ml neem oil / liter; repeat every 7-10 days as needed.',
#  'Armyworms': 'Spinosad or Bacillus thuringiensis (Bt) spray. Example: Spinosad 0.5-1 ml/liter, follow label.',
#  'Brown Marmorated Stink Bugs': 'Carbaryl or Pyrethroid sprays for heavy infestations (follow label). Use trap monitoring first.',
#  'Cabbage Loopers': 'Bacillus thuringiensis (Bt) spray or Spinosad. Example: Bt as directed on product label.',
#  'Citrus Canker': 'Cultural controls and copper-based bactericide sprays; follow agricultural extension guidance.',
#  'Colorado Potato Beetles': 'Use systemic insecticides or spinosad; handpick adults/larvae. Example: Spinosad per label instructions.',
#  'Corn Borers': 'Use Bacillus thuringiensis varieties or recommended pyrethroid sprays in heavy infestations; follow label.',
#  'Corn Earworms': 'Pyrethroid or Bacillus thuringiensis (Bt) sprays; pheromone traps for monitoring.',
#  'Fall Armyworms': 'Spinosad or lambda-cyhalothrin following label; apply early in infestation.',
#  'Fruit Flies': 'Use bait traps, protein bait sprays or spinosad-based baiting; follow product label.',
#  'Spider Mites': 'Use miticides such as abamectin or horticultural oils; example: horticultural oil spray, follow label.',
#  'Thrips': 'Use spinosad or insecticidal soaps; consider thrips-specific control measures.',
#  'Tomato Hornworms': 'Bacillus thuringiensis (Bt) or handpicking recommended; Bt per label.',
#  'Western Corn Rootworms': 'Crop rotation, soil insecticides or seed treatments; follow integrated pest management practices.'
# }

# CONCENTRATION_MAP_ML_PER_LITER = {
#  'Africanized Honey Bees (Killer Bees)': 0.0,
#  'Aphids': 2.0,
#  'Armyworms': 1.0,
#  'Brown Marmorated Stink Bugs': 2.0,
#  'Cabbage Loopers': 1.0,
#  'Citrus Canker': 10.0,
#  'Colorado Potato Beetles': 1.5,
#  'Corn Borers': 1.0,
#  'Corn Earworms': 1.0,
#  'Fall Armyworms': 1.0,
#  'Fruit Flies': 1.0,
#  'Spider Mites': 2.0,
#  'Thrips': 1.0,
#  'Tomato Hornworms': 1.0,
#  'Western Corn Rootworms': 2.0
# }

# MODEL_PATH = 'hybrid_model.keras'


# model = load_model(MODEL_PATH)

# _model = None
# _input_size = (224, 224)


# def get_model():
#     """Load the Keras model once and reuse it."""
#     global _model, _input_size
#     if _model is not None:
#         return _model
#     if load_model is None:
#         raise RuntimeError("TensorFlow not available. Install tensorflow.")
#     if not os.path.exists(MODEL_PATH):
#         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
#     _model = load_model(MODEL_PATH)
#     # detect input size
#     shape = _model.input_shape
#     if isinstance(shape, list):
#         # take first input's shape
#         _, h, w, c = shape[0]
#         _input_size = (int(h) if h else 224, int(w) if w else 224)
#     else:
#         _, h, w, c = shape
#         _input_size = (int(h) if h else 224, int(w) if w else 224)
#     return _model


# def preprocess_image_from_path(img_path):
#     """Load and normalize an image as float32 RGB."""
#     if keras_image is None:
#         raise RuntimeError("Keras image utilities not available (tensorflow missing)")
#     model = get_model()  # ensure model loaded and _input_size set
#     img = keras_image.load_img(img_path, target_size=_input_size)
#     x = keras_image.img_to_array(img)
#     x = x.astype('float32') / 255.0
#     x = np.expand_dims(x, axis=0)
#     return x


# def predict_from_path(img_path):
#     """Predict on a single RGB image."""
#     from tensorflow.keras.models import load_model
#     import numpy as np
#     from tensorflow.keras.preprocessing import image as keras_image

#     global model  # use the already loaded global model
#     if model is None:
#         model = load_model('hybrid_model.keras')

#     x = keras_image.load_img(img_path, target_size=(224, 224))
#     x = keras_image.img_to_array(x)
#     x = x.astype('float32') / 255.0
#     x = np.expand_dims(x, axis=0)

#     # handle hybrid model 2-input
#     if isinstance(model.input_shape, list) and len(model.input_shape) == 2:
#         preds = model.predict([x, x])
#     else:
#         preds = model.predict(x)

#     if isinstance(preds, list):
#         preds = preds[0]

#     preds = np.asarray(preds).ravel()
#     idx = int(np.argmax(preds))
#     label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f'class_{idx}'
#     confidence = float(preds[idx])
#     pesticide = PESTICIDE_MAP.get(label, 'No recommendation available — update PESTICIDE_MAP')

#     return {'label': label, 'confidence': confidence, 'pesticide': pesticide}



# def calculate_pesticide_for_area(area_sqft, insect_label):
#     coverage_per_liter_sqft = 1000.0
#     liters_needed = area_sqft / coverage_per_liter_sqft
#     ml_per_liter = CONCENTRATION_MAP_ML_PER_LITER.get(insect_label, 2.0)
#     pesticide_ml_total = liters_needed * ml_per_liter
#     return {
#         'area': area_sqft,
#         'liters_needed': round(liters_needed, 4),
#         'pesticide_ml_total': round(pesticide_ml_total, 4),
#         'insect_label': insect_label,
#         'ml_per_liter': ml_per_liter,
#         'notes': 'Assumptions: 1 L covers 1000 sq.ft. Concentrations are example values—follow product label.'
#     }
