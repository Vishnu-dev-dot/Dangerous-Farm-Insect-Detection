from django.test import TestCase

# Create your tests here.
from tensorflow.keras.models import load_model
import joblib

model = load_model("hybrid_model.keras")

# If you saved your encoder:
label_encoder = joblib.load("label_encoder.pkl")
class_names = label_encoder.classes_.tolist()
print(class_names)
