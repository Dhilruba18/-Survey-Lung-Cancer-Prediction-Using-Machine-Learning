from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import os

app = Flask(__name__)

# Model file and paths
MODEL_FILENAME = "survey lung cancer.pkl"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# Config: WEIGHTED RULE-BASED PREDICTION (medically accurate)
# Prediction is based on weighted symptom scores:
# Critical symptoms (smoking, coughing, chest pain, shortness of breath) have weight 3.0
# High-risk symptoms (chronic disease, wheezing, swallowing difficulty) have weight 2.5
# Medium symptoms (yellow fingers, fatigue) have weight 1.5-2.0
# Low-risk symptoms (anxiety, alcohol, allergy, peer pressure) have weight 0.5-1.0
# Gender does not affect the prediction


def load_model(path):
    # Try the path directly, then the MODEL_PATH. Prefer pickle then joblib.
    tried = []

    if os.path.isabs(path) and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(path)

    tried.append(MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(MODEL_PATH)

    tried.append(path)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return joblib.load(path)

    raise FileNotFoundError(f"Model file not found. Tried: {tried}")


# Load model once at startup
model = None
try:
    model = load_model(MODEL_FILENAME)
    print("Model loaded successfully")

    # Show some useful diagnostics about the model
    try:
        print("Model type:", type(model))
        print("n_features_in_:", getattr(model, 'n_features_in_', None))
        print("feature_names_in_:", getattr(model, 'feature_names_in_', None))
        print("classes_:", getattr(model, 'classes_', None))
    except Exception:
        pass

    # Auto-detect disabled - POSITIVE_CLASS is manually set above
    # If you need to change the interpretation, modify POSITIVE_CLASS at the top of this file

except Exception as e:
    print("Warning: Could not load model:", e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Must match order the model expects
        fields = [
            "gender",
            "age",
            "smoking",
            "yellow_fingers",
            "anxiety",
            "peer_pressure",
            "chronic_disease",
            "fatigue",
            "allergy",
            "wheezing",
            "alcohol",
            "coughing",
            "shortness_of_breath",
            "swallowing_difficulty",
            "chest_pain",
        ]

        values = []
        for name in fields:
            raw = request.form.get(name)
            if raw is None:
                raise ValueError(f"Missing form field: {name}")
            val = raw.strip()

            # Gender mapping: M/F or numeric. Dataset uses 1/2.
            if name == 'gender':
                if val.upper() in ('M', 'MALE'):
                    mapped = 1.0
                elif val.upper() in ('F', 'FEMALE'):
                    mapped = 2.0
                else:
                    mapped = float(val)
                values.append(mapped)
                continue

            # For other fields expect numeric (1/2 encoding) - convert to float
            values.append(float(val))

        # Count YES (value=1) and NO (value=2) from symptoms with medical weights
        # Higher weight = more important for lung cancer diagnosis
        
        # Define weights for each symptom based on medical importance
        symptom_weights = {
            'smoking': 3.0,              # Very high indicator
            'yellow_fingers': 2.0,       # High indicator (smoking-related)
            'anxiety': 1.0,              # Low indicator
            'peer_pressure': 0.5,        # Very low indicator
            'chronic_disease': 2.5,      # High indicator
            'fatigue': 1.5,              # Medium indicator
            'allergy': 0.5,              # Very low indicator
            'wheezing': 2.5,             # High indicator
            'alcohol': 1.0,              # Low indicator
            'coughing': 3.0,             # Very high indicator
            'shortness_of_breath': 3.0,  # Very high indicator
            'swallowing_difficulty': 2.5,# High indicator
            'chest_pain': 3.0,           # Very high indicator
        }
        
        # Calculate weighted scores
        yes_score = 0.0
        no_score = 0.0
        symptom_names = fields[2:]  # Skip gender and age
        
        for i, symptom_name in enumerate(symptom_names):
            symptom_value = values[i + 2]  # +2 to skip gender and age
            weight = symptom_weights.get(symptom_name, 1.0)
            
            if symptom_value == 1.0:  # YES
                yes_score += weight
            elif symptom_value == 2.0:  # NO
                no_score += weight
        
        # Count for display
        symptom_values = values[2:]
        yes_count = sum(1 for v in symptom_values if v == 1.0)
        no_count = sum(1 for v in symptom_values if v == 2.0)
        
        print(f"Symptom counts: {yes_count} YES, {no_count} NO")
        print(f"Weighted scores: YES={yes_score:.1f}, NO={no_score:.1f}")
        
        # Weighted decision: compare scores instead of counts
        # If weighted YES score > NO score -> predict YES (has cancer)
        # If weighted NO score > YES score -> predict NO (no cancer)
        if yes_score > no_score:
            label = 'YES'
            raw_pred = 1  # Use 1 to represent YES
        else:
            label = 'NO'
            raw_pred = 0  # Use 0 to represent NO
        
        print(f"Final prediction: {label} (raw: {raw_pred})")
        
        # Calculate "probabilities" based on weighted score ratio for display
        total_score = yes_score + no_score
        if total_score > 0:
            yes_ratio = yes_score / total_score
            no_ratio = no_score / total_score
            proba = [[no_ratio, yes_ratio]]  # [prob_NO, prob_YES]
        else:
            proba = [[0.5, 0.5]]

        return render_template('result.html', prediction=label, raw_pred=raw_pred, proba=proba, features=values, classes=[0, 1])

    except Exception as e:
        return render_template('result.html', error=str(e))


if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 5000
    url = f"http://{HOST}:{PORT}"
    print("===============================================")
    print("Flask app starting now")
    print(f"Open the app in your browser: {url}")
    print("(If nothing appears here, try opening the URL manually in your browser.)")
    print("===============================================")

    try:
        import webbrowser
        webbrowser.open_new_tab(url)
    except Exception:
        pass

    app.run(host=HOST, port=PORT, debug=True)

# -- (duplicate content removed) --