import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Load Assets ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Load Metadata
with open(os.path.join(MODEL_DIR, 'metadata.pkl'), 'rb') as f:
    metadata = pickle.load(f)

# Load Scaler
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
scaler = None
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

# Load Models
models = {}
for model_name in metadata['model_names']:
    with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), 'rb') as f:
        models[model_name] = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Defaults
    prediction_result = None
    probs = [0, 0, 0] # [Gold, Silver, Bronze]
    probs_formatted = {} # For text display
    current_values = metadata['stats']['mean'].copy()
    
    if request.method == 'POST':
        try:
            # 1. Collect Input
            input_data = []
            current_values = {}
            for feature in metadata['feature_names']:
                val = float(request.form.get(feature))
                input_data.append(val)
                current_values[feature] = val

            # 2. Preprocess
            features_array = np.array([input_data])
            if scaler:
                features_array = scaler.transform(features_array)

            # 3. Predict & Probabilities
            selected_model = request.form['model_choice']
            model = models[selected_model]
            
            # Prediction
            pred_class = model.predict(features_array)[0]
            prediction_result = metadata['tier_mapping'].get(pred_class, "Unknown")

            # Confidence (Probability)
            if hasattr(model, "predict_proba"):
                raw_probs = model.predict_proba(features_array)[0]
                probs = raw_probs.tolist()
                
                # Format for display (e.g., "85.5%")
                probs_formatted = {
                    'Gold': f"{raw_probs[0]*100:.1f}%",
                    'Silver': f"{raw_probs[1]*100:.1f}%",
                    'Bronze': f"{raw_probs[2]*100:.1f}%"
                }
            else:
                probs = [0, 0, 0]
                probs[pred_class] = 1
                probs_formatted = {'Gold': '0%', 'Silver': '0%', 'Bronze': '0%'}
                probs_formatted[metadata['tier_mapping'][pred_class].split()[0]] = '100%'

        except Exception as e:
            prediction_result = f"Error: {str(e)}"

    return render_template('predict.html', 
                           feature_names=metadata['feature_names'], 
                           model_names=metadata['model_names'],
                           stats=metadata['stats'],
                           presets=metadata['presets'],
                           current_values=current_values,
                           prediction=prediction_result,
                           probs=probs,
                           probs_formatted=probs_formatted)

@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')

if __name__ == "__main__":
    app.run(debug=True)