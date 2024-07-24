from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import joblib
import librosa
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained model from the notebook
model = joblib.load('parkinsons_detector.pkl')

# Function to extract voice features
def extract_voice_features(file_path):
    # Load the audio file (reduced to 3 seconds)
    y, sr = librosa.load(file_path, duration=3)
    
    # Extract pitch (fundamental frequency)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0]
    f0 = np.mean(pitches) if len(pitches) > 0 else 0
    fhi = np.max(pitches) if len(pitches) > 0 else 0
    flo = np.min(pitches) if len(pitches) > 0 else 0

    # Calculate jitter and shimmer (simplified)
    pitch_periods = librosa.feature.zero_crossing_rate(y)[0]
    amplitude_envelope = np.abs(librosa.stft(y))
    
    jitter_percent = np.std(pitch_periods) / np.mean(pitch_periods) if len(pitch_periods) > 0 else 0
    jitter_abs = np.std(pitch_periods) if len(pitch_periods) > 0 else 0
    shimmer = np.std(amplitude_envelope) / np.mean(amplitude_envelope) if np.mean(amplitude_envelope) > 0 else 0
    
    # Simplified calculations for other measures
    hnr = np.mean(librosa.feature.spectral_flatness(y=y))
    nhr = 1 / hnr if hnr > 0 else 0

    return {
        'MDVP:Fo(Hz)': f0,
        'MDVP:Fhi(Hz)': fhi,
        'MDVP:Flo(Hz)': flo,
        'MDVP:Jitter(%)': jitter_percent,
        'MDVP:Jitter(Abs)': jitter_abs,
        'MDVP:RAP': jitter_percent,  # Approximation
        'MDVP:PPQ': jitter_percent,  # Approximation
        'Jitter:DDP': jitter_percent,  # Approximation
        'MDVP:Shimmer': shimmer,
        'MDVP:Shimmer(dB)': 20 * np.log10(shimmer) if shimmer > 0 else 0,
        'Shimmer:APQ3': shimmer,  # Approximation
        'Shimmer:APQ5': shimmer,  # Approximation
        'MDVP:APQ': shimmer,  # Approximation
        'Shimmer:DDA': shimmer,  # Approximation
        'NHR': nhr,
        'HNR': hnr,
        'RPDE': np.mean(librosa.feature.rms(y=y)),  # Approximation
        'DFA': np.mean(np.abs(np.diff(y))),  # Simplified approximation
        'spread1': np.std(pitches) / np.mean(pitches) if len(pitches) > 0 else 0,
        'spread2': (np.percentile(pitches, 75) - np.percentile(pitches, 25)) / np.median(pitches) if len(pitches) > 0 else 0,
        'D2': np.mean(np.abs(np.diff(np.diff(y)))),  # Simplified approximation
        'PPE': np.std(pitches) / f0 if f0 > 0 else 0  # Approximation
    }

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle audio file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        # Extract features from the audio file
        features = extract_voice_features(file_path)
        logging.debug(f"Extracted features: {features}")
        
        # Convert features to DataFrame
        df_features = pd.DataFrame([features])
        
        # Ensure all features are float
        df_features = df_features.astype(float)
        logging.debug(f"Feature names: {df_features.columns}")
        logging.debug(f"Feature values: {df_features.values}")
        
        # Scale features if necessary
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_features)
        logging.debug(f"Scaled features: {features_scaled}")
        
        # Reshape features to match model input
        features_for_model = features_scaled.reshape(1, -1)
        
        # Get decision function scores
        decision_scores = model.decision_function(features_for_model)
        logging.debug(f"Decision scores: {decision_scores}")
        
        # Make prediction based on decision score
        threshold = 0  # The default threshold for SVM is 0
        prediction = (decision_scores > threshold).astype(int)
        logging.debug(f"Raw prediction: {prediction}")
        if prediction== 1:
            prediction_text = "Parkinson's Disease"
        elif prediction==0:
            prediction_text = "Healthy"
        
        logging.debug(f"Final prediction: {prediction_text}")
        
        # Remove the uploaded file after processing
        os.remove(file_path)
        
        # Render the prediction result template
        return render_template('prediction_result.html', prediction=prediction_text)
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        # If an error occurs, remove the file and return an error message
        os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
