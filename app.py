from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import joblib
import librosa
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained model from the notebook
model = joblib.load('parkinsons_detector.pkl')

# Function to extract voice features
def extract_voice_features(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    fo = np.mean(np.diff(np.unwrap(np.angle(signal))))


    # 2. MDVP:Fhi(Hz)

    fhi = np.max(np.diff(np.unwrap(np.angle(signal))))


    # 3. MDVP:Flo(Hz)

    flo = np.min(np.diff(np.unwrap(np.angle(signal))))


    # 4. MDVP:Jitter(%)

    jitter_percent = np.mean(np.abs(np.diff(np.diff(np.unwrap(np.angle(signal)))))) / fo * 100


    # 5. MDVP:Jitter(Abs)

    jitter_abs = np.mean(np.abs(np.diff(np.diff(np.unwrap(np.angle(signal))))))


    # 6. MDVP:RAP

    rap = np.mean(np.abs(np.diff(np.unwrap(np.angle(signal))))) / fo


    # 7. MDVP:PPQ

    ppq = np.mean(np.abs(np.diff(np.unwrap(np.angle(signal))))) / np.mean(np.abs(signal))


    # 8. Jitter:DDP

    ddp = np.mean(np.abs(np.diff(np.diff(np.unwrap(np.angle(signal)))))) / np.mean(np.abs(np.diff(np.unwrap(np.angle(signal)))))


    # 9. MDVP:Shimmer

    shimmer = np.mean(np.abs(np.diff(signal))) / np.mean(np.abs(signal))


    # 10. MDVP:Shimmer(dB)

    shimmer_db = 20 * np.log10(np.mean(np.abs(np.diff(signal))) / np.mean(np.abs(signal)))


    # 11. Shimmer:APQ3

    apq3 = np.mean(np.abs(np.diff(signal, n=3))) / np.mean(np.abs(signal))


    # 12. Shimmer:APQ5

    apq5 = np.mean(np.abs(np.diff(signal, n=5))) / np.mean(np.abs(signal))


    # 13. MDVP:APQ

    apq = np.mean(np.abs(np.diff(signal))) / np.mean(np.abs(signal))


    # 14. Shimmer:DDA

    dda = np.mean(np.abs(np.diff(signal, n=2))) / np.mean(np.abs(signal))


    # 15. NHR

    nhr = np.mean(np.abs(signal)) / np.max(np.abs(signal))


    # 16. HNR

    hnr = np.mean(np.abs(signal)) / np.std(signal)


    # 17. status (not calculated, as it's a label)

    status = None


    # 18. RPDE

    rpde = np.mean(np.abs(np.diff(signal))) / np.mean(np.abs(signal))


    # 19. DFA

    dfa = np.mean(np.abs(np.diff(signal, n=2))) / np.mean(np.abs(signal))


    # 20. spread1

    spread1 = np.std(signal)


    # 21. spread2

    spread2 = np.std(np.diff(signal))


    # 22. D2

    d2 = np.mean(np.abs(np.diff(signal, n=2))) / np.mean(np.abs(signal))


    # 23. PPE

    ppe = np.mean(np.abs(np.diff(signal, n=3))) / np.mean(np.abs(signal))


    features = pd.DataFrame({

        'MDVP:Fo(Hz)': [fo],

        'MDVP:Fhi(Hz)': [fhi],

        'MDVP:Flo(Hz)': [flo],

        'MDVP:Jitter(%)': [jitter_percent],

        'MDVP:Jitter(Abs)': [jitter_abs],

        'MDVP:RAP': [rap],

        'MDVP:PPQ': [ppq],

        'Jitter:DDP': [ddp],

        'MDVP:Shimmer': [shimmer],

        'MDVP:Shimmer(dB)': [shimmer_db],

        'Shimmer:APQ3': [apq3],

        'Shimmer:APQ5': [apq5],

        'MDVP:APQ': [apq],

        'Shimmer:DDA': [dda],

        'NHR': [nhr],

        'HNR': [hnr],

        'RPDE': [rpde],

        'DFA': [dfa],

        'spread1': [spread1],

        'spread2': [spread2],

        'D2': [d2],

        'PPE': [ppe]

    })


    return features

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
    
    # Extract features from the audio file
    features = extract_voice_features(file_path)
    
    # Convert features to DataFrame
    df_features = pd.DataFrame([features])
    
    # Handle NaN values with imputation
    imputer = SimpleImputer(strategy='mean')
    df_features_imputed = pd.DataFrame(imputer.fit_transform(df_features))
    
    # Reshape features to match model input
    features_for_model = np.array(df_features_imputed.values).reshape(1, -1)
    
    # Handle edge case: check if there are still NaN values
    if np.isnan(features_for_model).any():
        return jsonify({'error': 'NaN values found in features'}), 400
    
    # Predict using the loaded model
    prediction = model.predict(features_for_model)
    
    # Remove the uploaded file after processing
    os.remove(file_path)
    
    # Prepare prediction result for rendering in a new template
    if prediction[0] == 1:
        prediction_text = "Parkinson's Disease"
    elif prediction[0] == 0 :
        prediction_text = "Healthy"
    else:
        prediction_text = "Unknown"
    
    # Render a new template with the prediction result
    return render_template('prediction_result.html', prediction=prediction_text)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
