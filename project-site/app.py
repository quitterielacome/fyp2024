from flask import Flask, request, jsonify, render_template, session
import os
import numpy as np
import pydicom
from tensorflow.keras.models import load_model
import cv2
import tempfile
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'MyKey2024FYP'

# Load models
baseline_model = load_model('model_baseline.h5')
augmented_model = load_model('model_baseline_data_augmentation.h5')
vgg16_model = load_model('model_vgg16.h5')

def load_and_preprocess_dicom(path, size=128, is_vgg16=False):
    dicom_file = pydicom.dcmread(path)
    data = dicom_file.pixel_array
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    if is_vgg16:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        data = cv2.resize(data, (224, 224))
        return data.reshape((1, 224, 224, 3))
    else:
        data = cv2.resize(data, (size, size))
        return data.reshape((1, size, size, 1))

def predict_with_model(model, image_paths, is_vgg16=False):
    predictions = []
    for path in image_paths:
        preprocessed_image = load_and_preprocess_dicom(path, is_vgg16=is_vgg16)
        prediction = model.predict(preprocessed_image)
        predictions.append(prediction[0])

    average_predictions = np.mean(predictions, axis=0)
    predicted_class_index = np.argmax(average_predictions)
    final_decision = "Positive" if predicted_class_index == 1 else "Negative"
    confidence = max(average_predictions) * 100
    return final_decision, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    patient_name = request.form.get('patientName', "Unknown Patient")
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    temp_dir = tempfile.mkdtemp()
    image_paths = []

    try:
        for file in files:
            if file and file.filename.lower().endswith('.dcm'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                image_paths.append(file_path)

        if not image_paths:
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'No valid DICOM files found'}), 400

        # Process images with each model
        baseline_decision, baseline_confidence = predict_with_model(baseline_model, image_paths)
        augmented_decision, augmented_confidence = predict_with_model(augmented_model, image_paths)
        vgg16_decision, vgg16_confidence = predict_with_model(vgg16_model, image_paths, is_vgg16=True)

        # Cleanup temporary directory
        shutil.rmtree(temp_dir)
    except Exception as e:
        shutil.rmtree(temp_dir)  # Ensure cleanup on error
        return jsonify({'error': str(e)}), 500
    # Update session history
    if 'history' not in session:
        session['history'] = []
    
    session['history'].append({
        "patientName": patient_name,
        "results": [
            {
                "model": "Baseline Model",
                "result": baseline_decision,
                "confidence": f"{baseline_confidence:.2f}%"
            },
            {
                "model": "Augmented Model",
                "result": augmented_decision,
                "confidence": f"{augmented_confidence:.2f}%"
            },
            {
                "model": "VGG16 Model",
                "result": vgg16_decision,
                "confidence": f"{vgg16_confidence:.2f}%"
            }
        ]
    })
    session.modified = True
    return jsonify({
        "patientName": patient_name,
        "processedImages": len(image_paths),
        "baselineResult": f"{baseline_decision}, Confidence: {baseline_confidence:.2f}%",
        "augmentedResult": f"{augmented_decision}, Confidence: {augmented_confidence:.2f}%",
        "vgg16Result": f"{vgg16_decision}, Confidence: {vgg16_confidence:.2f}%"
    })




@app.route('/history')
def history():
    history_data = session.get('history', [])
    return render_template('history.html', history=history_data)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    session.pop('history', None)
    return jsonify({'success': 'History reset successful'}), 200

if __name__ == '__main__':
    app.run(debug=True)
