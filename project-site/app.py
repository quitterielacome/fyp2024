from flask import Flask, request, jsonify, render_template, session
import os
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from keras.models import load_model
import cv2
import tempfile
import shutil
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)
# Secret key for session management
app.secret_key = 'MyKey2024FYP'

# Load models 
baseline_model = load_model('model_baseline.h5')
augmented_model = load_model('model_baseline_data_augmentation.h5')
vgg16_model = load_model('model_vgg16.h5')

def load_and_preprocess_dicom(path, size=128):
    """
    Load and preprocess a DICOM image.
    Normalise, resize and reshape the image data for the model.
    """
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    resized_data = cv2.resize(data, (size, size))
    return resized_data.reshape((size, size, 1))  # Only reshape to (32, 32, 1)

def load_and_preprocess_dicom_vgg16(path):
    """
    Load and preprocess a DICOM image specifically for the VGG16 model.
    Converts image to RGB and resizes it to 224x224.
    """
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    data = cv2.resize(data, (224, 224))
    return data.reshape((1, 224, 224, 3))


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return filename.lower().endswith('.dcm')

def process_patient_images(image_paths):
    """Process patient DICOM images and return model predictions and average confidences."""

    images = {
        'baseline': [load_and_preprocess_dicom(path) for path in image_paths],
        'vgg16': [load_and_preprocess_dicom_vgg16(path) for path in image_paths]
    }

    predictions = {
        'baseline': [],
        'augmented': [],
        'vgg16': []
    }

    # Debugging output
    print("Processing baseline and augmented models...")
    # Process baseline and augmented model predictions
    for img in images['baseline']:
        img_batch = np.expand_dims(img, axis=0)  # Ensure the batch dimension is added
        if baseline_model:
            predictions['baseline'].append(baseline_model.predict(img_batch)[0][1])
        if augmented_model:
            predictions['augmented'].append(augmented_model.predict(img_batch)[0][1])

    print("Processing VGG16 model...")
    for img in images['vgg16']:
        if vgg16_model:
            prediction = vgg16_model.predict(img)
            predictions['vgg16'].append(prediction[0][1])
            print(f"VGG16 prediction: {prediction[0][1]}")  # Debugging output

    average_confidence = {
        'baseline': np.mean(predictions['baseline']) * 100 if predictions['baseline'] else 0,
        'augmented': np.mean(predictions['augmented']) * 100 if predictions['augmented'] else 0,
        'vgg16': np.mean(predictions['vgg16']) * 100 if predictions['vgg16'] else 0
    }

    predicted_class = {
        'baseline': "Positive" if average_confidence['baseline'] >= 50 else "Negative",
        'augmented': "Positive" if average_confidence['augmented'] >= 50 else "Negative",
        'vgg16': "Positive" if average_confidence['vgg16'] >= 50 else "Negative"
    }

    # Determine the confidence for the predicted class
    confidence = {
        'baseline': average_confidence['baseline'] if predicted_class['baseline'] == 'Positive' else 100 - average_confidence['baseline'],
        'augmented': average_confidence['augmented'] if predicted_class['augmented'] == 'Positive' else 100 - average_confidence['augmented'],
        'vgg16': average_confidence['vgg16'] if predicted_class['vgg16'] == 'Positive' else 100 - average_confidence['vgg16']
    }

    return predicted_class, confidence



@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle analysis of uploaded DICOM images."""
    patient_name = request.form.get('patientName') or "Unknown Patient"
    print(f"Received patient name: {patient_name}") 
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    temp_dir = tempfile.mkdtemp()
    image_paths = []
    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                image_paths.append(file_path)

        if not image_paths:
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'No valid DICOM files found'}), 400
        folder_name = os.path.basename(os.path.dirname(image_paths[0]))

        print("Processing DICOM files:", image_paths)
        predicted_class, average_confidence = process_patient_images(image_paths)
        processed_count = len(image_paths)
        shutil.rmtree(temp_dir)  # Cleanup

    except Exception as e:
        shutil.rmtree(temp_dir) # Clean up temporary directory
        return jsonify({'error': str(e)}), 500
    
    print("Session History Before Adding New Entry:", session.get('history'))

    # Add the analysis results to the session history
    if 'history' not in session:
        session['history'] = []
    
    session['history'].append({
        "patientName": patient_name,
        "results": [
            {
                "model": "Baseline Model",
                "result": predicted_class['baseline'],
                "confidence": f"{average_confidence['baseline']:.2f}%"
            },
            {
                "model": "Augmented Model",
                "result": predicted_class['augmented'],
                "confidence": f"{average_confidence['augmented']:.2f}%"
            },
            {
                "model": "VGG16 Model",
                "result": predicted_class['vgg16'],
                "confidence": f"{average_confidence['vgg16']:.2f}%"
            }
        ]
    })
    session.modified = True

    return jsonify({
        "processedImages": processed_count,
        "baselineResult": f"{predicted_class['baseline']}, Confidence: {average_confidence['baseline']:.2f}%",
        "augmentedResult": f"{predicted_class['augmented']}, Confidence: {average_confidence['augmented']:.2f}%",
        "vgg16Result": f"{predicted_class['vgg16']}, Confidence: {average_confidence['vgg16']:.2f}%"
    })

@app.route('/history')
def history():
    """Display the session history of analyses."""
    history_data = session.get('history', [])

    # Add debugging print statement
    print("History Data Retrieved for Rendering:", history_data)

    return render_template('history.html', history=history_data)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    """Clear the analysis history from the session."""
    session.pop('history', None)
    return jsonify({'success': 'History has been reset'}), 200

if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True)