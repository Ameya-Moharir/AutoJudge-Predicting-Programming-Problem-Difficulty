"""
Flask web application for AutoJudge
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import numpy as np
import joblib
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_preprocessor import TextPreprocessor
from src.utils.logger import logger

app = Flask(__name__)
CORS(app)

# Global variables for models
classifier = None
regressor = None
feature_extractor = None
scaler = None
preprocessor = None

# Model loading status
models_loaded = False
load_error = None


def load_models():
    """Load all trained models"""
    global classifier, regressor, feature_extractor, scaler, preprocessor, models_loaded, load_error
    
    try:
        model_dir = Path(__file__).parent.parent / 'models'
        
        logger.info("Loading models...")
        
        # Check if models exist
        required_files = ['classifier.pkl', 'regressor.pkl', 'feature_extractor.pkl', 'scaler.pkl']
        missing_files = [f for f in required_files if not (model_dir / f).exists()]
        
        if missing_files:
            load_error = f"Missing model files: {missing_files}. Please train models first using: python scripts/train_models.py --quick"
            logger.error(load_error)
            return False
        
        # Load models
        classifier = joblib.load(model_dir / 'classifier.pkl')
        regressor = joblib.load(model_dir / 'regressor.pkl')
        feature_extractor = joblib.load(model_dir / 'feature_extractor.pkl')
        scaler = joblib.load(model_dir / 'scaler.pkl')
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        models_loaded = True
        logger.info("Models loaded successfully!")
        return True
        
    except Exception as e:
        load_error = f"Error loading models: {str(e)}"
        logger.error(load_error)
        logger.error(traceback.format_exc())
        return False


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', models_loaded=models_loaded, load_error=load_error)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Check if models are loaded
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded. Please train models first.',
                'details': load_error
            }), 500
        
        # Get input data
        data = request.get_json()
        
        title = data.get('title', '')
        description = data.get('description', '')
        input_desc = data.get('input_description', '')
        output_desc = data.get('output_description', '')
        
        # Validate input
        if not description:
            return jsonify({'error': 'Problem description is required'}), 400
        
        # Combine fields
        combined_text = preprocessor.combine_fields(
            title=title,
            description=description,
            input_desc=input_desc,
            output_desc=output_desc
        )
        
        # Preprocess
        processed_text = preprocessor.preprocess(combined_text)
        
        # Extract features
        features = feature_extractor.transform([processed_text])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        predicted_class = classifier.predict(features_scaled)[0]
        class_probabilities = classifier.predict_proba(features_scaled)[0]
        predicted_score = float(regressor.predict(features_scaled)[0])
        
        # Get class names
        if hasattr(classifier, 'classes_'):
            class_names = classifier.classes_
        else:
    #        Fallback to alphabetical order (what LabelEncoder uses)
            class_names = ['easy', 'hard', 'medium']  # Alphabetical!
        
        # Prepare response
        response = {
            'difficulty_class': predicted_class,
            'difficulty_score': round(predicted_score, 2),
            'confidence': {
                'easy': round(float(class_probabilities[0]), 4),
                'medium': round(float(class_probabilities[1]), 4),
                'hard': round(float(class_probabilities[2]), 4)
            },
            'max_confidence': round(float(max(class_probabilities)), 4)
        }
        
        logger.info(f"Prediction: {predicted_class} (score: {predicted_score:.2f})")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if models_loaded else 'models not loaded',
        'models_loaded': models_loaded
    })


@app.route('/examples')
def examples():
    """Get example problems"""
    examples = [
        {
            'title': 'Sum of Array',
            'description': 'Given an array of integers, find the sum of all elements.',
            'input_description': 'First line contains n (number of elements). Second line contains n space-separated integers.',
            'output_description': 'Print a single integer representing the sum of all elements.'
        },
        {
            'title': 'Rotate Image',
            'description': 'You are given an n x n 2D matrix representing an image. Rotate the image by 90 degrees (clockwise). You must do this in-place or print the new matrix directly.',
            'input_description': 'First line contains n (matrix size). The next n lines contain n space-separated integers representing the rows of the matrix.',
            'output_description': 'Print the rotated matrix, with each row on a new line and elements separated by spaces.'
        },
        {
            'title': 'Shortest Path Visiting All Nodes (TSP)',
            'description': 'Given a weighted graph with N nodes, find the shortest path cost to visit every node exactly once and return to the starting node. Use Bitmask Dynamic Programming.',
            'input_description': 'First line contains N (nodes) and M (edges). The next M lines contain three integers u, v, w denoting a weighted edge between u and v with weight w.',
            'output_description': 'Print the minimum weight required to visit all nodes. If it is impossible, print -1.'
        }
    ]
    
    return jsonify(examples)


if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run app
    port = 5000
    logger.info(f"Starting Flask app on port {port}...")
    logger.info("Open http://localhost:5000 in your browser")
    
    app.run(host='0.0.0.0', port=port, debug=False)
