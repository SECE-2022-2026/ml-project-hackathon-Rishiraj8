from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename

# Define the CNN model class
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'crop_classifier_model.pth')

# Load the model with better error handling
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model_info = torch.load(model_path, weights_only=True)
    model = CNNClassifier(model_info['num_classes'])
    model.load_state_dict(model_info['model_state_dict'])
    model.eval()
    label_map = model_info['label_map']
    idx_to_class = {v: k for k, v in label_map.items()}
    print("Model loaded successfully!")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the model file exists in the correct location")
    model, label_map, idx_to_class = None, None, None
except Exception as e:
    print(f"Error loading model: {e}")
    model, label_map, idx_to_class = None, None, None

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Open and transform the image
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = outputs.max(1)
            prediction = idx_to_class[predicted.item()]
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence = confidence[predicted].item() * 100

        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': f'{confidence:.2f}%',
            'image_path': f'uploads/{filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()