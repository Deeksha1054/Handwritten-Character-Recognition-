import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

# Define Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

# Define model class (Must match your trained model)
class CNNModel(nn.Module):
    def __init__(self, num_classes=62):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Adjusted for 32x32 input
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load class names
class_names = sorted(os.listdir(r"C:\Users\DELL\OneDrive\Desktop\preprocessed_dataset"))



  # Replace with actual dataset path

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("handwritten_characters_model.pth", map_location=device))
model.eval()  # Set to evaluation mode

# Image preprocessing function
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((32, 32))  # Resize to 32x32
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return torch.tensor(image, dtype=torch.float32).to(device)

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(file)
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return jsonify({"prediction": class_names[predicted_class]})

# Home route to render HTML frontend
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
