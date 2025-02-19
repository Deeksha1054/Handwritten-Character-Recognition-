# 📝 Handwritten Character Recognition

Handwritten Character Recognition with Deep Learning and Flask A web-based application that uses a deep learning model to recognize handwritten English characters (0-9, A-Z, a-z) from images. Built with TensorFlow, Flask, and OpenCV, it provides both a web interface and an API endpoint for easy integration.
---

## 🚀 Features
- Upload an image containing a handwritten character.
- Model processes the image and predicts the corresponding character.
- Supports digits (0-9), uppercase (A-Z), and lowercase (a-z) letters.
- Provides an API endpoint for easy integration.
- Built with **Flask**, **pytorch**, and **OpenCV**.

---

## 🏗️ Project Structure
📂 handwritten_character_recognition/
│── 📂 uploads/            # Directory for uploaded images  
│── 📜 app.py              # Flask application  
│── 📜 fixed_model.h5      # Trained deep learning model  
│── 📜 requirements.txt    # Dependencies  
│── 📜 README.md           # Project documentation  
│── 📂 templates/          # Folder for HTML templates  
│   ├── index.html        # Web interface  



---

📌 Usage
Run the Flask App

python app.py
The app will run on http://127.0.0.1:5000/. Open it in your browser.

Make Predictions via API
Send a POST request with an image:
 http://127.0.0.1:5000/predict

🔗 API Endpoints
Endpoint	Method	Description
/POST	Web interface to upload images and get predictions
/predict	POST	API endpoint to get character prediction from an image

📦 Dependencies
This project uses:

Python 3.8+

Flask

PyTorch

OpenCV

NumPy

torchvision

🙏 Acknowledgement-
under guidance of: Dr Agughasi Victor Ikechukwu (https://github.com/Victor-Ikechukwu)

🤝 Contributing
Feel free to submit issues or pull requests to improve the project.

📜 License
This project is licensed under the MIT License.


---

