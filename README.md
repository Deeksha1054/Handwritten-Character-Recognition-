# 📝 Handwritten Character Recognition

Handwritten Character Recognition with Deep Learning and Flask A web-based application that uses a deep learning model to recognize handwritten English characters (0-9, A-Z, a-z) from images. Built with TensorFlow, Flask, and OpenCV, it provides both a web interface and an API endpoint for easy integration.
---

## 🚀 Features
- Upload an image containing a handwritten character.
- Model processes the image and predicts the corresponding character.
- Supports digits (0-9), uppercase (A-Z), and lowercase (a-z) letters.
- Provides an API endpoint for easy integration.
- Built with **Flask**, **TensorFlow**, and **OpenCV**.

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
2️⃣ Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
3️⃣ Install Dependencies

pip install -r requirements.txt
📌 Usage
Run the Flask App

python app.py
The app will run on http://127.0.0.1:5000/. Open it in your browser.

Make Predictions via API
Send a POST request with an image:


curl -X POST -F "file=@test.png" http://127.0.0.1:5000/predict
🔗 API Endpoints
Endpoint	Method	Description
/	GET/POST	Web interface to upload images and get predictions
/predict	POST	API endpoint to get character prediction from an image
📦 Dependencies
This project uses:

Python 3.10
Flask 2.2.5
TensorFlow 2.13.0
NumPy 1.23.5
OpenCV 4.7.0.72
Matplotlib 3.6.2
Pillow 9.3.0
Scikit-learn 1.2.0
Gunicorn 20.1.0
Werkzeug 2.2.2
Flask-CORS 3.0.10
Install them with:

🙏 Acknowledgement
under guidance of: Dr Agughasi Victor Ikechukwu (https://github.com/Victor-Ikechukwu)

🤝 Contributing
Feel free to submit issues or pull requests to improve the project.

📜 License
This project is licensed under the MIT License.


---

