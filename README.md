# 🦷 Mouth Cancer Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=flat&logo=tensorflow)
![Next.js](https://img.shields.io/badge/Frontend-Next.js-black?style=flat&logo=next.js)
![Vercel](https://img.shields.io/badge/Deployed-Vercel-black?style=flat&logo=vercel)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An end-to-end deep learning system for automated detection of oral cancer from clinical images. Built with a CNN-based Keras model, a Python prediction API, and a Next.js frontend — deployed live on Vercel.

🔗 **Live Demo:** [mouthcancerdetectionusingdeeplearni.vercel.app](https://mouthcancerdetectionusingdeeplearni.vercel.app/)

---

## 📌 Problem Statement

Oral cancer is one of the most common cancers in India, yet early detection rates remain critically low due to limited access to specialists. This project explores how deep learning can assist in early-stage detection by classifying oral cavity images as **cancerous** or **non-cancerous**, enabling faster screening and timely intervention.

---

## 🏗️ Project Structure
```
Mouth-Cancer-Detection/
│
├── oral-cancer-nextjs/          # Next.js frontend (deployed on Vercel)
├── mouth-cancer-detection.ipynb # Model training notebook
├── app.py                       # Flask/FastAPI web server
├── predict_api.py               # Prediction logic & model inference
├── oral_cancer_base_model_final.keras  # Trained Keras model (base)
├── oral_cancer_model_clean.h5   # Cleaned/optimized model weights
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container setup
└── README.md
```

---

## 🧠 Model Architecture

- **Base Architecture:** Convolutional Neural Network (CNN) built with TensorFlow/Keras
- **Training Data:** Oral cancer image dataset (cancerous vs. normal tissue samples)
- **Preprocessing:** Image resizing, normalization, data augmentation (flip, zoom, rotation)
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Output:** Binary classification — Cancerous / Non-Cancerous with confidence score

### Training Pipeline
```
Raw Images
    ↓
Preprocessing (Resize → Normalize → Augment)
    ↓
CNN Model (Conv2D → MaxPooling → Dropout → Dense)
    ↓
Binary Classification Output
    ↓
Deployed via predict_api.py
```

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| Model Training | Python, TensorFlow, Keras |
| Data Processing | NumPy, Pandas, OpenCV, Matplotlib |
| API Backend | Flask / FastAPI (`predict_api.py`) |
| Frontend | Next.js (React) |
| Containerization | Docker |
| Deployment | Vercel (Frontend) |

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/Ruturajmane1003/Mouth-Cancer-Detection.git
cd Mouth-Cancer-Detection
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the prediction API
```bash
python app.py
```
API will start at `http://localhost:5000`

### 4. Run the frontend
```bash
cd oral-cancer-nextjs
npm install
npm run dev
```
Frontend will start at `http://localhost:3000`

---

## 🐳 Docker Setup
```bash
docker build -t mouth-cancer-detection .
docker run -p 5000:5000 mouth-cancer-detection
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Training Accuracy | ~92% |
| Validation Accuracy | ~88% |
| Loss | Binary Cross-Entropy |
| Inference Time | < 500ms per image |

> Note: Model performance may vary based on image quality and lighting conditions.

---

## 🔍 How It Works

1. User uploads an oral cavity image through the web interface
2. The Next.js frontend sends the image to the Python prediction API
3. `predict_api.py` loads the trained Keras model and runs inference
4. The model returns a **prediction label** (Cancerous / Non-Cancerous) with a **confidence score**
5. Result is displayed on the frontend in real time

---

## 📁 Key Files

| File | Description |
|---|---|
| `mouth-cancer-detection.ipynb` | Complete training notebook — EDA, preprocessing, model training, evaluation |
| `predict_api.py` | Core inference logic — loads model, processes image, returns prediction |
| `app.py` | API server entry point |
| `oral_cancer_model_clean.h5` | Final trained model weights |
| `Dockerfile` | Docker container configuration for deployment |

---

## 🌐 Deployment

- **Frontend** is deployed on **Vercel** — auto-deployed from the `oral-cancer-nextjs` folder
- **Backend API** runs via Docker or directly with Python
- Live URL: [https://mouthcancerdetectionusingdeeplearni.vercel.app/](https://mouthcancerdetectionusingdeeplearni.vercel.app/)

---

## 🔮 Future Improvements

- [ ] Add Grad-CAM visualization to highlight the region the model focused on
- [ ] Expand dataset with more diverse oral cancer image samples
- [ ] Add multi-class classification (stage 1 / stage 2 / normal)
- [ ] Integrate speech/confidence analysis for patient interaction
- [ ] Deploy backend API on Render or Railway for full-stack live demo

---

## 👨‍💻 Author

**Ruturaj Mane**
- 📧 ruturajmane522@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/your-link)
- 🐙 [GitHub](https://github.com/Ruturajmane1003)
- 💼 [Portfolio](your-portfolio-link)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⭐ If this project helped you, please give it a star!
