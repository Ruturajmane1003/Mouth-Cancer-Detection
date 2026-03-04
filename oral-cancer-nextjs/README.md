# Oral Cancer Detection — Next.js Web App

This is the Next.js version of the Oral Cancer Detection app. Same functionality as the Streamlit app: one scrollable page, image upload, Predict button, and all result sections with improved visuals.

## Run the app

1. **Start the prediction API** (from the project root, not this folder):
   ```bash
   cd ..
   pip install flask flask-cors
   python predict_api.py
   ```
   The API runs at `http://localhost:5001`.

2. **Start the Next.js app** (from this folder):
   ```bash
   npm install
   npm run dev
   ```
   Open [http://localhost:3000](http://localhost:3000).

3. Upload an image, click **Predict**, and view results. The frontend calls the Python API for predictions.

## Optional: API URL

To use a different API URL, create `.env.local` in this folder:
```
NEXT_PUBLIC_PREDICT_API=http://localhost:5001
```
