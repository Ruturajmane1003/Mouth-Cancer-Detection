import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="Oral Cancer Detection", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS: centered container, max-width, consistent spacing, footer
st.markdown("""
<style>
    .main .block-container {
        max-width: 960px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .stMarkdown h1 {
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .stMarkdown h2 {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .instruction-box {
        padding: 1rem 1.25rem;
        background-color: #f8f9fa;
        border-left: 4px solid #0e1117;
        border-radius: 0 4px 4px 0;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .result-box {
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .disclaimer-box {
        padding: 1rem 1.25rem;
        background-color: #fff3cd;
        border: 1px solid #b8860b;
        border-radius: 4px;
        margin-top: 2rem;
        font-size: 0.9rem;
        color: #1a1a1a;
    }
    .disclaimer-box strong { color: #1a1a1a; }
    /* Center uploaded image block */
    div[data-testid="stImage"] {
        margin-left: auto;
        margin-right: auto;
        display: block;
    }
    .stImage img {
        max-width: 380px !important;
        width: 100% !important;
        height: auto !important;
        object-fit: contain;
        margin-left: auto;
        margin-right: auto;
        display: block;
    }
    .app-footer {
        margin-top: 2.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(128, 128, 128, 0.3);
        text-align: center;
        font-size: 0.85rem;
        color: #9ca3af;
        line-height: 1.8;
    }
    .app-footer a {
        color: #94a3b8;
        text-decoration: none;
    }
    .app-footer a:hover {
        text-decoration: underline;
    }
    .app-footer .footer-heading {
        font-size: 0.95rem;
        font-weight: 600;
        color: #d1d5db;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # oral_cancer_model_clean.h5 is a classification HEAD only (saved with Keras 3).
    # It expects input (batch, 7, 7, 1280). Build same architecture and load weights.
    head = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 1280)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="head")
    head.load_weights("oral_cancer_model_clean.h5", by_name=True)

    # Base: MobileNetV2 output is (7, 7, 1280). Use ImageNet weights for feature extraction.
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base.trainable = False
    # Full pipeline: image (224,224,3) -> base -> (7,7,1280) -> head -> (1,)
    model = tf.keras.Sequential([base, head], name="oral_cancer_model")
    return model


model = load_model()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ---------------------------------------------------------------------------
# 1. PROJECT TITLE & DESCRIPTION
# ---------------------------------------------------------------------------
st.title("Computer-Aided Diagnosis System for Invasive Oral Cancer Detection using Deep Learning Techniques")

st.markdown(
    "This project presents a computer-aided diagnosis system for automated detection of invasive oral cancer from oral cavity images using deep learning. "
    "A transfer learning-based Convolutional Neural Network using MobileNetV2 is employed to classify images into Cancer and Normal categories. "
    "The system is designed for academic and research purposes to assist in early screening and decision support."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# 2. MODEL & METHODOLOGY SECTION
# ---------------------------------------------------------------------------
st.header("Model Architecture and Methodology")
st.markdown("""
- **Base Model:** MobileNetV2 (Transfer Learning)
- **Pre-trained on:** ImageNet
- **Input image size:** 224 x 224 x 3 (RGB)
- **Architecture:** Global Average Pooling followed by Dense layers
- **Task:** Binary classification (Cancer vs Normal)
- **Training:** Optimized using Adam optimizer and Binary Cross-Entropy loss
- **Post-processing:** Threshold tuning applied to reduce false negatives
- **Deployment:** Final model deployed without retraining
""")
st.markdown("---")

# ---------------------------------------------------------------------------
# 3. DATASET & PERFORMANCE METRICS
# ---------------------------------------------------------------------------
st.header("Dataset and Model Performance")

st.markdown("**Dataset Sources:**")
st.markdown("""
- **Oral Cancer Dataset (Kaggle)**  
  https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset
- **Oral Cancer Images for Classification (Kaggle)**  
  https://www.kaggle.com/datasets/muhammadatef/oral-cancer-images-for-classification
""")

st.markdown("**Dataset Summary:**")
st.markdown("""
- Multiple public datasets were merged to improve diversity
- Total images used after merging datasets: **1988**
- Cancer images: **1185**
- Normal images: **803**
- Dataset split into training, validation, and test sets
""")

st.markdown("**Performance Metrics (Test Set):**")
st.markdown("""
- **Accuracy:** 95%
- **Precision (Cancer):** 0.94
- **Recall (Cancer):** 0.96
- **F1-score (Cancer):** 0.95
""")

st.markdown("**Confusion Matrix Summary:**")
st.markdown("""
- True Positives: 171 | False Negatives: 7  
- False Positives: 10 | True Negatives: 141
""")
st.markdown("*Recall was prioritized to reduce false negatives due to the critical medical importance of missed cancer cases.*")
st.markdown("---")

# ---------------------------------------------------------------------------
# 4. IMAGE UPLOAD INSTRUCTIONS
# ---------------------------------------------------------------------------
st.header("Image Upload Guidelines")
with st.container():
    st.markdown("- Upload clear oral cavity images only")
    st.markdown("- Supported formats: JPG, JPEG, PNG")
    st.markdown("- Image must be RGB (3-channel)")
    st.markdown("- Recommended resolution: minimum 224 x 224 pixels")
    st.markdown("- Avoid blurred or low-light images")
    st.markdown("- One image at a time")
st.markdown("")

# ---------------------------------------------------------------------------
# 5 & 6. UPLOAD WIDGET + LAYOUT (IMAGE CENTERED, THEN BUTTON AND RESULTS BELOW)
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload an image above to see the preview and the **Predict** button below.")

prediction_result = None
prob_value = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # 1. Uploaded image section - centered
    st.subheader("Uploaded Image")
    _, col_img, _ = st.columns([1, 2, 1])
    with col_img:
        st.image(image, caption="Uploaded Image", width=360)
    st.markdown("")

    # 2. Predict button and caption - directly under the image
    st.subheader("Prediction")
    st.caption("Click the button below to run the model on the uploaded image.")
    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing image..."):
            processed_image = preprocess_image(image)
            prob_value = float(model.predict(processed_image, verbose=0)[0][0])

        # Model was trained with: 1 = Normal, 0 = Cancer (high output = Normal)
        is_cancer = prob_value < 0.5
        confidence = abs(prob_value - 0.5) * 2 * 100

        prediction_result = {
            "detected_class": "Cancer" if is_cancer else "Normal",
            "confidence": confidence,
            "is_cancer": is_cancer,
        }

    # 3. Prediction result and risk interpretation - under the button
    if prediction_result is not None:
        det = prediction_result["detected_class"]
        conf = prediction_result["confidence"]
        is_cancer = prediction_result["is_cancer"]

        if conf < 40:
            risk_level = "Low Risk"
        elif conf <= 70:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        st.markdown("**Prediction Result**")
        st.markdown(f"- **Detected Class:** {det}")
        st.markdown(f"- **Confidence Score:** {conf:.1f}%")
        st.markdown("**Risk Interpretation**")
        st.markdown("- Confidence < 40%: Low Risk")
        st.markdown("- Confidence 40–70%: Moderate Risk")
        st.markdown("- Confidence > 70%: High Risk")
        if is_cancer:
            st.markdown(f"**Risk Level:** {risk_level} (based on confidence)")

        # ---------------------------------------------------------------------------
        # 8. STAGE ESTIMATION (NON-CLINICAL)
        # ---------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Stage Estimation (Confidence-Based, Non-Clinical)")
        st.markdown(
            "The model does not predict clinical cancer stages. However, based on prediction confidence, "
            "a tentative severity indication is shown for academic interpretation."
        )
        if is_cancer:
            if 40 <= conf < 60:
                stage_label = "Early Suspicion (40–60%)"
            elif 60 <= conf <= 80:
                stage_label = "Moderate Severity (60–80%)"
            elif conf > 80:
                stage_label = "High Severity (>80%)"
            else:
                stage_label = "Below 40% confidence (low confidence indication)"
            st.info(f"Tentative indication: **{stage_label}**")
        else:
            st.info("No stage estimation shown for Normal classification.")
        st.caption("This is not a clinical stage and must not be used for diagnosis.")

        # ---------------------------------------------------------------------------
        # VISUAL ANALYTICS (below prediction result)
        # ---------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Prediction Confidence Distribution")
        prob_cancer_pct = (1 - prob_value) * 100
        prob_normal_pct = prob_value * 100
        classes = ["Cancer", "Normal"]
        values = [round(prob_cancer_pct, 1), round(prob_normal_pct, 1)]
        colors = [
            "rgba(192, 57, 43, 0.9)" if is_cancer else "rgba(127, 140, 141, 0.4)",
            "rgba(46, 204, 113, 0.9)" if not is_cancer else "rgba(127, 140, 141, 0.4)",
        ]
        fig_dist = go.Figure(
            go.Bar(
                x=values,
                y=classes,
                orientation="h",
                text=[f"{v}%" for v in values],
                textposition="outside",
                marker_color=colors,
            )
        )
        fig_dist.update_layout(
            xaxis_title="Percentage (%)",
            yaxis_title="",
            xaxis=dict(range=[0, 105]),
            margin=dict(l=80, r=80, t=30, b=50),
            height=220,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            showlegend=False,
        )
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

        st.subheader("Model Confidence Level")
        st.progress(min(conf / 100.0, 1.0))
        if conf < 40:
            conf_label = "Low Confidence (below 40%)"
        elif conf <= 70:
            conf_label = "Moderate Confidence (40–70%)"
        else:
            conf_label = "High Confidence (above 70%)"
        st.caption(f"Current: **{conf_label}** — {conf:.1f}%")

        st.markdown("---")
        st.subheader("Model Performance Summary (Test Dataset)")
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
        metrics_values = [95, 94, 96, 95]
        fig_perf = go.Figure(
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                text=[f"{v}%" for v in metrics_values],
                textposition="outside",
                marker_color="rgba(52, 152, 219, 0.8)",
            )
        )
        fig_perf.update_layout(
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 105]),
            margin=dict(l=60, r=60, t=40, b=60),
            height=280,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
            showlegend=False,
        )
        st.caption("Test Set Performance")
        st.plotly_chart(fig_perf, use_container_width=True, config={"displayModeBar": False})

        st.subheader("Confusion Matrix (Test Dataset)")
        # sklearn convention: rows = True label, cols = Predicted label; class order ["Cancer", "Normal"]
        # Row 0 = True Cancer (TP=171, FN=7); Row 1 = True Normal (FP=10, TN=141)
        cm = np.array([[171, 7], [10, 141]])
        fig_cm = go.Figure(
            go.Heatmap(
                z=cm,
                x=["Cancer", "Normal"],
                y=["Cancer", "Normal"],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14},
                colorscale=[[0, "rgba(52, 73, 94, 0.3)"], [1, "rgba(52, 152, 219, 0.9)"]],
            )
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            yaxis=dict(autorange="reversed"),  # Cancer (first class) at TOP row
            margin=dict(l=100, r=80, t=30, b=80),
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
        st.caption("Top-left: Cancer predicted as Cancer (TP=171). Top-right: Cancer predicted as Normal (FN=7). Bottom-left: Normal predicted as Cancer (FP=10). Bottom-right: Normal predicted as Normal (TN=141).")

        st.subheader("How to Interpret These Results")
        st.markdown("""
- Higher confidence indicates stronger model certainty in the prediction.
- The graphs above provide transparency into how the model weighs Cancer vs Normal.
- False negatives are minimized in training to prioritize detection of cancer cases.
- Visual outputs improve trust and explainability of the system.
""")

# ---------------------------------------------------------------------------
# 9. DISCLAIMER
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="disclaimer-box">
<strong>Disclaimer:</strong> This system is developed strictly for academic and research purposes. 
The predictions generated by this application do not constitute medical advice and must not be used 
as a substitute for professional clinical diagnosis.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 10. FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="app-footer">
<p class="footer-heading">Project Author</p>
<p><strong>Ruturaj Mane</strong></p>
<p><a href="mailto:Ruturajmane522@gmail.com">Ruturajmane522@gmail.com</a></p>
<p><a href="https://www.linkedin.com/in/ruturaj-mane-13a8a3264/" target="_blank" rel="noopener noreferrer">LinkedIn</a> &middot; <a href="https://github.com/Ruturajmane1003" target="_blank" rel="noopener noreferrer">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
