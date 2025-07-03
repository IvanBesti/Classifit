import streamlit as st
import numpy as np
from PIL import Image
import os

# Try to import TensorFlow/TFLite, but make it optional
TF_AVAILABLE = False
try:
    import tflite_runtime.interpreter as tflite
    TF_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        TF_AVAILABLE = True
    except ImportError:
        st.warning("⚠️ TensorFlow not available. Running in demo mode.")
        TF_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="CLASSIFIT - Klasifikasi Citra Intel",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load class labels
@st.cache_data
def load_labels():
    """Load class labels from the label file"""
    labels_path = "tflite/label.txt"
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        return labels
    else:
        return ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Load the trained model
@st.cache_resource
def load_model():
    """Load the TensorFlow Lite model or return demo mode"""
    if not TF_AVAILABLE:
        return "demo_mode"
    
    try:
        model_path = "tflite/model.tflite"
        if os.path.exists(model_path):
            # Load TFLite model and allocate tensors
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter
        else:
            st.error("Model TFLite tidak ditemukan! Pastikan file 'tflite/model.tflite' ada.")
            return "demo_mode"
    except Exception as e:
        st.error(f"Error loading TFLite model: {str(e)}")
        return "demo_mode"

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size (MobileNetV2 typically uses 224x224)
    image = image.resize((224, 224))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize pixel values to [0, 1]
    image_array = image_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(model, image_array, labels):
    """Make prediction on the preprocessed image using TFLite interpreter or demo mode"""
    if model == "demo_mode":
        # Demo mode - return fake predictions
        import random
        random.seed(42)  # Consistent demo results
        
        # Generate fake but realistic-looking predictions
        fake_probs = [random.uniform(0.1, 0.9) for _ in labels]
        total = sum(fake_probs)
        fake_probs = [p/total for p in fake_probs]  # Normalize to sum to 1
        
        predicted_class_idx = np.argmax(fake_probs)
        confidence = fake_probs[predicted_class_idx]
        predicted_class = labels[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(fake_probs)[-3:][::-1]
        top_3_predictions = [
            (labels[idx], fake_probs[idx]) 
            for idx in top_3_idx
        ]
        
        return predicted_class, confidence, top_3_predictions
    
    try:
        # Get input and output details
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Set the tensor to point to the input data to be inferred
        model.set_tensor(input_details[0]['index'], image_array)
        
        # Run inference
        model.invoke()
        
        # Get the result
        predictions = model.get_tensor(output_details[0]['index'])
        
        # Handle predictions
        if len(predictions.shape) > 1:
            predictions = predictions[0]  # Get first batch
            
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx])
        predicted_class = labels[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = [
            (labels[idx], float(predictions[idx])) 
            for idx in top_3_idx
        ]
        
        return predicted_class, confidence, top_3_predictions
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def main():
    # Custom CSS for better styling with dark/light mode compatibility
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #f0f0f0;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    .category-tag {
        display: inline-block;
        background: var(--background-color);
        color: var(--text-color);
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.9rem;
        border: 1px solid var(--border-color);
    }
    .upload-section {
        background: transparent;
        margin-bottom: 0;
        padding: 0;
    }
    .result-section {
        background: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1px solid var(--border-color);
    }
    .info-box {
        background: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }
    .footer-section {
        text-align: center;
        padding: 2rem 0;
        background: var(--secondary-background-color);
        margin-top: 3rem;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        color: var(--text-color);
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    
    /* Streamlit auto dark mode detection */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #262730;
            --secondary-background-color: rgba(38, 39, 48, 0.8);
            --text-color: #fafafa;
            --border-color: #404040;
        }
    }
    
    @media (prefers-color-scheme: light) {
        :root {
            --background-color: #f8f9fa;
            --secondary-background-color: rgba(255, 255, 255, 0.8);
            --text-color: #495057;
            --border-color: #dee2e6;
        }
    }
    
    /* Streamlit specific dark mode class detection */
    .stApp[data-theme="dark"] {
        --background-color: #262730;
        --secondary-background-color: rgba(38, 39, 48, 0.8);
        --text-color: #fafafa;
        --border-color: #404040;
    }
    
    /* Force text color inheritance for dark mode compatibility */
    .info-box *, .footer-section * {
        color: inherit !important;
    }
    
    /* Default variables for compatibility */
    :root {
        --background-color: #f8f9fa;
        --secondary-background-color: rgba(255, 255, 255, 0.8);
        --text-color: #495057;
        --border-color: #dee2e6;
    }
    
    /* Hide empty space below file uploader */
    .stFileUploader > div > div > div:last-child {
        display: none;
    }
    
    /* Adjust file uploader styling */
    .stFileUploader > div > div {
        border: 2px dashed var(--border-color);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background-color: var(--background-color);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">CLASSIFIT</h1>
        <p class="subtitle">Intelligent Image Classification with Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Description section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h3>Klasifikasi Citra Intel dengan Transfer Learning</h3>
        <p style="font-size: 1.1rem; color: #666;">
            Upload gambar dan AI akan mengklasifikasikannya secara real-time menggunakan MobileNetV2
        </p>
        <div style="margin-top: 1rem;">
            <span class="category-tag">Buildings</span>
            <span class="category-tag">Forest</span>
            <span class="category-tag">Glacier</span>
            <span class="category-tag">Mountain</span>
            <span class="category-tag">Sea</span>
            <span class="category-tag">Street</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and labels
    model = load_model()
    labels = load_labels()
    
    # Show demo mode message if TensorFlow not available
    if model == "demo_mode":
        st.info("🚀 **Demo Mode**: TensorFlow tidak tersedia, tapi kamu masih bisa lihat UI dan test dengan prediksi demo!")
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("### Model Information")
        st.markdown("""
        <div class="info-box">
            <div style="color: var(--text-color);">
                <strong>Architecture:</strong> MobileNetV2 + Custom Layers<br>
                <strong>Accuracy:</strong> >91%<br>
                <strong>Dataset:</strong> Intel Image Classification<br>
                <strong>Classes:</strong> 6 categories
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Classification Categories")
        
        # Build categories list as single HTML block
        categories_html = '<div class="info-box"><div style="color: var(--text-color);">'
        for i, label in enumerate(labels, 1):
            categories_html += f"<strong>{i}.</strong> {label.title()}<br>"
        categories_html += "</div></div>"
        
        st.markdown(categories_html, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Drag and drop atau klik untuk memilih gambar",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Preprocess and predict
            with st.spinner("Processing image..."):
                processed_image = preprocess_image(image)
                predicted_class, confidence, top_3 = predict_image(model, processed_image, labels)
            
            if predicted_class is not None:
                with col2:
                    st.markdown("### Prediction Results")
                    
                    # Build entire result section as single HTML block
                    confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.5 else "confidence-low"
                    
                    # Start result section
                    result_html = f"""
                    <div class="result-section">
                        <div style="text-align: center; margin-bottom: 1.5rem;">
                            <h2 style="margin: 0; color: var(--text-color);">{predicted_class.title()}</h2>
                            <p class="{confidence_class}" style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                                {confidence:.1%} Confidence
                            </p>
                        </div>
                    """
                    
                    st.markdown(result_html, unsafe_allow_html=True)
                    
                    # Confidence progress bar
                    st.progress(confidence)
                    
                    # Top 3 predictions - separate section
                    st.markdown('<div style="color: var(--text-color); margin: 1rem 0;"><strong>Top 3 Predictions:</strong></div>', unsafe_allow_html=True)
                    
                    # Progress bars and labels for top 3
                    for i, (class_name, conf) in enumerate(top_3, 1):
                        conf_class = "confidence-high" if conf > 0.8 else "confidence-medium" if conf > 0.5 else "confidence-low"
                        
                        # Display prediction label and confidence
                        st.markdown(f'''
                        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; color: var(--text-color);">
                            <span>{i}. {class_name.title()}</span>
                            <span class="{conf_class}" style="font-weight: bold;">{conf:.1%}</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(conf)
                    
                    # Technical details
                    st.markdown("""
                    <div style="color: var(--text-color); margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-color);">
                        <strong>Technical Details:</strong><br>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                            <div>
                                Input: {0}×{1}<br>
                                Format: {2}
                            </div>
                            <div>
                                Model: 224×224<br>
                                Classes: 6
                            </div>
                        </div>
                    </div>
                    </div>
                    """.format(image.size[0], image.size[1], image.format), unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    tf_status = "TensorFlow Lite" if model != "demo_mode" else "Demo Mode"
    st.markdown(f"""
    <div class='footer-section'>
        <p style='margin: 0; color: var(--text-color);'>Powered by <strong>Streamlit</strong> and <strong>{tf_status}</strong></p>
        <p style='margin: 0; font-size: 0.9rem; opacity: 0.8; color: var(--text-color);'>Transfer Learning with MobileNetV2 + Custom CNN Layers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 