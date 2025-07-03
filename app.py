import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

# Enhanced TensorFlow/TFLite import with better error handling
TF_AVAILABLE = False
INTERPRETER = None

def initialize_model():
    """Initialize TensorFlow model with comprehensive error handling"""
    global TF_AVAILABLE, INTERPRETER
    
    # First, try tflite-runtime (lighter for deployment)
    try:
        import tflite_runtime.interpreter as tflite
        model_path = "tflite/model.tflite"
        
        if os.path.exists(model_path):
            INTERPRETER = tflite.Interpreter(model_path=model_path)
            INTERPRETER.allocate_tensors()
            TF_AVAILABLE = True
            return "tflite_runtime"
        else:
            st.warning(f"Model file not found at {model_path}")
            return "demo_mode"
            
    except ImportError:
        # Fallback to tensorflow.lite
        try:
            import tensorflow as tf
            model_path = "tflite/model.tflite"
            
            if os.path.exists(model_path):
                INTERPRETER = tf.lite.Interpreter(model_path=model_path)
                INTERPRETER.allocate_tensors()
                TF_AVAILABLE = True
                return "tensorflow"
            else:
                st.warning(f"Model file not found at {model_path}")
                return "demo_mode"
                
        except ImportError:
            return "demo_mode"
            
    except Exception as e:
        st.warning(f"Model loading error: {str(e)}")
        return "demo_mode"

# Initialize model at startup
MODEL_STATUS = initialize_model()

# Set page config with production optimizations
st.set_page_config(
    page_title="CLASSIFIT - AI Image Classification",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@classifit.com',
        'Report a bug': 'mailto:bugs@classifit.com',
        'About': """
        # CLASSIFIT v2.0.0
        
        Advanced AI-powered image classification using deep learning.
        Built with Streamlit, TensorFlow, and MobileNetV2.
        
        **Features:**
        - Real-time image classification
        - 6 category support (Buildings, Forest, Glacier, Mountain, Sea, Street)
        - Progressive web app capabilities
        - Dark/Light mode support
        - Mobile responsive design
        """
    }
)

# Load class labels with enhanced caching
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def load_labels():
    """Load class labels from the label file with fallback"""
    try:
        labels_path = "tflite/label.txt"
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            if labels:  # Only return if we have valid labels
                return labels
    except Exception as e:
        # Log error in production but don't show to user unless in debug mode
        if os.getenv('STREAMLIT_ENV') == 'development':
            st.warning(f"Could not load labels: {str(e)}")
    
    # Fallback labels (Intel Image Classification dataset)
    return ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def preprocess_image(image):
    """Preprocess the uploaded image for prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224 for MobileNetV2)
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        st.error(f"Image preprocessing error: {str(e)}")
        return None

def predict_image_demo(labels):
    """Generate demo predictions when TensorFlow is not available"""
    import random
    
    # Set seed for consistent demo results
    random.seed(hash("demo_prediction") % 2**32)
    
    # Generate realistic-looking fake predictions
    fake_probs = []
    for i, label in enumerate(labels):
        if i == 0:  # Make first class more likely
            prob = random.uniform(0.6, 0.9)
        else:
            prob = random.uniform(0.01, 0.3)
        fake_probs.append(prob)
    
    # Normalize probabilities
    total = sum(fake_probs)
    fake_probs = [p/total for p in fake_probs]
    
    # Get predictions
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

def predict_image_real(image_array, labels):
    """Make real prediction using TensorFlow Lite interpreter"""
    global INTERPRETER
    
    try:
        if INTERPRETER is None:
            return None, None, None
            
        # Get input and output details
        input_details = INTERPRETER.get_input_details()
        output_details = INTERPRETER.get_output_details()
        
        # Set input tensor
        INTERPRETER.set_tensor(input_details[0]['index'], image_array)
        
        # Run inference
        INTERPRETER.invoke()
        
        # Get prediction results
        predictions = INTERPRETER.get_tensor(output_details[0]['index'])
        
        # Handle predictions (remove batch dimension if present)
        if len(predictions.shape) > 1:
            predictions = predictions[0]
            
        # Get results
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
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def predict_image(image_array, labels):
    """Main prediction function with fallback to demo mode"""
    if TF_AVAILABLE and INTERPRETER is not None:
        result = predict_image_real(image_array, labels)
        if result[0] is not None:
            return result
    
    # Fallback to demo mode
    return predict_image_demo(labels)

def get_confidence_color(confidence):
    """Get color class based on confidence level"""
    if confidence > 0.8:
        return "confidence-high"
    elif confidence > 0.5:
        return "confidence-medium"
    else:
        return "confidence-low"

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def get_app_info():
    """Get application information for production monitoring"""
    return {
        "version": "2.0.0",
        "ml_backend": MODEL_STATUS,
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "streamlit_version": st.__version__
    }

def add_footer_analytics():
    """Add analytics and footer information"""
    app_info = get_app_info()
    
    # Add some basic analytics (non-intrusive)
    st.markdown("""
    <script>
    // Basic page view tracking (non-personal)
    if (typeof gtag !== 'undefined') {
        gtag('event', 'page_view', {
            'page_title': 'CLASSIFIT - AI Image Classification',
            'page_location': window.location.href
        });
    }
    </script>
    """, unsafe_allow_html=True)

def main():
    """Main application function with production optimizations"""
    
    # Enhanced CSS with perfect dark/light mode text support
    st.markdown("""
    <style>
    /* CSS Variables for comprehensive theme compatibility */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --danger-color: #dc3545;
        
        /* Light theme colors */
        --light-bg: #ffffff;
        --light-secondary-bg: rgba(248, 249, 250, 0.98);
        --light-text: #1a1a1a;
        --light-text-secondary: #495057;
        --light-border: #d1d5db;
        --light-shadow: rgba(0, 0, 0, 0.15);
        
        /* Dark theme colors */
        --dark-bg: #0e1117;
        --dark-secondary-bg: rgba(38, 39, 48, 0.95);
        --dark-text: #fafafa;
        --dark-text-secondary: #b3b3b3;
        --dark-border: #404040;
        --dark-shadow: rgba(255, 255, 255, 0.1);
    }
    
    /* Default light theme */
    :root {
        --bg-color: var(--light-bg);
        --secondary-bg: var(--light-secondary-bg);
        --text-color: var(--light-text);
        --text-secondary: var(--light-text-secondary);
        --border-color: var(--light-border);
        --shadow-color: var(--light-shadow);
    }
    
    /* Auto dark mode detection */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-color: var(--dark-bg);
            --secondary-bg: var(--dark-secondary-bg);
            --text-color: var(--dark-text);
            --text-secondary: var(--dark-text-secondary);
            --border-color: var(--dark-border);
            --shadow-color: var(--dark-shadow);
        }
    }
    
    /* Streamlit theme detection (most reliable) */
    .stApp[data-theme="dark"] {
        --bg-color: var(--dark-bg);
        --secondary-bg: var(--dark-secondary-bg);
        --text-color: var(--dark-text);
        --text-secondary: var(--dark-text-secondary);
        --border-color: var(--dark-border);
        --shadow-color: var(--dark-shadow);
    }
    
    .stApp[data-theme="light"] {
        --bg-color: var(--light-bg);
        --secondary-bg: var(--light-secondary-bg);
        --text-color: var(--light-text);
        --text-secondary: var(--light-text-secondary);
        --border-color: var(--light-border);
        --shadow-color: var(--light-shadow);
    }
    
    /* Force text color inheritance for all custom elements */
    .custom-text, .custom-text * {
        color: var(--text-color) !important;
    }
    
    .custom-text-secondary, .custom-text-secondary * {
        color: var(--text-secondary) !important;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 2px;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-weight: 300;
    }
    
    /* Status indicators */
    .status-demo {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(255,107,107,0.3);
    }
    
    .status-active {
        background: linear-gradient(135deg, #51cf66, #69db7c);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(81,207,102,0.3);
    }
    
    /* Category tags */
    .category-tag {
        display: inline-block;
        background: var(--secondary-bg);
        color: var(--text-color);
        padding: 0.4rem 1rem;
        margin: 0.25rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
        border: 2px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .category-tag:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Info boxes with enhanced text styling */
    .info-box {
        background: var(--secondary-bg);
        color: var(--text-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px var(--shadow-color);
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        box-shadow: 0 4px 16px var(--shadow-color);
        transform: translateY(-2px);
    }
    
    .info-box strong {
        color: var(--text-color) !important;
        font-weight: 700 !important;
    }
    
    .info-box br + strong {
        margin-top: 0.5rem;
        display: inline-block;
    }
    
    /* Better text styling in info boxes */
    .info-box, .info-box * {
        color: var(--text-color) !important;
        line-height: 1.6 !important;
    }
    
    /* Result section with improved text visibility */
    .result-section {
        background: var(--secondary-bg);
        color: var(--text-color);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 1rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px var(--shadow-color);
        transition: all 0.3s ease;
    }
    
    .result-section h2 {
        color: var(--text-color) !important;
    }
    
    /* Confidence colors */
    .confidence-high { 
        color: var(--success-color); 
        font-weight: bold;
    }
    .confidence-medium { 
        color: var(--warning-color); 
        font-weight: bold;
    }
    .confidence-low { 
        color: var(--danger-color); 
        font-weight: bold;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        border: 3px dashed var(--border-color);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: var(--bg-color);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--primary-color);
        background: var(--secondary-bg);
    }
    
    /* Footer with enhanced text styling */
    .footer-section {
        text-align: center;
        padding: 2rem;
        background: var(--secondary-bg);
        color: var(--text-color);
        margin-top: 3rem;
        border-radius: 15px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px var(--shadow-color);
        transition: all 0.3s ease;
    }
    
    .footer-section h4 {
        color: var(--text-color) !important;
        margin: 0;
    }
    
    .footer-section p {
        color: var(--text-color) !important;
    }
    
    .footer-section .footer-secondary {
        color: var(--text-secondary) !important;
        opacity: 0.8;
    }
    
    /* Progress bar enhancements */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
    
    /* Hide empty file uploader elements */
    .stFileUploader > div > div > div:last-child {
        display: none;
    }
    
    /* Enhanced text readability */
    .stMarkdown, .stMarkdown * {
        color: inherit;
    }
    
    /* Ensure all headings use proper colors */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
    }
    
    /* Improve text contrast in all sections */
    .stSidebar .info-box {
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 8px var(--shadow-color);
    }
    
    /* Enhanced sidebar styling for better readability */
    .stSidebar {
        background-color: var(--bg-color);
    }
    
    .stSidebar .stMarkdown {
        color: var(--text-color) !important;
    }
    
    .stSidebar h3 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Status indicators with better text contrast */
    .status-demo, .status-active {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Category tags with proper text colors */
    .category-tag {
        color: var(--text-color) !important;
        background: var(--secondary-bg) !important;
        border: 2px solid var(--border-color) !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px var(--shadow-color);
    }
    
    .category-tag:hover {
        background: var(--text-color) !important;
        color: var(--secondary-bg) !important;
        border-color: var(--text-color) !important;
    }
    
    /* Technical details section */
    .tech-details {
        background: var(--secondary-bg);
        color: var(--text-color);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
    }
    
    /* Ensure proper text visibility in all custom components */
    .custom-component {
        color: var(--text-color) !important;
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
    }
    
    /* Progress text enhancement */
    .stProgress .stProgress-text {
        color: var(--text-color) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">CLASSIFIT</h1>
        <p class="subtitle">🤖 AI-Powered Image Classification Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show model status
    if MODEL_STATUS == "demo_mode":
        st.markdown("""
        <div class="status-demo">
            🎭 <strong>Demo Mode Active</strong> - TensorFlow not available, showing simulated results
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-active">
            🚀 <strong>AI Model Active</strong> - Real-time inference ready
        </div>
        """, unsafe_allow_html=True)
    
    # Description and categories
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0; padding: 1.5rem; background: var(--secondary-bg); border-radius: 12px; border: 1px solid var(--border-color); box-shadow: 0 2px 8px var(--shadow-color);" class="custom-text">
        <h3 style="color: var(--text-color) !important; font-weight: 700; margin-bottom: 1rem; font-size: 1.8rem;">
            🧠 Advanced Image Classification with Deep Learning
        </h3>
        <p style="font-size: 1.1rem; color: var(--text-secondary) !important; line-height: 1.6; margin: 0;">
            Upload any image and our AI will classify it using state-of-the-art MobileNetV2 architecture
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load labels
    labels = load_labels()
    
    # Display supported categories
    categories_html = '<div style="text-align: center; margin: 1.5rem 0;">'
    for label in labels:
        categories_html += f'<span class="category-tag">{label.title()}</span>'
    categories_html += '</div>'
    st.markdown(categories_html, unsafe_allow_html=True)
    
    # Sidebar information
    with st.sidebar:
        st.markdown("""
        <h3 style="color: var(--text-color) !important; font-weight: 700 !important; margin-bottom: 1rem !important; font-size: 1.2rem;">
            🔧 Model Information
        </h3>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box" style="background: var(--secondary-bg) !important; color: var(--text-color) !important;">
            <div style="color: var(--text-color) !important; font-size: 0.95rem; line-height: 1.8;">
                <strong style="color: var(--text-color) !important; font-weight: 700;">🏗️ Architecture:</strong> 
                <span style="color: var(--text-secondary) !important;">MobileNetV2 + Custom Layers</span><br>
                <strong style="color: var(--text-color) !important; font-weight: 700;">📊 Accuracy:</strong> 
                <span style="color: var(--text-secondary) !important;">>91%</span><br>
                <strong style="color: var(--text-color) !important; font-weight: 700;">📁 Dataset:</strong> 
                <span style="color: var(--text-secondary) !important;">Intel Image Classification</span><br>
                <strong style="color: var(--text-color) !important; font-weight: 700;">🎯 Classes:</strong> 
                <span style="color: var(--text-secondary) !important;">{len(labels)} categories</span><br>
                <strong style="color: var(--text-color) !important; font-weight: 700;">⚡ Status:</strong> 
                <span style="color: var(--text-secondary) !important;">{MODEL_STATUS.replace('_', ' ').title()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="color: var(--text-color) !important; font-weight: 700 !important; margin: 1.5rem 0 1rem 0 !important; font-size: 1.2rem;">
            📂 Classification Categories
        </h3>
        """, unsafe_allow_html=True)
        
        # Build categories list as a single clean HTML string
        categories_list = ""
        for i, label in enumerate(labels, 1):
            categories_list += f"<strong style='color: var(--text-color) !important; font-weight: 700;'>{i}.</strong> <span style='color: var(--text-secondary) !important; font-weight: 500;'>{label.title()}</span><br>"
        
        st.markdown(f"""
        <div class="info-box" style="background: var(--secondary-bg) !important; color: var(--text-color) !important;">
            <div style="color: var(--text-color) !important; font-size: 0.95rem; line-height: 1.8;">
                {categories_list}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if MODEL_STATUS == "demo_mode":
            st.markdown("### ℹ️ Demo Mode Info")
            st.markdown("""
            <div class="info-box">
                <div style="color: var(--text-color) !important; font-size: 0.9rem; line-height: 1.6;">
                    <strong style="color: var(--text-color) !important;">🎭 Demo Mode Active</strong><br>
                    <span style="color: var(--text-secondary) !important;">
                    TensorFlow is not available in this deployment environment. 
                    The app shows realistic simulated predictions to demonstrate the UI and functionality.
                    </span><br><br>
                    <strong style="color: var(--text-color) !important;">📱 Full Features Available:</strong><br>
                    <span style="color: var(--text-secondary) !important;">
                    • Complete user interface<br>
                    • Image upload and processing<br>
                    • Realistic prediction simulation<br>
                    • All visual components working
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF (Max size: 200MB)"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(
                    image, 
                    caption=f"📸 Uploaded: {uploaded_file.name}", 
                    use_container_width=True
                )
                
                # Show image details
                st.markdown(f"""
                <div class="custom-text-secondary" style="text-align: center; margin: 1rem 0; font-size: 0.9rem;">
                    📏 Dimensions: {image.size[0]} × {image.size[1]} | 
                    🎨 Format: {image.format} | 
                    📁 Size: {uploaded_file.size / 1024:.1f} KB
                </div>
                """, unsafe_allow_html=True)
                
                # Process and predict
                with st.spinner("🔄 Processing image..."):
                    processed_image = preprocess_image(image)
                    if processed_image is not None:
                        predicted_class, confidence, top_3 = predict_image(processed_image, labels)
                    else:
                        predicted_class, confidence, top_3 = None, None, None
                
                if predicted_class is not None:
                    with col2:
                        st.markdown("### 🎯 Prediction Results")
                        
                        # Main prediction result
                        confidence_class = get_confidence_color(confidence)
                        
                        st.markdown(f"""
                        <div class="result-section custom-text">
                            <div style="text-align: center; margin-bottom: 2rem;">
                                <h2 style="margin: 0; color: var(--text-color) !important; font-size: 2.5rem; font-weight: 700;">
                                    🏷️ {predicted_class.title()}
                                </h2>
                                <p class="{confidence_class}" style="font-size: 2rem; margin: 1rem 0; font-weight: 600;">
                                    {confidence:.1%} Confidence
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence progress bar
                        st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                        
                        # Top 3 predictions
                        st.markdown("### 📊 Top 3 Predictions")
                        
                        for i, (class_name, conf) in enumerate(top_3, 1):
                            conf_class = get_confidence_color(conf)
                            
                            # Create columns for ranking
                            rank_col, name_col, conf_col = st.columns([0.5, 2, 1])
                            
                            with rank_col:
                                st.markdown(f"**{i}.**")
                            
                            with name_col:
                                st.markdown(f"**{class_name.title()}**")
                            
                            with conf_col:
                                st.markdown(f'<span class="{conf_class}">{conf:.1%}</span>', unsafe_allow_html=True)
                            
                            # Progress bar for each prediction
                            st.progress(conf)
                        
                        # Technical details
                        st.markdown("### 🔧 Technical Details")
                        tech_details = f"""
                        <div class="tech-details custom-text">
                            <strong style="color: var(--text-color) !important;">📥 Input Processing:</strong><br>
                            <span style="color: var(--text-secondary) !important;">
                            • Original: {image.size[0]} × {image.size[1]} pixels<br>
                            • Resized: 224 × 224 pixels<br>
                            • Normalized: [0, 1] range
                            </span><br><br>
                            <strong style="color: var(--text-color) !important;">🧠 Model Details:</strong><br>
                            <span style="color: var(--text-secondary) !important;">
                            • Architecture: MobileNetV2<br>
                            • Classes: {len(labels)}<br>
                            • Backend: {MODEL_STATUS.replace('_', ' ').title()}
                            </span>
                        </div>
                        """
                        st.markdown(tech_details, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.info("💡 Please try uploading a different image or check the file format.")
    
    # Footer
    st.markdown("---")
    backend_info = MODEL_STATUS.replace('_', ' ').title()
    if MODEL_STATUS == "demo_mode":
        backend_info += " (Simulated)"
    
    # Get app info for footer
    app_info = get_app_info()
    
    st.markdown(f"""
    <div class="footer-section custom-text">
        <h4 style="margin: 0; color: var(--text-color) !important; font-weight: 600;">
            ⚡ Powered by Streamlit & {backend_info}
        </h4>
        <p style="margin: 0.5rem 0; color: var(--text-color) !important;">
            🧠 Transfer Learning with MobileNetV2 + Custom CNN Layers
        </p>
        <p class="footer-secondary" style="margin: 0; font-size: 0.9rem;">
            🎯 Built for intelligent image classification and computer vision tasks
        </p>
        <p class="footer-secondary" style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.6;">
            v{app_info['version']} | Python {app_info['python_version']} | Streamlit {app_info['streamlit_version']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add analytics (non-intrusive)
    add_footer_analytics()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Production error handling
        st.error("🚨 An unexpected error occurred. Please refresh the page or try again later.")
        
        # Log error details for debugging (only in development)
        if os.getenv('STREAMLIT_ENV') == 'development':
            st.exception(e)
        else:
            # In production, log to system but don't show user sensitive info
            import logging
            logging.error(f"CLASSIFIT App Error: {str(e)}", exc_info=True) 