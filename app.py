import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="🧠 Klasifikasi Citra Intel",
    page_icon="🧠",
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
    """Load the trained TensorFlow model using TFSMLayer for Keras 3 compatibility"""
    try:
        model_path = "saved_model"
        if os.path.exists(model_path):
            # For Keras 3, use TFSMLayer to load SavedModel
            tfsm_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
            
            # Create a simple wrapper model
            inputs = tf.keras.Input(shape=(224, 224, 3))
            outputs = tfsm_layer(inputs)
            
            # Handle dictionary output from TFSMLayer
            if isinstance(outputs, dict):
                # Get the actual prediction output (usually the first/main output)
                output_key = list(outputs.keys())[0]
                outputs = outputs[output_key]
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model
        else:
            st.error("Model tidak ditemukan! Pastikan folder 'saved_model' ada.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Fallback: try loading with TensorFlow's SavedModel loader
        try:
            model = tf.saved_model.load(model_path)
            return model
        except Exception as e2:
            st.error(f"Fallback loading juga gagal: {str(e2)}")
            return None

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
    """Make prediction on the preprocessed image"""
    try:
        # Check if model is a Keras model or TF SavedModel
        if hasattr(model, 'predict'):
            # Keras model
            predictions = model.predict(image_array)
        else:
            # TF SavedModel - use inference function
            infer = model.signatures['serving_default']
            input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
            
            # Get input name dynamically
            input_keys = list(infer.structured_input_signature[1].keys())
            input_name = input_keys[0] if input_keys else 'input_1'
            
            predictions = infer(**{input_name: input_tensor})
            
            # Extract predictions from output dict
            output_keys = list(predictions.keys())
            predictions = predictions[output_keys[0]].numpy()
        
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
    # Title and description
    st.title("🧠 Klasifikasi Citra Intel dengan CNN")
    st.markdown("""
    **Aplikasi web untuk mengklasifikasikan gambar menggunakan Transfer Learning (MobileNetV2)**
    
    Upload gambar dan model akan mengklasifikasikannya ke dalam salah satu dari 6 kategori:
    🏢 Buildings • 🌲 Forest • 🏔️ Glacier • ⛰️ Mountain • 🌊 Sea • 🛣️ Street
    """)
    
    # Load model and labels
    model = load_model()
    labels = load_labels()
    
    if model is None:
        st.stop()
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ Informasi Model")
        st.write("**Arsitektur:** MobileNetV2 + Custom Layers")
        st.write("**Akurasi:** >91%")
        st.write("**Dataset:** Intel Image Classification")
        st.write("**Classes:** 6 kategori")
        
        st.header("📊 Class Labels")
        for i, label in enumerate(labels, 1):
            emoji_map = {
                "buildings": "🏢",
                "forest": "🌲", 
                "glacier": "🏔️",
                "mountain": "⛰️",
                "sea": "🌊",
                "street": "🛣️"
            }
            emoji = emoji_map.get(label.lower(), "📸")
            st.write(f"{i}. {emoji} {label.title()}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Gambar")
        uploaded_file = st.file_uploader(
            "Pilih gambar untuk diklasifikasi",
            type=['png', 'jpg', 'jpeg'],
            help="Upload gambar dalam format PNG, JPG, atau JPEG"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diupload", use_container_width=True)
            
            # Preprocess and predict
            with st.spinner("🔄 Memproses gambar..."):
                processed_image = preprocess_image(image)
                predicted_class, confidence, top_3 = predict_image(model, processed_image, labels)
            
            if predicted_class is not None:
                with col2:
                    st.header("🎯 Hasil Prediksi")
                    
                    # Main prediction
                    emoji_map = {
                        "buildings": "🏢",
                        "forest": "🌲", 
                        "glacier": "🏔️",
                        "mountain": "⛰️",
                        "sea": "🌊",
                        "street": "🛣️"
                    }
                    emoji = emoji_map.get(predicted_class.lower(), "📸")
                    
                    st.success(f"**Prediksi:** {emoji} {predicted_class.title()}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Top 3 predictions
                    st.subheader("📊 Top 3 Prediksi")
                    for i, (class_name, conf) in enumerate(top_3, 1):
                        emoji = emoji_map.get(class_name.lower(), "📸")
                        st.write(f"{i}. {emoji} {class_name.title()}: {conf:.2%}")
                        st.progress(conf)
                    
                    # Additional info
                    st.subheader("ℹ️ Detail Teknis")
                    st.write(f"**Resolusi Input:** {image.size}")
                    st.write(f"**Model Input:** 224×224 pixels")
                    st.write(f"**Format:** {image.format}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🚀 Dibuat dengan <strong>Streamlit</strong> dan <strong>TensorFlow</strong></p>
        <p>Model: Transfer Learning dengan MobileNetV2 + Custom CNN Layers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 