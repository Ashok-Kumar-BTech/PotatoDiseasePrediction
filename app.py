import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .early-blight {
        background-color: #FFE4B5;
        border-left: 5px solid #FF8C00;
    }
    .late-blight {
        background-color: #F0E68C;
        border-left: 5px solid #BDB76B;
    }
    .healthy {
        background-color: #90EE90;
        border-left: 5px solid #228B22;
    }
    .confidence-meter {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🥔 Potato Disease Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
Upload an image of a potato leaf to detect whether it's healthy or affected by early blight or late blight disease.
Our AI model will analyze the image and provide instant results with confidence scores.
""")

# Load model and class indices
@st.cache_resource
def load_model_and_classes():
    try:
        model = load_model('model/potato_disease_model.h5')
        
        with open('model/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        
        # Create reverse mapping
        index_to_class = {v: k for k, v in class_indices.items()}
        return model, index_to_class
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, index_to_class = load_model_and_classes()

# Disease information
disease_info = {
    'Potato___Early_blight': {
        'name': 'Early Blight',
        'description': 'Early blight is a fungal disease caused by Alternaria solani. It appears as dark brown to black spots with concentric rings on older leaves.',
        'treatment': 'Remove affected leaves, apply fungicides containing chlorothalonil or mancozeb, ensure proper spacing for air circulation.',
        'prevention': 'Crop rotation, resistant varieties, proper irrigation, and regular monitoring.'
    },
    'Potato___Late_blight': {
        'name': 'Late Blight',
        'description': 'Late blight is a serious disease caused by Phytophthora infestans. It causes water-soaked lesions that quickly turn dark brown/black.',
        'treatment': 'Apply fungicides containing metalaxyl or copper compounds, remove infected plants immediately.',
        'prevention': 'Use certified seed tubers, avoid overhead irrigation, ensure good drainage, and monitor weather conditions.'
    },
    'Potato___healthy': {
        'name': 'Healthy',
        'description': 'The potato leaf appears healthy with no visible signs of disease. Keep up the good agricultural practices!',
        'treatment': 'Continue regular monitoring, maintain proper irrigation and fertilization.',
        'prevention': 'Continue preventive measures: crop rotation, proper spacing, and regular field inspections.'
    }
}

# Sidebar with information
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.info(f"""
**Model Architecture:**
- CNN with 4 convolutional layers
- Max pooling after each layer
- Dropout for regularization
- 3 output classes

**Performance:**
- Overall Accuracy: 78.42%
- Best performance on Early Blight detection
- Trained on PlantVillage dataset
""")

st.sidebar.markdown("### 🎯 Quick Tips")
st.sidebar.markdown("""
- Upload clear, well-lit images
- Ensure the leaf fills most of the frame
- Avoid blurry or dark images
- Include both sides of the leaf if possible
""")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a potato leaf image...",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )

with col2:
    st.markdown("### 🖼️ Sample Images")
    st.markdown("""
    <small>
    • Clear leaf images work best<br>
    • Include disease symptoms if present<br>
    • Avoid shadows and reflections
    </small>
    """, unsafe_allow_html=True)

# Image processing and prediction
if uploaded_file is not None:
    # Display uploaded image
    st.markdown("### 📸 Uploaded Image")
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded potato leaf image", use_column_width=True)
    
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    if model is not None:
        with st.spinner('🔍 Analyzing image...'):
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100
            predicted_class = index_to_class[predicted_class_index]
        
        # Display results
        st.markdown("### 🎯 Prediction Results")
        
        # Confidence meter
        st.markdown(f'<div class="confidence-meter">Confidence: {confidence:.1f}%</div>', 
                   unsafe_allow_html=True)
        
        # Progress bar for confidence
        st.progress(float(confidence / 100))
        
        # Disease information
        disease = disease_info[predicted_class]
        
        # Color-coded alert
        alert_class = {
            'Potato___Early_blight': 'early-blight',
            'Potato___Late_blight': 'late-blight',
            'Potato___healthy': 'healthy'
        }[predicted_class]
        
        st.markdown(f'''
        <div class="disease-alert {alert_class}">
            <h3>🏷️ {disease['name']}</h3>
            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Detailed information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Description")
            st.write(disease['description'])
            
            st.markdown("#### 💊 Treatment")
            st.write(disease['treatment'])
        
        with col2:
            st.markdown("#### 🛡️ Prevention")
            st.write(disease['prevention'])
            
            # Probability distribution
            st.markdown("#### 📈 Probability Distribution")
            probs = prediction[0]
            for i, (class_name, prob) in enumerate(zip(index_to_class.values(), probs)):
                class_display = class_name.replace('Potato___', '').replace('_', ' ')
                st.write(f"**{class_display}:** {prob*100:.1f}%")
                st.progress(float(prob))
        
        # Recommendations based on confidence
        st.markdown("### 💡 Recommendations")
        if confidence >= 80:
            st.success("✅ High confidence prediction! You can rely on this result.")
        elif confidence >= 60:
            st.warning("⚠️ Moderate confidence. Consider getting a second opinion or testing another image.")
        else:
            st.error("❌ Low confidence. Please upload a clearer image or consult with an agricultural expert.")
        
        # Action buttons
        st.markdown("### 🔄 Next Steps")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📷 Upload Another Image"):
                st.rerun()
        
        with col2:
            if st.button("📊 View Model Stats"):
                st.sidebar.markdown("### 📈 Detailed Model Performance")
                st.sidebar.markdown("""
                - **Early Blight Detection**: 96.50% accuracy
                - **Late Blight Detection**: 57.00% accuracy  
                - **Healthy Detection**: 100.00% accuracy
                - **Overall Accuracy**: 78.42%
                """)
        
        with col3:
            if st.button("📞 Contact Expert"):
                st.info("For professional agricultural advice, contact your local agricultural extension office.")

else:
    # Instructions when no image is uploaded
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Upload an Image**: Click the file uploader above and select a potato leaf image
    2. **Wait for Analysis**: Our AI model will process the image automatically
    3. **View Results**: Get instant classification with confidence scores and recommendations
    
    ### 🌱 About Potato Diseases
    
    **Early Blight**: Caused by Alternaria solani fungus, appears as dark spots with concentric rings.
    
    **Late Blight**: Caused by Phytophthora infestans, appears as water-soaked lesions that turn dark brown.
    
    **Healthy**: No visible disease symptoms, normal green coloration.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🥔 Potato Disease Detection System | Powered by Deep Learning | Built with Streamlit</p>
    <p><small>For educational purposes only. Always consult with agricultural experts for important decisions.</small></p>
</div>
""", unsafe_allow_html=True)
