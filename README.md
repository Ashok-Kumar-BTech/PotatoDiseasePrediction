# Potato Disease Detection System 🥔

A deep learning-powered web application for detecting potato plant diseases using Convolutional Neural Networks (CNN). The system classifies potato leaves as healthy or affected by Early Blight or Late Blight diseases.

## 🎯 Features

- **CNN Model**: 4-layer convolutional neural network with max pooling
- **Three Classes**: Potato___Early_blight, Potato___Late_blight, Potato___healthy
- **Web Interface**: Interactive Streamlit application for image upload and classification
- **Performance Metrics**: Comprehensive evaluation with confusion matrix and classification report
- **Real-time Prediction**: Instant results with confidence scores and disease insights

## 📊 Model Performance

- **Overall Accuracy**: 78.42%
- **Early Blight Detection**: 96.50% accuracy
- **Late Blight Detection**: 57.00% accuracy  
- **Healthy Detection**: 100.00% accuracy

## 🏗️ Project Structure

```
temp-project/
├── training/
│   └── PlantVillage/
│       ├── train/           # Training dataset
│       └── val/             # Validation dataset
├── model/
│   ├── potato_disease_model.h5      # Trained model
│   ├── class_indices.json           # Class mapping
│   ├── evaluation_results.json      # Performance metrics
│   ├── confusion_matrix.png         # Visualization
│   └── training_history.png         # Training curves
├── train_model.py                   # Model training script
├── evaluate_model.py                # Model evaluation script
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🚀 Installation & Setup

1. **Clone/Download the project** to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional - model already included):
   ```bash
   python train_model.py
   ```

4. **Evaluate the model** (optional - results already saved):
   ```bash
   python evaluate_model.py
   ```

5. **Run the web application**:
   ```bash
   streamlit run app.py
   ```

The application will open in your browser at `http://localhost:8501`

## 📱 Using the Web Application

1. **Upload an Image**: Click the file uploader and select a potato leaf image (JPG, JPEG, PNG)

2. **View Results**: Get instant classification with:
   - Disease prediction (Early Blight, Late Blight, or Healthy)
   - Confidence percentage
   - Detailed disease information
   - Treatment recommendations
   - Prevention tips

3. **Analyze Confidence**: The app provides recommendations based on prediction confidence:
   - **≥80%**: High confidence - reliable result
   - **60-79%**: Moderate confidence - consider testing another image
   - **<60%**: Low confidence - upload a clearer image

## 🧬 Model Architecture

The CNN model consists of:

- **4 Convolutional Layers** with ReLU activation
- **Max Pooling** after each convolutional layer
- **Dropout layers** for regularization (0.5 and 0.3)
- **Dense layers**: 512 → 256 → 3 (output)
- **Softmax activation** for multi-class classification

### Training Parameters

- **Image Size**: 128x128 pixels
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: Rotation, shifts, flips, zoom

## 📈 Dataset

The model is trained on the PlantVillage dataset with potato leaf images:

- **Training Samples**: 1,721 images
- **Validation Samples**: 431 images
- **Classes**: 3 (Early Blight, Late Blight, Healthy)

## 🔧 Technical Details

### Dependencies

- **TensorFlow 2.16+**: Deep learning framework
- **Keras**: High-level neural network API
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning metrics
- **Pillow**: Image processing
- **Matplotlib/Seaborn**: Visualization

### Model Files

- `potato_disease_model.h5`: Trained Keras model in HDF5 format
- `class_indices.json`: Mapping between class names and indices
- `evaluation_results.json`: Detailed performance metrics

## 🎨 Web Interface Features

- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Processing**: Instant predictions
- **Visual Feedback**: Progress bars and confidence meters
- **Educational Content**: Disease information and recommendations
- **User-friendly**: Intuitive interface with clear instructions

## 📊 Evaluation Metrics

The evaluation script provides:

- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: Visual representation of predictions
- **Per-class Accuracy**: Individual class performance
- **Overall Accuracy**: General model performance

## 🌱 Disease Information

### Early Blight
- **Cause**: Alternaria solani fungus
- **Symptoms**: Dark brown spots with concentric rings
- **Treatment**: Fungicides, leaf removal, proper spacing

### Late Blight
- **Cause**: Phytophthora infestans
- **Symptoms**: Water-soaked lesions turning dark brown
- **Treatment**: Metalaxyl/copper fungicides, plant removal

### Healthy
- **Status**: No disease symptoms detected
- **Recommendation**: Continue good agricultural practices

## 🔬 Research & Development

This project demonstrates:

- **Computer Vision** applications in agriculture
- **Deep Learning** for plant disease detection
- **Web-based ML** deployment with Streamlit
- **End-to-end ML pipeline** from training to deployment

## 📝 Future Improvements

- **Enhanced Accuracy**: More training data and hyperparameter tuning
- **Additional Diseases**: Expand to more potato diseases
- **Mobile App**: Native mobile application
- **Real-time Detection**: Camera integration for live analysis
- **Geolocation**: Disease prevalence mapping

## ⚠️ Disclaimer

This system is for educational and assistance purposes only. Always consult with agricultural experts for critical farming decisions and disease management.

## 📧 Support

For questions or issues with the application, please refer to the code documentation or contact your local agricultural extension office for professional advice.

---

**Built with ❤️ using TensorFlow, Keras, and Streamlit**
