import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = load_model('model/potato_disease_model.h5')

# Load class indices
with open('model/class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Create reverse mapping (index to class name)
index_to_class = {v: k for k, v in class_indices.items()}
class_names = [index_to_class[i] for i in range(len(index_to_class))]

print(f"Class names: {class_names}")

# Image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Prepare validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    'training/PlantVillage/val',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    classes=list(class_indices.keys()),
    class_mode='categorical',
    shuffle=False
)

print(f"Validation samples: {val_generator.samples}")

# Get predictions
print("Making predictions...")
y_pred_prob = model.predict(val_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_generator.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Generate classification report
print("\nClassification Report:")
print("=" * 60)
report = classification_report(y_true, y_pred, 
                              target_names=class_names,
                              digits=4)
print(report)

# Generate confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
plt.show()

# Calculate per-class accuracy
print("\nPer-class Accuracy:")
print("=" * 40)
for i, class_name in enumerate(class_names):
    class_mask = (y_true == i)
    if np.sum(class_mask) > 0:
        class_accuracy = np.sum(y_pred[class_mask] == i) / np.sum(class_mask)
        print(f"{class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

# Save evaluation results
evaluation_results = {
    'overall_accuracy': float(accuracy),
    'classification_report': report,
    'confusion_matrix': cm.tolist(),
    'per_class_accuracy': {}
}

for i, class_name in enumerate(class_names):
    class_mask = (y_true == i)
    if np.sum(class_mask) > 0:
        class_accuracy = np.sum(y_pred[class_mask] == i) / np.sum(class_mask)
        evaluation_results['per_class_accuracy'][class_name] = float(class_accuracy)

with open('model/evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

print("\nEvaluation results saved to model/evaluation_results.json")
print("Confusion matrix saved to model/confusion_matrix.png")
