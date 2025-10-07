# Plant Disease Detection System

A deep learning-based system for detecting and classifying plant diseases from leaf images using Meta's DINOv2 vision transformer.

---

## Table of Contents
1. [What This Project Does](#what-this-project-does)
2. [What is DINOv2](#what-is-dinov2)
3. [Project Architecture](#project-architecture)
4. [How It Works](#how-it-works)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Current Limitations](#current-limitations)
8. [Recommended Improvements](#recommended-improvements)
9. [File Structure](#file-structure)

---

## What This Project Does

This system:
1. **Classifies plant diseases** from images of plant leaves
2. **Provides confidence scores** for predictions
3. **Shows top 5 possible diagnoses** to help with uncertain cases
4. **Offers a web interface** using Gradio for easy image upload and prediction

### Key Features
- Uses state-of-the-art DINOv2 vision transformer
- Transfer learning from pre-trained model
- Web-based interface for easy access
- Multi-class classification support
- Confidence-based prediction quality assessment

---

## What is DINOv2?

**DINOv2 (Distillation with NO labels v2)** is a vision transformer model developed by Meta AI Research.

### Key Characteristics:
- **Self-supervised learning**: Trained on 142 million images without manual labels
- **Vision Transformer architecture**: Uses attention mechanisms instead of convolutions
- **Rich feature extraction**: Learns general visual features applicable to many tasks
- **Transfer learning ready**: Pre-trained features work well for downstream tasks

### Why DINOv2 for Plant Disease Detection?
1. **Strong visual understanding**: Pre-trained on diverse images, understands textures, patterns, shapes
2. **Transfer learning**: Saves time and resources compared to training from scratch
3. **Fine-grained features**: Good at detecting subtle differences (like disease symptoms)
4. **No ImageNet bias**: Trained on broader dataset than traditional ImageNet models

### Model Variants Used:
- **Small (vits14)**: 22M parameters, 384-dim features - Fast inference
- **Base (vitb14)**: 86M parameters, 768-dim features - Balanced (currently used)
- **Large (vitl14)**: 304M parameters, 1024-dim features - Best accuracy but slower

---

## Project Architecture

```
Input Image → Preprocessing → DINOv2 Backbone → Classification Head → Disease Prediction
                  ↓                    ↓                   ↓
             Resize 224x224      Extract Features    Dense Layers
             Normalize          (768 dimensions)    + Dropout
```

### Model Components:

1. **DINOv2 Backbone** (Frozen during initial training)
   - Pre-trained vision transformer
   - Extracts 768-dimensional feature vectors
   - Contains attention mechanisms for feature extraction

2. **Classification Head** (Trainable)
   - Linear layer: 768 → 384 dimensions
   - ReLU activation
   - Dropout (0.3) for regularization
   - Linear layer: 384 → num_classes
   - Softmax for probability distribution

---

## How It Works

### 1. Data Preparation (`prepare_data.py`)
- Scans dataset folder for disease class subfolders
- Splits data into train (70%), validation (15%), test (15%)
- Creates CSV files mapping image paths to class IDs
- Saves class names to JSON file
- Uses stratified splitting to maintain class balance

### 2. Model Training (`train.py`)
**Current Approach:**
- Freezes DINOv2 backbone weights
- Only trains classification head
- Uses cross-entropy loss
- Adam optimizer with learning rate 0.001
- Trains for 15 epochs
- Saves best model based on validation accuracy

**Data Augmentation Applied:**
- Resize to 224×224
- Random horizontal flip
- Random rotation (±15°)
- Normalization with ImageNet statistics

### 3. Prediction (`predict.py`)
- Loads trained model weights
- Preprocesses input image (resize, normalize)
- Passes through model
- Applies softmax for probabilities
- Returns top-k predictions with confidence scores

### 4. Web Interface (`app.py`)
- Gradio-based web UI
- Image upload functionality
- Displays primary diagnosis with confidence
- Shows alternative possibilities
- Color-coded confidence levels:
  - **High Confidence**: >90%
  - **Moderate Confidence**: 70-90%
  - **Low Confidence**: <70%

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd plant_disease_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Structure
```
archive/
├── class1_disease_name/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2_disease_name/
│   ├── image1.jpg
│   └── ...
└── ...
```

---

## Usage

### 1. Prepare Data
```bash
cd src
python prepare_data.py
```
This creates `train.csv`, `val.csv`, `test.csv`, and `classes.json`

### 2. Train Model
```bash
python train.py
```
This trains the model and saves `best_model.pth`

### 3. Run Web Interface
```bash
python app.py
```
Access at `http://localhost:7860`

### 4. Make Predictions Programmatically
```python
from predict import DiseasePredictor

predictor = DiseasePredictor('best_model.pth', 'classes.json')
results = predictor.predict('path/to/image.jpg', top_k=5)

for result in results:
    print(f"{result['disease']}: {result['confidence']:.2f}%")
```

---

## Current Limitations

### Why Model May Fail on New Black Rot Images:

1. **Overfitting to Training Data**
   - Model memorized specific training images rather than learning general patterns
   - Limited data augmentation causes poor generalization

2. **Frozen Backbone**
   - DINOv2 features not adapted to plant disease specifics
   - Only classification head learns disease patterns

3. **Limited Augmentation**
   - Only flip and rotation applied
   - Real-world variations (lighting, scale, background) not covered

4. **No Test-Time Augmentation**
   - Predictions based on single view of image
   - Doesn't average over multiple transformations

5. **Dataset Limitations**
   - May have limited variety within each disease class
   - Training images may be too similar (same background, lighting, etc.)

---

## Recommended Improvements

### Priority 1: Enhanced Data Augmentation

**Add to training:**
```python
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3)
```

**Why:** Helps model learn disease patterns regardless of lighting, position, scale

### Priority 2: Two-Stage Training

**Stage 1: Train with frozen backbone (done)**
- Fast initial learning
- Classification head adapts to task

**Stage 2: Fine-tune entire model**
```python
# Unfreeze backbone
model = create_model(num_classes, freeze_backbone=False)
model.load_state_dict(torch.load('best_model.pth'))

# Use lower learning rate
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

**Why:** Adapts DINOv2 features specifically for plant diseases

### Priority 3: Test-Time Augmentation (TTA)

```python
def predict_with_tta(image_path, num_augmentations=5):
    predictions = []
    for _ in range(num_augmentations):
        # Apply random augmentation
        augmented_image = augment(image)
        pred = model(augmented_image)
        predictions.append(pred)

    # Average predictions
    return torch.mean(torch.stack(predictions), dim=0)
```

**Why:** More robust predictions by averaging over multiple views

### Priority 4: Learning Rate Scheduling

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
```

**Why:** Adapts learning rate when validation accuracy plateaus

### Priority 5: Additional Techniques

1. **Class Weighting**: Handle imbalanced datasets
2. **Mixup/CutMix**: Advanced augmentation during training
3. **Ensemble Models**: Combine multiple model predictions
4. **Larger Model**: Use DINOv2-large for better capacity
5. **More Training Data**: Collect more diverse samples per class
6. **Focal Loss**: Better handle hard-to-classify examples
7. **Gradient Accumulation**: Simulate larger batch sizes

### Priority 6: Evaluation & Debugging

1. **Confusion Matrix**: See which diseases are confused
2. **Per-Class Metrics**: Identify weak classes
3. **Error Analysis**: Manually inspect misclassified images
4. **Feature Visualization**: Use t-SNE/UMAP to visualize learned features
5. **Grad-CAM**: Visualize which image regions influence predictions

---

## File Structure

```
plant_disease_detection/
├── archive/                  # Dataset folder (disease class subfolders)
├── src/
│   ├── config.py            # Centralized configuration
│   ├── prepare_data.py      # Data splitting and CSV generation
│   ├── dataset.py           # PyTorch Dataset class
│   ├── model.py             # DINOv2 model architecture
│   ├── train.py             # Training script
│   ├── predict.py           # Prediction/inference module
│   └── app.py               # Gradio web interface
├── venv/                    # Virtual environment
├── requirements.txt         # Python dependencies
├── train.csv               # Training data paths (generated)
├── val.csv                 # Validation data paths (generated)
├── test.csv                # Test data paths (generated)
├── classes.json            # Class names mapping (generated)
├── best_model.pth          # Saved model weights (generated)
└── README.md               # This file
```

### File Descriptions:

**`config.py`**: Centralized configuration for easy hyperparameter tuning
- Data split ratios
- Model architecture settings
- Training hyperparameters
- File paths

**`prepare_data.py`**: Dataset organization
- Scans dataset folders
- Creates stratified train/val/test splits
- Generates CSV files with image paths and labels

**`dataset.py`**: PyTorch Dataset implementation
- Loads images from CSV
- Applies transformations
- Returns tensors for training

**`model.py`**: Model architecture definition
- Loads DINOv2 from torch.hub
- Creates classification head
- Handles freezing/unfreezing backbone

**`train.py`**: Training loop
- Loads datasets
- Trains model with validation
- Saves best model checkpoint
- Tracks metrics

**`predict.py`**: Inference module
- Loads trained model
- Preprocesses images
- Returns top-k predictions with confidence

**`app.py`**: Web interface
- Gradio UI setup
- Image upload handling
- Result display formatting

---

## Technical Details

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 15
- **Image Size**: 224×224
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Model Architecture Details
- **Input**: RGB image (3, 224, 224)
- **DINOv2 Backbone**: Vision Transformer Base (86M params)
- **Feature Dimension**: 768
- **Classification Head**:
  - Linear(768, 384)
  - ReLU
  - Dropout(0.3)
  - Linear(384, num_classes)

---

## Contributing

To improve the model:
1. Implement the recommended improvements
2. Test on held-out test set
3. Document results and findings
4. Share insights on what worked best

---

## Dataset

This project uses a plant disease dataset containing images of various plant species and their associated diseases.

### Dataset Structure:
- **Plants included**: Apple, Cassava, Cherry, Chili, Coffee, and more
- **Disease types**: Black rot, rust, scab, bacterial blight, mosaic disease, powdery mildew, leaf spot, etc.
- **Healthy samples**: Includes healthy leaves for comparison

### Citation:
If you're using the PlantVillage Dataset or similar, please cite:

```bibtex
@article{hughes2015open,
  title={An open access repository of images on plant health to enable the development of mobile disease diagnostics},
  author={Hughes, David P and Salath{\'e}, Marcel},
  journal={arXiv preprint arXiv:1511.08060},
  year={2015}
}
```

**Dataset Source**: Please specify your dataset source (e.g., Kaggle, PlantVillage, custom collection)

### Data Availability:
Due to size constraints, the dataset is not included in this repository. To use this code:
1. Download the dataset from your source
2. Place it in the `archive/` folder
3. Organize as: `archive/[class_name]/[images]`

---

## License

This project uses Meta's DINOv2 model, which is licensed under Apache 2.0.

---

## Acknowledgments

- **DINOv2**: Meta AI Research - https://github.com/facebookresearch/dinov2
  - Paper: "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)
- **PlantVillage Dataset**: Hughes & Salathe (2015) - Open access plant disease images
- **PyTorch**: Deep learning framework - https://pytorch.org
- **Gradio**: Web interface library - https://gradio.app

---

## Troubleshooting

### Model gives wrong predictions on new images
- **Cause**: Overfitting, limited augmentation
- **Solution**: Implement Priority 1-3 improvements above

### Out of memory errors
- **Cause**: Batch size too large or large model variant
- **Solution**: Reduce batch size in config.py, use smaller model variant

### Training is very slow
- **Cause**: No GPU, large batch size, or unfrozen backbone
- **Solution**: Use GPU, reduce batch size, keep backbone frozen initially

### Low validation accuracy
- **Cause**: Model complexity, insufficient training, poor hyperparameters
- **Solution**: Train longer, tune learning rate, add regularization

---

## Contact & Support

For questions or issues, please open an issue in the GitHub repository.
