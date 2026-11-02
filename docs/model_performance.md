# BloomAI: Flower Classification Model Performance & Evaluation

This document outlines the verified evaluation metrics and architecture of the models trained on the Five-Class Flower Dataset (Daisy, Dandelion, Rose, Sunflower, Tulip).

## 1. Selected Production Model: ResNet-18 (Transfer Learning)

The primary model selected for deployment in BloomAI is **ResNet-18**, pre-trained on ImageNet and fine-tuned for 5-class flower classification. ResNet-18 demonstrated superior generalization and accuracy compared to custom Convolutional Neural Networks and Multi-Layer Perceptrons.

### Key Specifications:
- **Architecture**: Deep Residual Learning (18 layers)
- **Classifier Head**:
  - `Linear(in_features=512, out_features=256, bias=True)`
  - `ReLU(inplace=True)`
  - `Dropout(p=0.5)`
  - `Linear(in_features=256, out_features=5, bias=True)`
- **Input Dimension**: `(3, 224, 224)`
- **Preprocessing Pipeline**:
  1. `Resize(256)`
  2. `CenterCrop(224)`
  3. `ToTensor()`
  4. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Output Classes**: `['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']`
- **Checkpoint File**: `best_resnet_model.pth` (Size: 43.22 MB / 45,317,723 bytes)

---

## 2. Comparative Evaluation Results

Evaluation performed on the held-out test split of the dataset (735 test samples):

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP (Baseline)** | 48.98% | 49.25% | 49.12% | 47.90% | 24.63 MB |
| **Original CNN** | 52.65% | 52.78% | 52.55% | 51.84% | 98.37 MB |
| **Optimized CNN** | 67.21% | 67.36% | 67.04% | 66.97% | 4.63 MB |
| **ResNet-18 (Selected)** | **87.21%** | **87.25%** | **87.20%** | **87.20%** | **43.22 MB** |

### Verified Test Report for ResNet-18:
```json
{
  "test_accuracy": 0.8721088435374149,
  "test_precision": 0.8725001415630436,
  "test_recall": 0.8720217810072682,
  "test_f1_score": 0.8719699447150348
}
```

---

## 3. Why ResNet-18 Excels in Flower Classification

1. **Feature Reusability**: Pre-trained representations from ImageNet provide edge, texture, and petal structure detectors out of the box, allowing the model to quickly adapt to subtle differences between floral species (such as petal density and stamen shapes).
2. **Residual Connections**: Skip connections (`F(x) + x`) prevent the vanishing gradient problem, allowing robust gradient propagation throughout the 18 layers.
3. **Generalization & Regularization**: The combination of `Dropout(p=0.5)` in the classification head and batch normalization layers ensures the model avoids overfitting on limited training samples, achieving an **87.20% Macro F1-score**.

