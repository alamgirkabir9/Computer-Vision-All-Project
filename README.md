# Computer Vision Projects Repository

A comprehensive collection of computer vision projects focusing on image classification using deep learning techniques and transfer learning approaches.

## 📋 Table of Contents
- [Overview](#overview)
- [Projects](#projects)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models & Techniques](#models--techniques)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This repository contains multiple computer vision projects implementing various image classification tasks using deep learning frameworks. The projects demonstrate different approaches including transfer learning, custom CNN architectures, and multi-class classification on diverse datasets.

## 🎯 Projects

### 1. **CIFAR-10 Classification**
- **Dataset**: CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Task**: Multi-class image classification
- **Image Size**: 32x32 pixels
- **Classes**: 10 categories

### 2. **Dog Breed Classification**
- **Dataset**: Custom dog breeds dataset
- **Task**: Multi-class breed identification
- **Breeds Included**:
  - Beagle
  - Bulldog
  - Dalmatian
  - German Shepherd
  - Husky
  - Poodle
  - Rottweiler
- **Image Size**: 32x32 pixels (resized)

### 3. **Ball Classification**
- **Dataset**: Sports ball classification dataset
- **Task**: Multi-class sports equipment identification
- **Classes**: 15 different types of balls
  - American Football
  - Baseball
  - Basketball
  - Billiard Ball
  - Bowling Ball
  - Cricket Ball
  - Football (Soccer)
  - Golf Ball
  - Hockey Ball
  - Hockey Puck
  - Rugby Ball
  - Shuttlecock
  - Table Tennis Ball
  - Tennis Ball
  - Volleyball
- **Image Size**: 32x32 pixels (resized)

### 4. **Pandas vs Bears Classification**
- **Dataset**: Custom binary classification dataset
- **Task**: Binary image classification
- **Classes**: Bears (label 0) and Pandas (label 1)
- **Image Size**: 32x32 pixels
## 📊 Datasets

### CIFAR-10
- **Source**: Official CIFAR-10 dataset
- **Format**: Pickled batch files
- **Size**: 50,000 training images, 10,000 test images
- **Loading**: Custom unpickle function for batch processing

### Custom Datasets
- **Format**: Directory-based organization
- **Structure**: 
  ```
  dataset/
  ├── class1/
  │   ├── image1.jpg
  │   └── image2.jpg
  └── class2/
      ├── image1.jpg
      └── image2.jpg
  ```
- **Supported Formats**: JPG, JPEG, PNG

## 🛠 Installation

### Prerequisites
```bash
python >= 3.7
```

### Required Libraries
```bash
pip install numpy
pip install pillow
pip install tensorflow  # or pytorch
pip install matplotlib
pip install scikit-learn
```

### Clone Repository
```bash
git clone https://github.com/yourusername/computer-vision-projects.git
cd computer-vision-projects
```

## 🚀 Usage

### Loading Custom Datasets

#### Method 1: Automatic Directory Loading
```python
from dataset_loader import load_dataset_from_directory

# Load dataset with automatic class detection
images, labels, label_names = load_dataset_from_directory(
    root_dir="path/to/dataset",
    image_size=(32, 32)
)
```

#### Method 2: Manual Loading
```python
from dataset_loader import load_images_from_folder

# Load specific classes
bear_images, bear_labels = load_images_from_folder(
    folder="path/to/bears", 
    label=0, 
    image_size=(32, 32)
)
```

### Loading Ball Classification Dataset
```python
from dataset_loader import load_dataset_from_directory

# Load ball classification dataset
train_dir = "/path/to/ball_classification/train"
data, labels, label_names = load_dataset_from_directory(train_dir)

print("✅ Loaded shape:", data.shape)
print("📌 Label names:", label_names)
# Output: 15 different ball types including american_football, baseball, basketball, etc.
```
```python
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load training batches
data_list = []
labels_list = []

for i in range(1, 6):
    batch = unpickle(f"data_batch_{i}")
    data_list.append(batch[b'data'])
    labels_list.extend(batch[b'labels'])

data = np.vstack(data_list)
labels = np.array(labels_list)
```

## 📁 Project Structure

```
computer-vision-projects/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── cifar10_classification.ipynb
│   ├── dog_breed_classification.ipynb
│   ├── ball_classification.ipynb
│   └── pandas_bears_classification.ipynb
├── src/
│   ├── dataset_loader.py
│   ├── models/
│   │   ├── cnn_models.py
│   │   └── transfer_learning.py
│   └── utils/
│       ├── preprocessing.py
│       └── visualization.py
├── data/
│   ├── cifar-10/
│   ├── dog-breeds/
│   ├── ball-classification/
│   └── pandas-bears/
└── results/
    ├── models/
    └── plots/
```

## 🧠 Models & Techniques

### Implemented Approaches
- **Custom CNN Architectures**
- **Transfer Learning** (VGG, ResNet, EfficientNet)
- **Data Augmentation**
- **Preprocessing Pipelines**

### Key Features
- **Automatic Dataset Loading**: Flexible functions for various dataset structures
- **Image Preprocessing**: Resizing, normalization, format conversion
- **Multi-class Classification**: Support for both binary and multi-class problems
- **Batch Processing**: Efficient handling of large datasets

## 📈 Results

### Performance Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Model Comparisons
Results and comparisons between different approaches are documented in individual project notebooks.

## 🔧 Key Functions

### `load_dataset_from_directory(root_dir, image_size)`
Automatically loads images from a directory structure with subdirectories as classes.

**Parameters:**
- `root_dir`: Path to the dataset root directory
- `image_size`: Tuple specifying target image dimensions

**Returns:**
- `images`: NumPy array of image data
- `labels`: NumPy array of integer labels
- `label_names`: List of class names

### `load_images_from_folder(folder, label, image_size)`
Loads images from a single folder with a specified label.

**Parameters:**
- `folder`: Path to the image folder
- `label`: Integer label for all images in the folder
- `image_size`: Tuple specifying target image dimensions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 Notes

- All images are automatically resized to 32x32 pixels for consistency
- RGB format is enforced for all loaded images
- Supported image formats: JPG, JPEG, PNG
- Directory structure should follow the pattern: `root/class_name/images`

## 🔬 Future Enhancements

- [ ] Add support for larger image sizes
- [ ] Implement advanced data augmentation techniques
- [ ] Add model ensemble methods
- [ ] Include object detection capabilities
- [ ] Add real-time inference pipeline

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or suggestions, please open an issue or contact [alomgirkabir720@gmail.com].

---

⭐ **If you find this repository helpful, please consider giving it a star!**
