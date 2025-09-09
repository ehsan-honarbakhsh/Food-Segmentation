# FoodSeg103 Image Segmentation with U-Net

## Overview

This project implements an image segmentation algorithm for the FoodSeg103 dataset using a U-Net model built with TensorFlow. The algorithm segments food images into 104 classes (including background) to identify various food items and ingredients. The implementation includes data preprocessing, a custom U-Net architecture with regularization, and evaluation using Mean Intersection over Union (mIoU). It also provides visualizations of segmentation results with annotated masks.

The project is designed to run on platforms like Google Colab, leveraging TensorFlow's data pipeline for efficient dataset handling and model training. The code processes images and masks from the FoodSeg103 dataset, trains a U-Net model, and evaluates its performance on a test set.

Key features:

- Custom U-Net architecture with batch normalization, dropout, and L2 regularization.
- Efficient data loading and preprocessing with TensorFlow's `tf.data` API.
- Evaluation using a custom MeanIoU metric.
- Visualization of segmentation masks with class annotations.
- Support for the FoodSeg103 dataset with 104 classes.

## Requirements

- Python 3.12+
- Libraries:
  - TensorFlow (`tensorflow>=2.10`)
  - NumPy (`numpy`)
  - Matplotlib (`matplotlib`)
  - Pillow (`pillow`)
  - SciPy (`scipy`)

Install dependencies using:

```
pip install tensorflow numpy matplotlib pillow scipy
```

## Dataset

The code is designed for the FoodSeg103 dataset, which contains food images and corresponding segmentation masks. The dataset structure is expected as follows:

- Root directory: `/content/drive/MyDrive/FoodSeg103`
- Images: `Images/img_dir/train/*.jpg` (training) and `Images/img_dir/test/*.jpg` (validation/test)
- Masks: `Images/ann_dir/train/*.png` (training) and `Images/ann_dir/test/*.png` (validation/test)
- Text files: `ImageSets/train.txt` and `ImageSets/test.txt` containing image IDs

The dataset is split into:

- Training set: IDs from `train.txt`
- Validation and test sets: IDs from `test.txt`, with 10% allocated to testing and the rest for validation

Ensure the dataset is accessible (e.g., mounted on Google Drive for Colab) or adjust paths accordingly.

## Usage

1. Place the FoodSeg103 dataset in the specified directory (e.g., `/content/drive/MyDrive/FoodSeg103`).

   Note: If using Google Colab, ensure the script is configured to access Google Drive (`/content/drive/MyDrive`).

2. The script will:
   - Load and preprocess the dataset (resize to 256x256, normalize images, etc.).
   - Create TensorFlow datasets for training, validation, and testing.
   - Define and compile a U-Net model.
   - (Note: Training code is not included in the provided snippet; add model training as needed.)
   - Predict segmentation masks for a sample image and test set.
   - Compute mIoU on the test set.
   - Visualize original images, ground truth masks, and predicted masks with class annotations.

Customization:

- Adjust `img_height`, `img_width` (default: 256x256) for different resolutions.
- Modify `batch_size` (default: 16) based on available memory.
- Tune `l2_lambda` (default: 0.01) for regularization strength.
- Change the number of classes (`num_class`, default: 104) if using a different dataset.
- Add training code with an optimizer (e.g., Adam) and loss function (e.g., sparse categorical crossentropy).

## Algorithm Details

### 1. Data Preprocessing

- **Image IDs**: Read from `train.txt` and `test.txt` to create lists of image and mask paths.
- **Dataset Split**: Randomly split `test.txt` IDs into validation (90%) and test (10%) sets.
- **Data Loading**:
  - Images: Load JPG files, decode to RGB, resize to 256x256, normalize to [0, 1].
  - Masks: Load PNG files, decode to grayscale, resize with nearest-neighbor interpolation, cast to uint8.
- **TensorFlow Dataset**:
  - Create `tf.data.Dataset` for training, validation, and testing.
  - Apply shuffling (training only), batching, and prefetching for efficiency.

### 2. U-Net Model

- **Architecture**:
  - **Encoder**: Four levels (64, 128, 256, 512 filters) with two Conv2D (3x3, same padding), BatchNormalization, ReLU, and MaxPooling2D. Dropout (0.1–0.2) after pooling.
  - **Bottleneck**: 1024 filters with dilated convolutions (dilation_rate=2), L2 regularization, BatchNormalization, ReLU, and Dropout (0.2).
  - **Decoder**: Four levels (512, 256, 128, 64 filters) with UpSampling2D, skip connections (Concatenate), two Conv2D (3x3), BatchNormalization, ReLU, and Dropout (0.3–0.4). L2 regularization in the final level.
  - **Output**: 1x1 Conv2D with `num_classes` filters and softmax activation for pixel-wise classification.
- **Input Shape**: (256, 256, 3)
- **Output Shape**: (256, 256, 104)

### 3. Custom MeanIoU Metric

- Extends `tf.keras.metrics.Metric` to compute Mean Intersection over Union.
- Preprocesses predictions (argmax to class indices) and ground truth (cast to int32, squeeze if needed).
- Uses TensorFlow's built-in `MeanIoU` for computation.

### 4. Prediction and Visualization

- **Single Image Prediction**:
  - Load a test image (e.g., `test2.jpg`), resize, normalize, and predict the mask.
  - Visualize the original image and predicted mask (jet colormap).
- **Test Set Evaluation**:
  - Predict masks for test dataset images.
  - Compute overall mIoU.
  - Visualize up to 7 samples (original image, ground truth mask, predicted mask).
- **Annotated Masks**:
  - Use SciPy's `label` to identify regions in predicted masks.
  - Annotate each region with the corresponding class name from a predefined list of 104 classes (e.g., "background", "candy", "steak").

### 5. Evaluation Metric

- **MeanIoU**: Measures overlap between predicted and ground truth masks across all classes.

## Results

The script outputs:

- **Visualizations**:
  - Single image: Original and predicted mask.
  - Test set: Up to 7 triplets (original image, ground truth mask, predicted mask with class annotations).
- **Metric**: Overall mIoU on the test set, indicating segmentation accuracy.

The U-Net model is expected to achieve reasonable segmentation performance on the FoodSeg103 dataset, with mIoU depending on training quality and dataset complexity.

## Limitations

- **Computational Resources**: Large batch sizes or high-resolution images may require GPU support.
- **Class Imbalance**: Some classes may be underrepresented, affecting mIoU; consider class weighting.
- **Annotation**: Text annotations may overlap in dense masks; adjust fontsize or placement if needed.
