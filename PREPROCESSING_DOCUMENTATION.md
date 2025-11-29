# Pre-processing Techniques Used (Differentiator)

## Overview
This document details all pre-processing techniques applied to improve extraction accuracy for diverse bill formats, including multilingual and handwritten documents.

## Pre-processing Pipeline

### 1. **Image Resizing & Resolution**
- **Max Side**: 2000px (increased from 1024px for complex bills)
- **Target DPI**: 300 DPI
- **Minimum Width**: 1200px (upscaled if smaller)
- **Purpose**: Ensures sufficient resolution for OCR accuracy

### 2. **Grayscale Conversion**
- Convert RGB images to grayscale (L mode)
- Reduces processing complexity while maintaining text clarity

### 3. **Denoising**
- **Method**: Fast Non-Local Means Denoising (OpenCV)
- **Parameters**: h=10, templateWindowSize=7, searchWindowSize=21
- **Purpose**: Removes noise from scanned documents

### 4. **Brightness Normalization**
- **Target Brightness**: 128 (mid-gray)
- **Adjustment**: ±30 threshold for normalization
- **Purpose**: Handles overexposed/underexposed scans

### 5. **Deskewing**
- **Method**: Hough Line Transform + Rotation Correction
- **Angle Detection**: Median angle from detected lines
- **Purpose**: Corrects skewed scans (common in mobile captures)

### 6. **Adaptive Thresholding (Multi-Method)**
- **Primary**: OTSU Threshold (automatic threshold selection)
- **Fallback 1**: Adaptive Gaussian Threshold (for varying lighting)
- **Fallback 2**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Purpose**: Handles varying contrast across document

### 7. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- **Clip Limit**: 2.0-3.0 (higher for handwritten)
- **Tile Grid Size**: 8x8
- **Purpose**: Enhances local contrast without over-amplifying noise

### 8. **Auto-Binary for Faded Text**
- **Detection**: Mean brightness > 200
- **Action**: Aggressive thresholding (240 threshold)
- **Purpose**: Recovers faded/light text

### 9. **Morphological Operations**
- **Operation**: Closing (dilation + erosion)
- **Kernel**: 3x3 rectangular
- **Purpose**: Connects broken characters (common in low-quality scans)

### 10. **Multilingual Enhancement**
- **Method**: Sharpening filter (3x3 kernel)
- **Purpose**: Improves character edge definition for non-English scripts
- **Impact**: Better recognition of Devanagari, Tamil, Telugu, etc.

### 11. **Handwritten Text Enhancement**
- **Method**: Enhanced CLAHE (clipLimit=3.0)
- **Purpose**: Improves contrast for handwritten text (typically lower contrast)
- **Impact**: Better extraction from handwritten bills

### 12. **Minimum Width Enforcement**
- **Threshold**: 1200px width
- **Method**: Upscaling with cubic interpolation
- **Purpose**: Ensures OCR has sufficient resolution

## Pre-processing Selection Logic

The pipeline uses adaptive selection:

1. **OTSU Threshold** → If good contrast (mean 50-200)
2. **Adaptive Threshold** → If OTSU fails
3. **CLAHE** → Final fallback

This ensures optimal pre-processing for each document type.

## Performance Optimizations

- **Fast Mode**: Skips expensive operations (denoising, deskewing, morphological)
- **Caching**: Intermediate results cached for repeated processing
- **Early Exit**: Blank page detection skips processing

## Differentiator Highlights

1. **Multilingual Support**: Sharpening filter enhances non-English character recognition
2. **Handwritten Support**: Enhanced CLAHE improves handwritten text extraction
3. **Adaptive Methods**: Multiple thresholding methods with automatic selection
4. **Faded Text Recovery**: Auto-binary mode for light/faded documents

## Usage

```python
from src.preprocessing.image_utils import preprocess_image_for_ocr

# Standard preprocessing
processed = preprocess_image_for_ocr(image, max_side=2000, fast_mode=False)

# With multilingual/handwritten enhancement (default)
processed = preprocess_image_for_ocr(
    image,
    max_side=2000,
    enhance_for_multilingual=True,
    enhance_for_handwritten=True
)
```

