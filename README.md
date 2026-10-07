# 🖼️ Image Analytics Lab

An interactive **Computer Vision and Image Analytics web application** built using Python and Streamlit.

The application provides a simple and user-friendly interface to explore different image processing, face detection, face analysis, and face verification techniques.

## 🚀 Live Demo

👉 **[Try Image Analytics Lab](https://imageprocessinglab-kx2nyoscwhkdppmngx9gwn.streamlit.app/)**

---

## 📌 Features

### 🔍 1. Template Matching
- Detects a specific template/object inside a larger image.
- Uses OpenCV's `matchTemplate()` method.
- Supports multiple template matching techniques.
- Displays the detected region with a bounding box.
- Shows matching confidence.

### 👤 2. Viola-Jones Face Detection
- Detects human faces using the Viola-Jones algorithm.
- Uses OpenCV Haar Cascade Classifiers.
- Draws bounding boxes around detected faces.
- Allows adjustment of detection parameters.

### 🧠 3. DeepFace Analysis
- Performs AI-based facial analysis.
- Provides:
  - Age estimation
  - Gender analysis
  - Emotion detection
  - Race/ethnicity analysis
- Supports different face detector backends.

### 🔐 4. FaceNet Face Verification
- Compares two face images.
- Generates facial embeddings using FaceNet.
- Calculates the distance between embeddings.
- Determines whether the two faces are likely to belong to the same person.

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **OpenCV**
- **NumPy**
- **Pillow**
- **DeepFace**
- **FaceNet**
- **PyTorch**
- **TorchVision**

---

## 🏗️ Application Structure

```text
Image Analytics Lab
│
├── 🏠 Overview
│
├── 🔍 Template Matching
│
├── 👤 Viola-Jones Face Detection
│
├── 🧠 DeepFace Analysis
│
└── 🔐 FaceNet Face Verification
