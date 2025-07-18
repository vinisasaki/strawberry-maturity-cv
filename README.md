# Strawberry Maturity Classification (Classical Computer Vision)

A Classical Computer Vision project using traditional techniques—no machine learning or neural networks—to segment and classify strawberry ripeness stages from images.

---

## 📝 Description

This repository contains:

- **strawberry_maturity_cv.py**  
  Main script implementing:  
  1. **Pre-processing** (filtering, histogram equalization)  
  2. **Segmentation** (thresholding, morphological operations, Watershed)  
  3. **Feature Extraction** (area, mean color, contours, bounding boxes)  
  4. **Rule-based Classification** (green, mid-ripe, ripe)

- **dataset/**  
  Folder with sample strawberry images for testing and evaluation.

---

## 📂 Structure

- **strawberry_maturity_cv.py**  
  Main training/evaluation script

- **dataset/**  
  Strawberry image dataset

- **.gitignore**  
  Specifies files/folders to ignore

- **README.md**
  This documentation file
---

## 🚀 How to Run

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/vinisasaki/ComputerVision/strawberry-maturity.git
   cd strawberry-maturity-cv
2. **(Optional) Set up a virtual environment:**  
    ```bash
    python3 -m venv venv
    source venv/bin/activate    # Linux/macOS
    venv/Scripts/activate       # Windows
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
4. **Execute the script:**
    ```bash
    python strawberry_maturity_cv.py
---
## 📊 Dataset
Images were obtained from StrawDI:

StrawDI. “The Strawberry Digital Images Data Set”. 2018. Available at: https://strawdi.github.io/ (accessed 06/24/2025).
