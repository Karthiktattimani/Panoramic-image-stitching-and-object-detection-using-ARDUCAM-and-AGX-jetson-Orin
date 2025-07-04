# Panoramic-image-stitching-and-object-detection-using-ARDUCAM-and-AGX-jetson-Orin

## 📌 Introduction

This project demonstrates real-time panoramic image stitching and object detection using dual Arducam cameras interfaced with the **NVIDIA Jetson AGX Orin**. The main objective is to capture simultaneous images from two cameras, stitch them to form a wider field of view, and detect objects within the composite image.

Such a system is especially useful for applications like:
- Smart surveillance systems
- Autonomous vehicles
- Robotics
- Augmented reality


## 📂 Project Structure

```
project-root/
├── code
│   ├── final.txt
│   ├── object_detection.py
│   └── stitch_images.py
├── dependences
│   ├── camera
│   ├── object
├── images
│   ├── camera captured
│   ├── final images
│   └── stitched images
└── README.md
```


## 🛠️ Hardware Used

- **NVIDIA Jetson AGX Orin**
- **Arducam Cameras**  
  Supported Models:
  - 8MP IMX219
  - 12MP IMX477 (used)
  - 16MP IMX519
- Camera interface cables
- Power and peripheral accessories


> 🔧 **Driver Installation:**  
> Before using the Arducam camera, install the proper driver and configuration for your camera model.

### 📷 Camera Setup for NVIDIA Jetson AGX Orin
Follow the appropriate link based on your camera model:

- **[8MP IMX219 Camera Setup](https://docs.arducam.com/Nvidia-Jetson-Camera/Nvidia-Jetson-Orin-Series/NVIDIA-Jetson-AGX-Orin/Quick-Start-Guide/#8mp-imx219-camera)**
- **[12MP IMX477 Camera Setup](https://docs.arducam.com/Nvidia-Jetson-Camera/Nvidia-Jetson-Orin-Series/NVIDIA-Jetson-AGX-Orin/Quick-Start-Guide/#12mp-imx477-camera)**
- **[16MP IMX519 Camera Setup](https://docs.arducam.com/Nvidia-Jetson-Camera/Nvidia-Jetson-Orin-Series/NVIDIA-Jetson-AGX-Orin/Quick-Start-Guide/#16mp-imx519-camera)**

Each guide contains steps for:
- Enabling camera support on the Jetson
- Installing necessary kernel drivers
- Running test capture scripts


## 🧠 Methodology

### 1. **Image Capture**
- Two high-resolution images are captured simultaneously from Arducam modules (e.g., 4032x3040 resolution).
- Synchronized capture ensures spatial continuity for stitching.

### 2. **Keypoint Detection**
- Feature detection is performed using **ORB (Oriented FAST and Rotated BRIEF)** for both images.

### 3. **Feature Matching**
- Keypoints are matched to align overlapping areas using techniques like Brute-Force matching.

### 4. **Image Stitching**
- Homography matrix is computed and used to blend the two images into a seamless panoramic view.

### 5. **Object Detection**
- **YOLOv10** – High-speed, real-time performance
- Final detection is applied to the stitched image.


## ✅ Conclusion

- Achieved **real-time panoramic stitching** and **object detection** on the Jetson AGX Orin.
- **YOLOv10** provided efficient real-time performance on edge devices.
- The system is a strong prototype for **smart surveillance** and **vision-based automation**.


## 🧑‍💻 Team Members

| Name              |
|-------------------|
| Karthik Tattimani | 
| Vinay Math        | 
| Shrishail Anagwadi| 
| Arunkumar Kalloli | 


For detailed troubleshooting and information about the arducam on jetson agx orin refer the link: 
https://docs.arducam.com/Nvidia-Jetson-Camera/Nvidia-Jetson-Orin-Series/NVIDIA-Jetson-AGX-Orin/Quick-Start-Guide/#for-imx219-camera-module