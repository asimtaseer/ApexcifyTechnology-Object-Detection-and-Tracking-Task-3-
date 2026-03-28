# Task 3 – Apexcify Technology Task 3

---

## 📌 Introduction

This project is part of **Task 3 at Apexcify Technology**, where the goal is to perform **real-time object detection and tracking** using advanced computer vision techniques. 


The system uses:

* **YOLOv8 (Ultralytics)** for object detection
* **DeepSORT** for multi-object tracking

The model detects only **persons (class 0 from COCO dataset)** and assigns a **unique ID** to each individual. These IDs remain consistent across frames, allowing accurate tracking of people in a video.

---

## ⚙️ How It Works

1. Load a pre-trained YOLOv8 model it will download it in first run  (`yolov8x.pt`)
2. Read video input frame-by-frame
3. Detect objects (filter only **person class**)
4. Pass detections to DeepSORT tracker
5. Assign unique IDs to each person
6. Draw bounding boxes and tracking IDs
7. Display the processed video output
---

## ▶️ How to Run the Project

Follow these steps to run the code:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/asimtaseer/ApexcifyTechnology-Object-Detection-and-Tracking-Task-3-.git
cd ApexcifyTechnology-Object-Detection-and-Tracking-Task-3-

```

### 2️⃣ Install Required Libraries

```bash
pip install ultralytics opencv-python numpy deep-sort-realtime
```

### 3️⃣ Add Your Video File

* Place your input video file in the project directory
* Rename it as:

```bash
1.mp4
```

### 4️⃣ Run the Script

```bash
python main.py
```

### 5️⃣ Exit

* Press **`q`** to stop the video

---

## 📦 Required Libraries

Make sure you have the following Python libraries installed:

* `Python version 10`
* `ultralytics`
* `opencv-python`
* `numpy`
* `deep-sort-realtime`

---

## ⚡ System Configuration

* This project is **running on CPU (no GPU used)**
* It works without CUDA or GPU acceleration
* Performance may be slower compared to GPU-based systems
* For faster processing, a GPU can be used optionally

---

## 🎯 Features

* Real-time person detection
* Multi-object tracking with unique IDs
* High accuracy using YOLOv8x model
* Smooth tracking with DeepSORT
* Simple and clean implementation



## 📌 Note

* You can replace `yolov8x.pt` with smaller models like `yolov8n.pt` for faster inference on CPU
* Ensure your system meets basic performance requirements for smooth execution

---



✨ *This project demonstrates the power of combining deep learning with tracking algorithms for real-world applications.*\

---

## 👨‍💻 Author

**Asim Taseer Qureshi**

* 🔗 GitHub: [https://github.com/asimtaseer](https://github.com/asimtaseer)
* 🔗 LinkedIn: [https://www.linkedin.com/in/asimtaseer/](https://www.linkedin.com/in/asimtaseer/)

---

