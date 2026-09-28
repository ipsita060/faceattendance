# 👁️ VisionAttend AI - Smart Face Recognition Attendance System

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-00e6b8?style=for-the-badge&logo=github)](https://ipsita060.github.io/faceattendance/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Face Recognition](https://img.shields.io/badge/Face%20Recognition-dlib%20%7C%20ResNet--34-orange?style=for-the-badge)](https://github.com/ageitgey/face_recognition)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org)

An intelligent, high-accuracy biometric facial recognition attendance system with real-time video processing, edge AI inference in the browser, CSV logging, personnel enrollment, and a dark-mode dashboard.

---

## 🌐 Live Web Application

Experience the live deployed web application directly in your browser:
👉 **[https://ipsita060.github.io/faceattendance/](https://ipsita060.github.io/faceattendance/)**

> **Note**: Camera permissions are required for live webcam facial detection. You can also test by uploading photos or using the quick simulation buttons.

---

## ✨ Features

- **⚡ Real-Time Face Recognition**: 128-dimensional deep facial descriptor extraction running in real-time.
- **🛡️ High Accuracy**: Powered by deep convolutional neural networks (ResNet-34 architecture & dlib/face-api).
- **📋 Intelligent Attendance Logging**:
  - Automatically records `Name`, `Date`, `Time`, and `Status` (`Present`).
  - Single-check-in per day logic prevents duplicate logs.
  - 10-second debounce prevents repeated entries.
- **📸 Live Personnel Enrollment**:
  - Enroll new faces directly via webcam snapshot or portrait photo upload.
  - Automatically computes 128-d descriptor embedding and saves locally.
- **📊 Real-Time Analytics & Dashboard**:
  - Registered vs Present headcount stats.
  - Search by personnel name and date filtering.
  - Export attendance logs to CSV (`attendance_report.csv`).
  - Audio feedback chime upon verified check-in.
- **💻 Dual Mode Execution**:
  1. **Browser / Web App**: Client-side inference powered by `@vladmandic/face-api`, deployed on GitHub Pages.
  2. **Desktop / Python App**: High-speed OpenCV + `face_recognition` script with live video overlay and command-line options.

---

## 👥 Preloaded Personnel

The system is pre-configured with the following registered team members:
| Personnel | Portrait | Biometric Status |
| :--- | :---: | :---: |
| **Ipsita** | `images/ipsita.jpeg` | ✅ Enrolled |
| **Kanishka** | `images/kanishka.jpeg` | ✅ Enrolled |
| **Senna** | `images/senna.jpeg` | ✅ Enrolled |
| **Tanvi** | `images/tanvi.jpeg` | ✅ Enrolled |

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com/ipsita060/faceattendance.git
cd faceattendance
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional: Install `cmake` and `dlib` via `brew install cmake` or `pip install cmake` if building on macOS/Linux).*

---

## 🖥️ Running the Python Desktop App

Run the real-time OpenCV desktop application:
```bash
python faceattendance.py
```

### Command-Line Arguments:
```bash
# Use custom camera index (e.g. external USB camera or virtual cam)
python faceattendance.py --camera 1

# Stricter matching threshold (default 0.5)
python faceattendance.py --tolerance 0.45

# Specify custom portrait directory or CSV file
python faceattendance.py --image_dir images --csv attendance.csv

# Allow multiple attendance check-ins per day
python faceattendance.py --allow_multiple
```

- Press **`q`**, **`ESC`**, or **`ENTER`** to exit the camera window.

---

## 🌐 Running the Flask Web Server Locally

To run the full backend REST API and web application locally:
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

### API Endpoints:
- `GET /` — Serves the interactive attendance dashboard.
- `GET /api/status` — Returns registered faces and system health.
- `GET /api/attendance` — Retrieves all attendance records as JSON.
- `POST /api/attendance` — Manually logs attendance for a user.
- `POST /api/attendance/clear` — Clears attendance logs.
- `GET /api/attendance/export` — Downloads `attendance.csv` report.
- `POST /api/register` — Enrolls a new person via base64 portrait image.

---

## 📁 Repository Structure

```text
├── .gitignore               # Ignored files (DS_Store, pycache, venv)
├── .nojekyll                # GitHub Pages configuration
├── README.md                # Documentation and guide
├── app.js                   # Client-side face recognition and UI logic
├── app.py                   # Flask REST server for local backend
├── attendance.csv           # Attendance database (CSV)
├── faceattendance.py        # Python OpenCV face recognition script
├── favicon.svg              # Biometric vision icon
├── images/                  # Registered portrait images
│   ├── ipsita.jpeg
│   ├── kanishka.jpeg
│   ├── senna.jpeg
│   └── tanvi.jpeg
├── index.html               # Main dashboard web interface
├── registered_faces.js      # Precomputed 128-d face descriptors
├── registered_faces.json    # JSON representation of facial descriptors
├── requirements.txt         # Python package dependencies
└── style.css                # Dark-mode glassmorphism styling
```

---

## 📄 License
MIT License. Built for educational and commercial biometric attendance automation.
