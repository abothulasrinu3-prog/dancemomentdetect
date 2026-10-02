# 💃 Dance Movement Analysis using YOLO11-Pose

An advanced AI-powered computer vision & biomechanics platform for **Dance Movement Analysis**, **Pose Trajectory Tracking**, and **Choreography Comparison** built with **Ultralytics YOLO11 Pose** and **Streamlit**.

---

## 🌟 Key Features

1. **YOLO11 Keypoint Pose Estimation**:
   - Detects 17 COCO body joints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) in real-time.
   - Computes dynamic joint angles (elbow flexion, knee extension, shoulder elevation, torso tilt).

2. **Kinematic Dance Metrics & Rhythm Engine**:
   - **Fluidity & Jerk Index**: Measures motion smoothness and sudden velocity changes.
   - **Body Symmetry & Alignment**: Evaluates bilateral balance between left and right limbs.
   - **Center of Gravity Stability**: Tracks postural sway and balance control.
   - **Cadence & Rhythm Detection (BPM)**: Frequency domain beat analysis on kinetic energy waves.

3. **Choreography Comparison via DTW**:
   - Compare student dance videos against a reference instructor video using **Dynamic Time Warping (DTW)**.
   - Radar chart profile breaking down movement accuracy by dimension.

4. **Stylized Skeleton Overlay & Motion Trails**:
   - Customizable color palettes: *Cyberpunk Neon*, *Sunset Orange*, *Emerald Glow*, *Royal Gold*.
   - Dynamic motion trajectory tails tracking wrist and ankle paths over time.

5. **Built-in Synthetic Demo Video Generator**:
   - Test the app out of the box with 1-click sample dance video generation!

6. **Export & Executive Reporting**:
   - Download YOLO11 annotated MP4 video.
   - Download CSV frame telemetry data log.
   - Export standalone styled HTML Dance Summary Report.

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
Make sure Python 3.9+ is installed on your system.

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/dance-movement-analysis-yolo11.git
cd dance-movement-analysis-yolo11

pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

Deploying this application to **Streamlit Community Cloud** is 100% free and takes less than 3 minutes:

### Step 1: Push Code to GitHub
1. Create a new repository on [GitHub](https://github.com/new) (e.g. `dance-yolo11-pose`).
2. Run the following in your local terminal:
```bash
git init
git add .
git commit -m "Initial commit for YOLO11 Dance Analysis"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dance-yolo11-pose.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Sign in to [share.streamlit.io](https://share.streamlit.io/).
2. Click **New app**.
3. Select your repository (`YOUR_USERNAME/dance-yolo11-pose`), branch (`main`), and Main file path (`app.py`).
4. Click **Deploy!**

> **Note**: The included `packages.txt` ensures system dependencies like `ffmpeg` and `libgl1` are automatically installed on Streamlit Cloud's Linux environment!

---

## 📂 Project Architecture

```
.
├── app.py                   # Main Streamlit web application & UI tabs
├── pose_analyzer.py          # YOLO11 pose estimation, math & kinematics engine
├── demo_generator.py         # Synthetic sample dance video generator
├── report_generator.py       # Styled HTML export report builder
├── requirements.txt          # Python dependencies
├── packages.txt              # Linux OS packages for Streamlit Cloud
├── .streamlit/
│   └── config.toml           # Theme & max upload configuration
└── README.md                 # Documentation
```

---

## 🛠️ Built With
- [Ultralytics YOLO11](https://docs.ultralytics.com/) - Pose Estimation
- [Streamlit](https://streamlit.io/) - Web UI & Telemetry Dashboard
- [OpenCV](https://opencv.org/) - Frame & Video Processing
- [Plotly](https://plotly.com/) - Interactive Graphs & Radar Charts
- [PyTorch](https://pytorch.org/) - Deep Learning Backend
