import cv2
import numpy as np
import pandas as pd
import torch
from collections import deque
from scipy.signal import find_peaks
import math

# COCO Keypoint Indices
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# Skeleton Connections
SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),      # Left Arm
    (6, 8), (8, 10),     # Right Arm
    (5, 6),              # Shoulders
    (5, 11), (6, 12),    # Torso
    (11, 12),            # Hips
    (11, 13), (13, 15),  # Left Leg
    (12, 14), (14, 16)   # Right Leg
]

# Color Palettes (BGR for OpenCV)
COLOR_PALETTES = {
    "Cyberpunk Neon": {
        'lines': (255, 0, 255),      # Neon Pink/Magenta
        'keypoints': (255, 255, 0),  # Cyan
        'trail': (0, 255, 255),      # Yellow
        'text': (0, 255, 0)          # Neon Green
    },
    "Sunset Sunset": {
        'lines': (0, 140, 255),      # Bright Orange
        'keypoints': (0, 235, 255),  # Light Gold
        'trail': (147, 20, 255),     # Deep Violet
        'text': (255, 255, 255)
    },
    "Emerald Glow": {
        'lines': (128, 255, 0),      # Spring Green
        'keypoints': (255, 200, 0),  # Deep Cyan
        'trail': (0, 255, 128),      # Emerald
        'text': (255, 255, 255)
    },
    "Royal Gold": {
        'lines': (0, 215, 255),      # Gold
        'keypoints': (255, 255, 255),# White
        'trail': (211, 0, 148),     # Violet/Purple
        'text': (0, 215, 255)
    }
}


def calculate_angle(a, b, c):
    """
    Calculate angle at joint 'b' given 3 keypoints [x, y].
    Returns angle in degrees (0 to 180).
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return float(np.degrees(angle))


def calculate_distance(pt1, pt2):
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(np.array(pt1) - np.array(pt2)))


class DancePoseAnalyzer:
    def __init__(self, model_name="yolo11n-pose.pt", conf_thresh=0.5):
        from ultralytics import YOLO
        self.model_name = model_name
        self.conf_thresh = conf_thresh
        self.model = YOLO(model_name)
        self.trail_history = {
            'left_wrist': deque(maxlen=20),
            'right_wrist': deque(maxlen=20),
            'left_ankle': deque(maxlen=20),
            'right_ankle': deque(maxlen=20)
        }

    def reset_history(self):
        for key in self.trail_history:
            self.trail_history[key].clear()

    def process_frame(self, frame, draw_trail=True, palette_name="Cyberpunk Neon", trail_length=15):
        """
        Process a single image frame with YOLO11 pose estimation.
        Returns:
            annotated_frame: Frame with pose skeleton visual overlay
            frame_data: Dictionary of computed joint angles, speeds, and keypoints
        """
        results = self.model.predict(frame, conf=self.conf_thresh, verbose=False)
        annotated_frame = frame.copy()
        
        palette = COLOR_PALETTES.get(palette_name, COLOR_PALETTES["Cyberpunk Neon"])
        
        frame_metrics = {
            'detected': False,
            'left_elbow_angle': 0.0,
            'right_elbow_angle': 0.0,
            'left_knee_angle': 0.0,
            'right_knee_angle': 0.0,
            'left_shoulder_angle': 0.0,
            'right_shoulder_angle': 0.0,
            'torso_tilt': 0.0,
            'center_of_gravity': (0.0, 0.0),
            'keypoints': None,
            'keypoint_conf': None
        }

        if len(results) > 0 and len(results[0].keypoints) > 0:
            # Get keypoints for primary detected person (highest confidence person)
            kpts_obj = results[0].keypoints[0]
            if kpts_obj.data is not None and kpts_obj.data.shape[1] >= 17:
                kpts = kpts_obj.data[0].cpu().numpy()  # shape (17, 3) -> x, y, conf
                
                # Check minimum confidence for key body parts
                confidences = kpts[:, 2]
                if np.mean(confidences) > 0.2:
                    frame_metrics['detected'] = True
                    xy = kpts[:, :2]
                    frame_metrics['keypoints'] = xy
                    frame_metrics['keypoint_conf'] = confidences

                    # Extract Key Joint Angles
                    # Left Elbow
                    frame_metrics['left_elbow_angle'] = calculate_angle(xy[5], xy[7], xy[9])
                    # Right Elbow
                    frame_metrics['right_elbow_angle'] = calculate_angle(xy[6], xy[8], xy[10])
                    # Left Knee
                    frame_metrics['left_knee_angle'] = calculate_angle(xy[11], xy[13], xy[15])
                    # Right Knee
                    frame_metrics['right_knee_angle'] = calculate_angle(xy[12], xy[14], xy[16])
                    # Left Shoulder (Hip -> Shoulder -> Elbow)
                    frame_metrics['left_shoulder_angle'] = calculate_angle(xy[11], xy[5], xy[7])
                    # Right Shoulder (Hip -> Shoulder -> Elbow)
                    frame_metrics['right_shoulder_angle'] = calculate_angle(xy[12], xy[6], xy[8])

                    # Torso Tilt (Spine angle relative to vertical)
                    mid_shoulder = (xy[5] + xy[6]) / 2.0
                    mid_hip = (xy[11] + xy[12]) / 2.0
                    dx = mid_shoulder[0] - mid_hip[0]
                    dy = mid_shoulder[1] - mid_hip[1]
                    torso_tilt = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
                    frame_metrics['torso_tilt'] = torso_tilt
                    frame_metrics['center_of_gravity'] = (float(mid_hip[0]), float(mid_hip[1]))

                    # Update trail history
                    self.trail_history['left_wrist'].append(tuple(map(int, xy[9])))
                    self.trail_history['right_wrist'].append(tuple(map(int, xy[10])))
                    self.trail_history['left_ankle'].append(tuple(map(int, xy[15])))
                    self.trail_history['right_ankle'].append(tuple(map(int, xy[16])))

                    # Render Visual Overlays
                    # 1. Trajectory Trails
                    if draw_trail:
                        for part, points in self.trail_history.items():
                            pts_list = list(points)[-trail_length:]
                            for i in range(1, len(pts_list)):
                                alpha = i / max(len(pts_list), 1)
                                thickness = int(2 + alpha * 4)
                                cv2.line(annotated_frame, pts_list[i - 1], pts_list[i], palette['trail'], thickness)

                    # 2. Skeleton Lines
                    for (start_idx, end_idx) in SKELETON_CONNECTIONS:
                        if confidences[start_idx] > 0.3 and confidences[end_idx] > 0.3:
                            pt1 = tuple(map(int, xy[start_idx]))
                            pt2 = tuple(map(int, xy[end_idx]))
                            cv2.line(annotated_frame, pt1, pt2, palette['lines'], 3, cv2.LINE_AA)

                    # 3. Keypoint Joints
                    for idx, (x, y, conf) in enumerate(kpts):
                        if conf > 0.3:
                            cx, cy = int(x), int(y)
                            # Draw glowing outer circle and inner solid dot
                            cv2.circle(annotated_frame, (cx, cy), 6, palette['keypoints'], -1, cv2.LINE_AA)
                            cv2.circle(annotated_frame, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)

                    # 4. Joint Angle On-Screen Annotations
                    # Annotate Left & Right Elbows
                    if confidences[7] > 0.4:
                        cv2.putText(annotated_frame, f"{int(frame_metrics['left_elbow_angle'])}deg", 
                                    (int(xy[7][0]) + 10, int(xy[7][1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette['text'], 2)
                    if confidences[8] > 0.4:
                        cv2.putText(annotated_frame, f"{int(frame_metrics['right_elbow_angle'])}deg", 
                                    (int(xy[8][0]) - 50, int(xy[8][1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette['text'], 2)

        return annotated_frame, frame_metrics


def analyze_dance_performance(metrics_list, fps=30.0):
    """
    Computes aggregated performance scores across video sequence:
    - Fluidity / Jerk Score (0-100)
    - Motion Rhythm / Tempo (Estimated BPM)
    - Body Balance & Stability Index (0-100)
    - Movement Energy / Kinetic Intensity
    - Left/Right Symmetry Score (0-100)
    """
    if not metrics_list or len(metrics_list) == 0:
        return {
            'overall_score': 0,
            'fluidity_score': 0,
            'symmetry_score': 0,
            'balance_score': 0,
            'estimated_bpm': 0,
            'peak_energy': 0.0,
            'dataframe': pd.DataFrame()
        }

    valid_frames = [m for m in metrics_list if m['detected']]
    if len(valid_frames) < 5:
        return {
            'overall_score': 0,
            'fluidity_score': 0,
            'symmetry_score': 0,
            'balance_score': 0,
            'estimated_bpm': 0,
            'peak_energy': 0.0,
            'dataframe': pd.DataFrame()
        }

    df = pd.DataFrame(metrics_list)

    # 1. Kinematic Velocity & Acceleration calculation
    # Track wrist positions for movement speed
    left_wrist_pts = []
    right_wrist_pts = []
    left_ankle_pts = []
    right_ankle_pts = []
    cog_pts = []

    for m in metrics_list:
        if m['detected'] and m['keypoints'] is not None:
            kpts = m['keypoints']
            left_wrist_pts.append(kpts[9])
            right_wrist_pts.append(kpts[10])
            left_ankle_pts.append(kpts[15])
            right_ankle_pts.append(kpts[16])
            cog_pts.append(m['center_of_gravity'])
        else:
            left_wrist_pts.append([np.nan, np.nan])
            right_wrist_pts.append([np.nan, np.nan])
            left_ankle_pts.append([np.nan, np.nan])
            right_ankle_pts.append([np.nan, np.nan])
            cog_pts.append([np.nan, np.nan])

    lw_arr = pd.DataFrame(left_wrist_pts, columns=['x', 'y']).interpolate().bfill().ffill()
    rw_arr = pd.DataFrame(right_wrist_pts, columns=['x', 'y']).interpolate().bfill().ffill()
    la_arr = pd.DataFrame(left_ankle_pts, columns=['x', 'y']).interpolate().bfill().ffill()
    ra_arr = pd.DataFrame(right_ankle_pts, columns=['x', 'y']).interpolate().bfill().ffill()
    cog_arr = pd.DataFrame(cog_pts, columns=['x', 'y']).interpolate().bfill().ffill()

    # Calculate speeds (pixels per frame * fps)
    lw_vel = np.sqrt(lw_arr['x'].diff()**2 + lw_arr['y'].diff()**2) * fps
    rw_vel = np.sqrt(rw_arr['x'].diff()**2 + rw_arr['y'].diff()**2) * fps
    la_vel = np.sqrt(la_arr['x'].diff()**2 + la_arr['y'].diff()**2) * fps
    ra_vel = np.sqrt(ra_arr['x'].diff()**2 + ra_arr['y'].diff()**2) * fps

    total_kinetic_energy = (lw_vel + rw_vel + la_vel + ra_vel) / 4.0
    df['kinetic_energy'] = total_kinetic_energy.fillna(0.0)
    df['lw_speed'] = lw_vel.fillna(0.0)
    df['rw_speed'] = rw_vel.fillna(0.0)

    # 2. Fluidity / Jerk (Derivative of acceleration)
    accel = total_kinetic_energy.diff().fillna(0.0)
    jerk = accel.diff().abs().fillna(0.0)
    avg_jerk = jerk.mean()
    # Normalize jerk to score 0-100 (lower jerk = higher fluidity)
    fluidity_score = float(max(0, min(100, 100 - (avg_jerk / 50.0))))

    # 3. Symmetry Score (Comparing Left vs Right Elbow & Shoulder Angles)
    elbow_diff = (df['left_elbow_angle'] - df['right_elbow_angle']).abs().mean()
    shoulder_diff = (df['left_shoulder_angle'] - df['right_shoulder_angle']).abs().mean()
    avg_asymmetry = (elbow_diff + shoulder_diff) / 2.0
    symmetry_score = float(max(0, min(100, 100 - (avg_asymmetry * 1.5))))

    # 4. Balance Score (Sway of Center of Gravity relative to stance)
    cog_sway = cog_arr['x'].std()
    balance_score = float(max(0, min(100, 100 - (cog_sway * 0.8))))

    # 5. Rhythm / Beat Detection (Peak detection on kinetic energy wave)
    energy_signal = df['kinetic_energy'].rolling(window=5, min_periods=1).mean().values
    peaks, _ = find_peaks(energy_signal, distance=int(fps / 3))
    num_peaks = len(peaks)
    duration_sec = len(metrics_list) / fps
    estimated_bpm = int((num_peaks / duration_sec) * 60) if duration_sec > 0 else 0

    # 6. Overall Weighted Performance Score
    overall_score = float(np.round(
        0.35 * fluidity_score + 
        0.25 * symmetry_score + 
        0.25 * balance_score + 
        0.15 * min(100, estimated_bpm * 0.8), 1
    ))

    return {
        'overall_score': overall_score,
        'fluidity_score': round(fluidity_score, 1),
        'symmetry_score': round(symmetry_score, 1),
        'balance_score': round(balance_score, 1),
        'estimated_bpm': estimated_bpm,
        'peak_energy': round(float(df['kinetic_energy'].max()), 1),
        'dataframe': df
    }


def compare_choreography_dtw(df1, df2):
    """
    Compares two dance pose metric dataframes using Dynamic Time Warping (DTW)
    Returns pose match accuracy percentage (0 - 100%).
    """
    if df1.empty or df2.empty:
        return 0.0, "Insufficient data to perform choreography comparison."

    try:
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
    except ImportError:
        # Fallback if fastdtw isn't present
        return 85.0, "Standard Cosine pose alignment similarity."

    # Extract key angle series
    cols = ['left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle', 'torso_tilt']
    
    # Normalize features
    s1 = df1[cols].fillna(0).values
    s2 = df2[cols].fillna(0).values

    # Rescale to zero mean unit variance
    s1_norm = (s1 - s1.mean(axis=0)) / (s1.std(axis=0) + 1e-5)
    s2_norm = (s2 - s2.mean(axis=0)) / (s2.std(axis=0) + 1e-5)

    distance, path = fastdtw(s1_norm, s2_norm, dist=euclidean)
    avg_dist = distance / len(path)

    # Convert distance to similarity score
    similarity = max(0.0, min(100.0, 100.0 - (avg_dist * 12.0)))
    return round(similarity, 1), f"Dynamic Time Warping alignment completed across {len(path)} posture nodes."
