import cv2
import numpy as np
import os
import math

def generate_sample_dance_video(output_path="sample_dance.mp4", duration_sec=5, fps=30, width=640, height=480):
    """
    Generates a realistic synthetic dance video featuring an animated silhouette figure
    performing dynamic rhythmic dance movements (arm raises, knee bends, spin/sway).
    """
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        return output_path

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    center_x = width // 2
    base_y = int(height * 0.75)

    for i in range(total_frames):
        t = i / float(fps)  # Time in seconds
        
        # Dark studio background with subtle ambient stage lights
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (15, 15, 25) # Deep navy black

        # Draw stage light spots
        cv2.circle(frame, (int(center_x + math.sin(t * 2) * 100), int(height * 0.2)), 140, (30, 20, 50), -1)
        cv2.circle(frame, (int(center_x - math.sin(t * 2) * 100), int(height * 0.3)), 120, (20, 40, 60), -1)

        # Dance motion equations (sinusoidal choreography)
        sway_x = math.sin(t * 3.5) * 40.0
        bounce_y = abs(math.sin(t * 7.0)) * 25.0
        
        # Key Joint Positions
        hip = (int(center_x + sway_x), int(base_y - 120 - bounce_y))
        neck = (int(center_x + sway_x), int(hip[1] - 130))
        head = (int(neck[0]), int(neck[1] - 40))

        # Dynamic Arm Motions (Rhythmic swinging)
        left_arm_angle = math.sin(t * 5.0) * 1.2
        right_arm_angle = math.cos(t * 5.0) * 1.2
        
        l_elbow = (int(neck[0] - 60 * math.cos(left_arm_angle)), int(neck[1] + 50 * math.sin(left_arm_angle)))
        l_wrist = (int(l_elbow[0] - 50 * math.cos(left_arm_angle * 1.5)), int(l_elbow[1] - 60 * math.sin(left_arm_angle * 1.5)))

        r_elbow = (int(neck[0] + 60 * math.cos(right_arm_angle)), int(neck[1] + 50 * math.sin(right_arm_angle)))
        r_wrist = (int(r_elbow[0] + 50 * math.cos(right_arm_angle * 1.5)), int(r_elbow[1] - 60 * math.sin(right_arm_angle * 1.5)))

        # Dynamic Leg Motions (Knee bends)
        l_knee = (int(hip[0] - 35), int(hip[1] + 70 + bounce_y * 0.4))
        l_ankle = (int(hip[0] - 45), int(base_y))

        r_knee = (int(hip[0] + 35), int(hip[1] + 70 + bounce_y * 0.4))
        r_ankle = (int(hip[0] + 45), int(base_y))

        # Color palette for synthetic dancer figure (High contrast for YOLO pose detection)
        body_color = (240, 240, 240)  # White/Light Grey dancer
        limb_thickness = 18

        # Render Head
        cv2.circle(frame, head, 24, body_color, -1)
        # Torso
        cv2.line(frame, neck, hip, body_color, 24)
        # Shoulders bar
        cv2.line(frame, (neck[0] - 30, neck[1]), (neck[0] + 30, neck[1]), body_color, limb_thickness)

        # Arms
        cv2.line(frame, (neck[0] - 30, neck[1]), l_elbow, body_color, limb_thickness)
        cv2.line(frame, l_elbow, l_wrist, body_color, limb_thickness)

        cv2.line(frame, (neck[0] + 30, neck[1]), r_elbow, body_color, limb_thickness)
        cv2.line(frame, r_elbow, r_wrist, body_color, limb_thickness)

        # Legs
        cv2.line(frame, (hip[0] - 20, hip[1]), l_knee, body_color, limb_thickness)
        cv2.line(frame, l_knee, l_ankle, body_color, limb_thickness)

        cv2.line(frame, (hip[0] + 20, hip[1]), r_knee, body_color, limb_thickness)
        cv2.line(frame, r_knee, r_ankle, body_color, limb_thickness)

        # Feet
        cv2.circle(frame, l_ankle, 12, body_color, -1)
        cv2.circle(frame, r_ankle, 12, body_color, -1)

        out.write(frame)

    out.release()

    # Convert to browser-compatible H.264 video
    try:
        import imageio_ffmpeg, subprocess, shutil
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_h264 = output_path + ".h264.mp4"
        cmd = [ffmpeg_exe, '-y', '-i', output_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', temp_h264]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_h264) and os.path.getsize(temp_h264) > 1000:
            shutil.move(temp_h264, output_path)
    except Exception:
        pass

    return output_path

if __name__ == "__main__":
    generate_sample_dance_video()
    print("Sample dance video created successfully!")
