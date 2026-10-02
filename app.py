import torch
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as gg
import tempfile
import os
import time

from pose_analyzer import DancePoseAnalyzer, analyze_dance_performance, compare_choreography_dtw, COLOR_PALETTES
from demo_generator import generate_sample_dance_video
from report_generator import generate_html_report

# Page Config
st.set_page_config(
    page_title="YOLO11 Pose | Dance Movement Analysis",
    page_icon="💃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Dark Theme Glow & Custom Styling */
    .stApp {
        background-color: #0F111A;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
    }
    
    p, span, label, div[data-testid="stMarkdownContainer"] {
        color: #CBD5E1;
    }

    button[data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 500;
        font-size: 1.05rem;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #C084FC !important;
        font-weight: 700;
        border-bottom-color: #C084FC !important;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #311042 50%, #0F111A 100%);
        padding: 2.5rem;
        border-radius: 18px;
        border: 1px solid #4C1D95;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(138, 43, 226, 0.25);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A855F7, #EC4899, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #C084FC;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: #1A1D2C;
        border: 1px solid #3A2B5E;
        padding: 1.2rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #8B5CF6;
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .metric-lbl {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .stButton>button {
        background: linear-gradient(90deg, #7C3AED, #DB2777);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4);
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #6D28D9, #BE185D);
        box-shadow: 0 6px 18px rgba(124, 58, 237, 0.6);
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric/96/dancer.png", width=70)
st.sidebar.title("🎮 Analysis Settings")

model_choice = st.sidebar.selectbox(
    "YOLO11 Pose Model",
    ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt"],
    index=0,
    help="yolo11n-pose is ultra-fast; yolo11m-pose offers higher precision keypoints."
)

conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10, max_value=0.90, value=0.45, step=0.05
)

palette_choice = st.sidebar.selectbox(
    "Visual Skeleton Style",
    ["Cyberpunk Neon", "Sunset Sunset", "Emerald Glow", "Royal Gold"],
    index=0
)

enable_trail = st.sidebar.checkbox("Enable Motion Trajectory Trails", value=True)
trail_length = st.sidebar.slider("Trajectory Tail Length", 5, 30, 15) if enable_trail else 0
frame_skip = st.sidebar.selectbox("Processing Speed (Frame Skip)", [1, 2, 3], index=0, 
                                  help="Set to 2 or 3 to process video faster.")

# Cache YOLO pose model in Streamlit memory
@st.cache_resource
def load_pose_model(model_name):
    from ultralytics import YOLO
    return YOLO(model_name)

# App Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">💃 YOLO11 Pose: Dance Movement Analysis</div>
    <div class="hero-subtitle">
        AI-Powered Choreography Analytics, Joint Kinematics & Dynamic Time Warping (DTW) Pose Tracking
    </div>
</div>
""", unsafe_allow_html=True)

# Application Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Single Video Kinematics", 
    "🪞 Side-by-Side Comparison", 
    "📸 Live Snapshot Telemetry", 
    "📊 Export & Performance Report"
])

# Utility to process video and analyze metrics
def process_video_stream(video_path, analyzer, palette_name, trail_length, skip):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_path = tfile.name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps / skip, (width, height))

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    preview_placeholder = st.empty()

    analyzer.reset_history()
    metrics_list = []
    frame_idx = 0
    processed_count = 0

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % skip != 0:
            continue

        annotated_frame, f_metrics = analyzer.process_frame(
            frame, 
            draw_trail=enable_trail, 
            palette_name=palette_name, 
            trail_length=trail_length
        )
        out.write(annotated_frame)
        metrics_list.append(f_metrics)

        processed_count += 1
        pct = min(1.0, frame_idx / max(total_frames, 1))
        progress_bar.progress(pct)

        # Update preview frame periodically
        if processed_count % 10 == 0:
            rgb_preview = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(rgb_preview, caption=f"Processing Frame {frame_idx}/{total_frames}", use_container_width=True)
            status_text.text(f"Processing frame {frame_idx}/{total_frames} ({int(pct * 100)}%)")

    cap.release()
    out.release()

    # Convert annotated video to web-compatible H.264 format
    final_video_path = out_path
    try:
        import imageio_ffmpeg, subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        h264_path = out_path.replace('.mp4', '_h264.mp4')
        cmd = [
            ffmpeg_exe, '-y', '-i', out_path,
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            h264_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(h264_path) and os.path.getsize(h264_path) > 1000:
            final_video_path = h264_path
    except Exception:
        pass

    elapsed = time.time() - start_time
    status_text.success(f"Processing complete in {elapsed:.2f} seconds ({processed_count} frames analyzed)!")
    progress_bar.empty()
    preview_placeholder.empty()

    return final_video_path, metrics_list, fps / skip


# TAB 1: Single Video Analysis
with tab1:
    st.subheader("📹 Video Movement Kinematics & Rhythm Tracking")
    
    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        uploaded_video = st.file_uploader("Upload Dance Video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
    with col_u2:
        st.write("Or test immediately:")
        use_sample = st.button("🎬 Use Sample Dance Video")

    if 'current_video' not in st.session_state:
        st.session_state['current_video'] = None

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        st.session_state['current_video'] = tfile.name
    elif use_sample:
        st.session_state['current_video'] = generate_sample_dance_video()

    video_to_process = st.session_state.get('current_video')

    if video_to_process:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("### 📥 Original Input Video")
            st.video(video_to_process)

        if st.button("🚀 Analyze Dance Kinematics", key="run_single_analysis"):
            with st.spinner("Initializing YOLO11 Pose Engine..."):
                analyzer = DancePoseAnalyzer(model_name=model_choice, conf_thresh=conf_thresh)
                
            out_video_path, metrics_list, effective_fps = process_video_stream(
                video_to_process, analyzer, palette_choice, trail_length, frame_skip
            )

            # Store in session state for export tab
            perf_results = analyze_dance_performance(metrics_list, fps=effective_fps)
            st.session_state['perf_results'] = perf_results
            st.session_state['out_video_path'] = out_video_path
            st.session_state['video_name'] = os.path.basename(video_to_process)

            with col_v2:
                st.markdown("### 💃 YOLO11 Pose Annotated Video")
                st.video(out_video_path)

            # Metric Scorecards
            st.markdown("### 🏆 Choreography Performance Dashboard")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.markdown(f'<div class="metric-card"><div class="metric-val">{perf_results["overall_score"]}</div><div class="metric-lbl">Overall Score</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-val">{perf_results["fluidity_score"]}%</div><div class="metric-lbl">Fluidity (Jerk)</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-val">{perf_results["symmetry_score"]}%</div><div class="metric-lbl">Body Symmetry</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-card"><div class="metric-val">{perf_results["balance_score"]}%</div><div class="metric-lbl">Balance Index</div></div>', unsafe_allow_html=True)
            m5.markdown(f'<div class="metric-card"><div class="metric-val">{perf_results["estimated_bpm"]}</div><div class="metric-lbl">Cadence (BPM)</div></div>', unsafe_allow_html=True)

            df_metrics = perf_results['dataframe']

            if not df_metrics.empty:
                st.markdown("---")
                st.markdown("### 📈 Interactive Motion Dynamics & Joint Telemetry")

                c_g1, c_g2 = st.columns(2)

                # Chart 1: Joint Angles over Time
                with c_g1:
                    fig_angles = px.line(
                        df_metrics,
                        y=['left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 'right_knee_angle', 'torso_tilt'],
                        labels={'index': 'Frame Number', 'value': 'Angle (Degrees)'},
                        title="📐 Joint Angle Dynamics Across Frames",
                        template="plotly_dark",
                        color_discrete_sequence=['#38BDF8', '#F43F5E', '#10B981', '#F59E0B', '#A855F7']
                    )
                    fig_angles.update_layout(legend_title_text='Joint Node')
                    st.plotly_chart(fig_angles, use_container_width=True)

                # Chart 2: Movement Speed & Kinetic Energy Profile
                with c_g2:
                    fig_speed = px.area(
                        df_metrics,
                        y=['lw_speed', 'rw_speed', 'kinetic_energy'],
                        labels={'index': 'Frame Number', 'value': 'Velocity (px/sec)'},
                        title="⚡ Hand Movement Speeds & Kinetic Intensity",
                        template="plotly_dark",
                        color_discrete_sequence=['#EC4899', '#8B5CF6', '#06B6D4']
                    )
                    st.plotly_chart(fig_speed, use_container_width=True)


# TAB 2: Side-by-Side Comparison
with tab2:
    st.subheader("🪞 Dance Pose & Choreography DTW Comparison")
    st.info("Upload two videos (e.g. Student vs Instructor) to compute pose similarity and alignment match using Dynamic Time Warping (DTW).")

    col_cmp1, col_cmp2 = st.columns(2)
    with col_cmp1:
        vid_a = st.file_uploader("Upload Video A (Dancer 1 / Student)", type=["mp4", "mov"], key="vid_a")
    with col_cmp2:
        vid_b = st.file_uploader("Upload Video B (Dancer 2 / Reference Instructor)", type=["mp4", "mov"], key="vid_b")

    if vid_a and vid_b:
        if st.button("⚔️ Perform Dynamic Choreography Comparison"):
            with st.spinner("Extracting YOLO11 keypoints from both videos..."):
                analyzer = DancePoseAnalyzer(model_name=model_choice, conf_thresh=conf_thresh)

                # Process A
                t_a = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                t_a.write(vid_a.read())
                _, metrics_a, fps_a = process_video_stream(t_a.name, analyzer, palette_choice, 0, frame_skip)
                res_a = analyze_dance_performance(metrics_a, fps=fps_a)

                # Process B
                t_b = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                t_b.write(vid_b.read())
                _, metrics_b, fps_b = process_video_stream(t_b.name, analyzer, palette_choice, 0, frame_skip)
                res_b = analyze_dance_performance(metrics_b, fps=fps_b)

                sim_score, msg = compare_choreography_dtw(res_a['dataframe'], res_b['dataframe'])

            st.markdown(f"""
            <div class="score-box" style="background: linear-gradient(135deg, #1A1D2C, #0F2042); margin-top:20px;">
                <div style="font-size: 20px; color: #38BDF8;">CHOREOGRAPHY MATCH ACCURACY (DTW)</div>
                <div style="font-size: 72px; font-weight: bold; color: #10B981;">{sim_score}%</div>
                <div style="color: #94A3B8;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)

            # Radar Chart Breakdown
            categories = ['Fluidity', 'Symmetry', 'Balance', 'Tempo Matching']
            fig_radar = gg.Figure()

            fig_radar.add_trace(gg.Scatterpolar(
                r=[res_a['fluidity_score'], res_a['symmetry_score'], res_a['balance_score'], min(100, res_a['estimated_bpm'])],
                theta=categories,
                fill='toself',
                name='Dancer A (Student)',
                line_color='#EC4899'
            ))

            fig_radar.add_trace(gg.Scatterpolar(
                r=[res_b['fluidity_score'], res_b['symmetry_score'], res_b['balance_score'], min(100, res_b['estimated_bpm'])],
                theta=categories,
                fill='toself',
                name='Dancer B (Instructor)',
                line_color='#38BDF8'
            ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                template="plotly_dark",
                title="🎯 Performance Profile Radar Comparison"
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# TAB 3: Live Snapshot Telemetry
with tab3:
    st.subheader("📸 Real-Time Dance Snapshot Pose Telemetry")
    st.write("Capture a frame from your webcam or camera to instantly evaluate pose alignment and 17 COCO keypoints.")

    img_file_buffer = st.camera_input("Take a Pose Snapshot")

    if img_file_buffer is not None:
        # Read image
        bytes_data = img_file_buffer.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        analyzer = DancePoseAnalyzer(model_name=model_choice, conf_thresh=conf_thresh)
        annotated_img, metrics = analyzer.process_frame(cv_img, draw_trail=False, palette_name=palette_choice)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("### 📷 YOLO11 Pose Overlay")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_s2:
            st.markdown("### 🔍 Live Joint Telemetry")
            if metrics['detected']:
                st.success("✅ Human Pose Keypoints Detected!")
                st.metric("Left Elbow Angle", f"{int(metrics['left_elbow_angle'])}°")
                st.metric("Right Elbow Angle", f"{int(metrics['right_elbow_angle'])}°")
                st.metric("Left Knee Angle", f"{int(metrics['left_knee_angle'])}°")
                st.metric("Right Knee Angle", f"{int(metrics['right_knee_angle'])}°")
                st.metric("Torso Alignment Tilt", f"{int(metrics['torso_tilt'])}°")
            else:
                st.warning("⚠️ No clear pose detected. Ensure full body is visible in camera frame.")


# TAB 4: Export Reports & Data
with tab4:
    st.subheader("📊 Export & Download Performance Analytics")

    if 'perf_results' in st.session_state and st.session_state['perf_results']:
        p_res = st.session_state['perf_results']
        vid_path = st.session_state.get('out_video_path', '')
        v_name = st.session_state.get('video_name', 'Dance_Analysis.mp4')

        st.markdown("### 📥 Download Processed Artifacts")

        col_d1, col_d2, col_d3 = st.columns(3)

        # 1. Download Processed Video
        if vid_path and os.path.exists(vid_path):
            with open(vid_path, 'rb') as vf:
                col_d1.download_button(
                    label="🎥 Download Annotated Video",
                    data=vf,
                    file_name=f"yolo11_pose_{v_name}",
                    mime="video/mp4"
                )

        # 2. Download CSV Telemetry
        df_export = p_res['dataframe']
        if not df_export.empty:
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            col_d2.download_button(
                label="📊 Download Metrics CSV Log",
                data=csv_data,
                file_name="dance_pose_telemetry.csv",
                mime="text/csv"
            )

        # 3. Download HTML Report
        html_rep = generate_html_report(p_res, video_filename=v_name)
        col_d3.download_button(
            label="📄 Download HTML Summary Report",
            data=html_rep,
            file_name="Dance_Movement_Analysis_Report.html",
            mime="text/html"
        )

        st.markdown("---")
        st.markdown("### 📑 Telemetry Data Table Preview")
        st.dataframe(df_export.head(20), use_container_width=True)

    else:
        st.info("👈 Run a video analysis in Tab 1 first to generate exportable reports and logs.")
