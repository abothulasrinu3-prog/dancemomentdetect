import pandas as pd
import datetime

def generate_html_report(metrics_dict, video_filename="Dance_Video.mp4"):
    """
    Generates a stylized standalone HTML report summarizing dance movement metrics.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_score = metrics_dict.get('overall_score', 0)
    fluidity = metrics_dict.get('fluidity_score', 0)
    symmetry = metrics_dict.get('symmetry_score', 0)
    balance = metrics_dict.get('balance_score', 0)
    bpm = metrics_dict.get('estimated_bpm', 0)
    peak_energy = metrics_dict.get('peak_energy', 0.0)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Dance Movement Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0F111A;
            color: #E6E6FA;
            margin: 0;
            padding: 40px;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #8A2BE2;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #C77DFF;
            font-size: 32px;
            margin-bottom: 5px;
        }}
        .header p {{
            color: #A0AAB0;
            font-size: 14px;
        }}
        .score-box {{
            background: linear-gradient(135deg, #1A1D2C, #2A1B4E);
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(138, 43, 226, 0.2);
            margin-bottom: 30px;
        }}
        .overall-score {{
            font-size: 64px;
            font-weight: bold;
            color: #00E5FF;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background-color: #1A1D2C;
            border: 1px solid #3A2B5E;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #E6E6FA;
            margin-top: 10px;
        }}
        .card .label {{
            font-size: 14px;
            color: #9A9CB0;
        }}
        .table-container {{
            background-color: #1A1D2C;
            border-radius: 12px;
            padding: 20px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            color: #E6E6FA;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #2A2D3E;
        }}
        th {{
            background-color: #25283A;
            color: #C77DFF;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #6A6D80;
            font-size: 12px;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>💃 YOLO11 Pose Dance Movement Analysis</h1>
        <p>Report Generated on: {now_str} | File: {video_filename}</p>
    </div>

    <div class="score-box">
        <div class="label" style="font-size:18px; color:#C77DFF;">OVERALL CHOREOGRAPHY SCORE</div>
        <div class="overall-score">{overall_score} / 100</div>
        <p style="color:#A0AAB0;">Evaluated via Ultralytics YOLO11 Keypoint Kinematics</p>
    </div>

    <div class="grid">
        <div class="card">
            <div class="label">Movement Fluidity</div>
            <div class="value">{fluidity}%</div>
        </div>
        <div class="card">
            <div class="label">Postural Symmetry</div>
            <div class="value">{symmetry}%</div>
        </div>
        <div class="card">
            <div class="label">Center Balance</div>
            <div class="value">{balance}%</div>
        </div>
        <div class="card">
            <div class="label">Estimated Tempo</div>
            <div class="value">{bpm} BPM</div>
        </div>
        <div class="card">
            <div class="label">Peak Kinetic Energy</div>
            <div class="value">{peak_energy}</div>
        </div>
    </div>

    <div class="table-container">
        <h3 style="color:#C77DFF; margin-top:0;">Metric Dimensions Breakdown</h3>
        <table>
            <thead>
                <tr>
                    <th>Dimension</th>
                    <th>Target Benchmark</th>
                    <th>Measured Score</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Fluidity & Jerk Suppression</td>
                    <td>> 80.0%</td>
                    <td>{fluidity}%</td>
                    <td>{'✅ Excellent' if fluidity >= 80 else '⚡ Needs Smoother Transitions'}</td>
                </tr>
                <tr>
                    <td>Left/Right Arm-Leg Symmetry</td>
                    <td>> 75.0%</td>
                    <td>{symmetry}%</td>
                    <td>{'✅ Balanced' if symmetry >= 75 else '⚠️ Asymmetrical Motion'}</td>
                </tr>
                <tr>
                    <td>Postural Stability & Center of Gravity</td>
                    <td>> 85.0%</td>
                    <td>{balance}%</td>
                    <td>{'✅ Stable Base' if balance >= 85 else '⚠️ High Sway Observed'}</td>
                </tr>
                <tr>
                    <td>Choreography Cadence & Rhythm</td>
                    <td>90 - 150 BPM</td>
                    <td>{bpm} BPM</td>
                    <td>{'✅ In Rhythm' if 60 <= bpm <= 180 else 'ℹ️ Variable Cadence'}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="footer">
        Powered by Ultralytics YOLO11 Pose Estimation & Streamlit Analytics Engine
    </div>

</body>
</html>
"""
    return html_content
