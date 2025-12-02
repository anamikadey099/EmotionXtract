from flask import Blueprint, render_template, Response, request, jsonify
import os
from datetime import datetime
import pandas as pd
from .camera import Camera
from .emotion_detection import EmotionDetector
from .utils import generate_charts

# Blueprint for routes
main = Blueprint('main', __name__)

# Initialize camera and emotion detector instances
camera = Camera()
emotion_detector = EmotionDetector()

# Start camera initialization in the background
camera.start_async_initialization()

# Global session variables
session_data = pd.DataFrame(columns=['Timestamp', 'Emotion'])

# Routes
@main.route('/')
def index():
    """Home Page - Start Live Feed"""
    return render_template('index.html')


@main.route('/video_feed')
def video_feed():
    """Stream video feed with emotion detection."""
    return Response(camera.generate_frames(emotion_detector, session_data),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route('/end_session')
def end_session():
    """End the session, release the camera, save data, and generate charts."""
    camera.release()
    global session_data
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save session data as CSV
    session_folder = os.path.join('static', 'sessions', f'session_{session_id}')
    os.makedirs(session_folder, exist_ok=True)
    session_data_path = os.path.join(session_folder, f'session_{session_id}.csv')
    session_data.to_csv(session_data_path, index=False)

    # Generate charts (pie chart and trend chart)
    pie_chart_path, trend_chart_path = generate_charts(session_data, session_folder)

    # Clean up session data and release camera
    session_data = pd.DataFrame(columns=['Timestamp', 'Emotion'])

    return jsonify({
        'message': 'Session ended, camera turned off, and data saved successfully.',
        'pie_chart_path': pie_chart_path,
        'trend_chart_path': trend_chart_path
    })


@main.route('/dashboard')
def dashboard():
    """Display the dashboard for the most recent session."""
    sessions_dir = os.path.join('static', 'sessions')
    sessions = sorted(os.listdir(sessions_dir), reverse=True)

    if not sessions:
        return render_template('dashboard.html', dominant_emotion="No Data", dominant_percentage=0,
                               pie_chart_path=None, trend_chart_path=None)

    # Load the latest session
    latest_session = sessions[0]
    session_folder = os.path.join(sessions_dir, latest_session)
    session_csv_path = os.path.join(session_folder, f"{latest_session}.csv")

    # Calculate dominant emotion and chart paths
    df = pd.read_csv(session_csv_path)
    dominant_emotion = df['Emotion'].value_counts().idxmax()
    dominant_percentage = round((df['Emotion'].value_counts().max() / len(df)) * 100, 1)

    pie_chart_path = f"{session_folder}/pie_chart.png"
    trend_chart_path = f"{session_folder}/trend_chart.png"

    return render_template('dashboard.html',
                           dominant_emotion=dominant_emotion,
                           dominant_percentage=dominant_percentage,
                           pie_chart_path=pie_chart_path,
                           trend_chart_path=trend_chart_path)


@main.route('/session_history')
def session_history():
    """Show all past session histories."""
    sessions_dir = os.path.join('static', 'sessions')
    session_list = []

    if os.path.exists(sessions_dir):
        for session in sorted(os.listdir(sessions_dir), reverse=True):
            session_parts = session.split('_')
            if len(session_parts) > 1:
                session_id = session_parts[1]
                session_folder = os.path.join(sessions_dir, session)
                csv_path = os.path.join(session_folder, f"{session}.csv")

                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    total_emotions = len(df)
                    dominant_emotion = df['Emotion'].value_counts().idxmax()
                    try:
                        date = datetime.strptime(session_id, "%Y%m%d_%H%M%S").strftime("%dth %b %Y %I:%M:%S %p")
                    except ValueError:
                        date = "Unknown"

                    session_list.append({
                        'session_id': session_id,
                        'date': date,
                        'total_emotions': total_emotions,
                        'dominant_emotion': dominant_emotion
                    })

    return render_template('session_history.html', sessions=session_list)