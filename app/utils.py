import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def generate_charts(session_data, session_folder):
    """
    Generate emotion charts (pie chart and trend chart) for a session.

    Args:
        session_data: DataFrame containing the session's emotion and timestamp data.
        session_folder: Path to the session's folder for saving charts.

    Returns:
        Tuple of paths to the generated charts (pie_chart_path, trend_chart_path).
    """
    pie_chart_path = os.path.join(session_folder, 'pie_chart.png')
    trend_chart_path = os.path.join(session_folder, 'trend_chart.png')

    # Define custom colors for each emotion
    emotion_colors = {
        'Angry': '#ff3d3d',
        'Disgust': '#7bff3d',
        'Fear': '#9e3dff',
        'Happy': '#f4a300',
        'Neutral': '#c4c4c4',
        'Sad': '#3d79ff',
        'Surprise': '#ff9f00'
    }

    # --- Generate Pie Chart ---
    emotion_counts = session_data['Emotion'].value_counts()
    if not emotion_counts.empty:
        colors = [emotion_colors.get(emotion, '#cccccc') for emotion in emotion_counts.index]
        plt.figure(figsize=(5, 5))
        plt.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors)
        plt.title('Emotion Distribution')
        plt.tight_layout()
        plt.savefig(pie_chart_path)
        plt.close()
    else:
        print("Warning: No data available for pie chart.")

    # --- Generate Emotion Trend Chart ---
    if 'Timestamp' not in session_data.columns:
        print("Error: 'Timestamp' column missing from session data.")
        return pie_chart_path, None

    # Ensure timestamps are valid
    session_data['Timestamp'] = pd.to_datetime(session_data['Timestamp'], errors='coerce')
    session_data = session_data.dropna(subset=['Timestamp'])

    if not session_data.empty:
        session_data.set_index('Timestamp', inplace=True)

        # Resample and process data
        emotion_trend = (session_data['Emotion']
                         .groupby(pd.Grouper(freq='5S'))
                         .value_counts()
                         .unstack(fill_value=0))

        if not emotion_trend.empty:
            plt.figure(figsize=(5, 5))
            emotion_trend.plot(kind='line', colormap='tab10', linewidth=2)
            plt.title("Emotion Trends Over Time")
            plt.xlabel("Time")
            plt.ylabel("Emotion Count")
            plt.legend(title="Emotions", loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(trend_chart_path)
            plt.close()
        else:
            print("Warning: No data available for trend chart.")
            trend_chart_path = None
    else:
        print("Error: Session data is empty after processing timestamps.")
        trend_chart_path = None

    return pie_chart_path, trend_chart_path