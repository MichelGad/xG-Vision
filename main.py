# main.py
import tempfile
import streamlit as st
import cv2
from ultralytics import YOLO
import os
from detection import detect_and_annotate_video, analyze_goal_events

def main():
    st.set_page_config(page_title="xG-Vision: Advanced Goal Analysis", layout="wide")
    st.title("xG-Vision: Advanced Goal Analysis ⚽")
    st.subheader("From Live Player Tracking to Automated Goal Event Extraction")

    # --- Sidebar Setup ---
    st.sidebar.title("Settings")
    st.sidebar.markdown("Select a video source to begin.")
    
    # --- File and Path Management ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEMO_VIDEOS = {
        "Demo Video 1": os.path.join(SCRIPT_DIR, 'demo', 'demo_vid_1.mp4'),
        "Demo Video 2": os.path.join(SCRIPT_DIR, 'demo', 'demo_vid_2.mp4')
    }
    # Corrected model paths without version numbers
    MODEL_PATHS = {
        "players": os.path.join(SCRIPT_DIR, 'models', 'football-player-detection.pt'),
        "keypoints": os.path.join(SCRIPT_DIR, 'models', 'football-pitch-detection.pt'),
        "ball": os.path.join(SCRIPT_DIR, 'models', 'football-ball-detection.pt')
    }
    
    demo_selected = st.sidebar.selectbox("Select Demo Video", list(DEMO_VIDEOS.keys()))
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Or Upload Your Own Video", type=['mp4', 'mov', 'avi'])
    
    video_path = ""
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.sidebar.video(video_path)
    else:
        video_path = DEMO_VIDEOS[demo_selected]
        if not os.path.exists(video_path):
            st.sidebar.error(f"Demo video not found at {video_path}. Please check the 'demo' folder.")
            st.stop()
        st.sidebar.video(video_path)

    # --- Model Loading ---
    @st.cache_resource
    def load_model(path: str):
        if not os.path.exists(path):
            st.error(f"Model file not found: {path}. Please place it in the 'models' folder.")
            return None
        return YOLO(path)

    models = {name: load_model(path) for name, path in MODEL_PATHS.items()}
    if any(model is None for model in models.values()):
        st.stop()

    # --- Main Page Tabs ---
    tab1, tab2 = st.tabs(["Live Tactical Analysis", "Goal Event Analysis"])

    with tab1:
        st.header("Real-Time Player and Ball Tracking")
        st.markdown("""
        This tab provides a real-time view of the player detection, team prediction, and tactical map generation. 
        Use the controls below to configure and run the live analysis.
        """)
        
        stframe_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_text = st.empty()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detection Hyperparameters")
            detection_hyper_params = {
                "player_conf": st.slider('Player Confidence', 0.0, 1.0, 0.6, key='live_player_conf'),
                "keypoint_conf": st.slider('Keypoint Confidence', 0.0, 1.0, 0.5, key='live_kp_conf'),
            }
            st.subheader("Ball Tracking Hyperparameters")
            ball_track_hyperparams = {
                "no_ball_thresh": st.number_input("Track Reset Threshold (frames)", 1, 1000, 30, key='live_ball_thresh'),
                "dist_thresh": st.number_input("Track Distance Threshold (pixels)", 1, 1280, 100, key='live_dist_thresh'),
                "max_len": st.number_input("Max Track Length (detections)", 1, 1000, 35, key='live_max_len')
            }
        with col2:
            st.subheader("Annotation Options")
            plot_hyperparams = {
                "show_keypoints": st.toggle("Show Keypoints", False, key='live_show_kp'),
                "show_ball_tracks": st.toggle("Show Ball Tracks", True, key='live_show_ball'),
                "show_players": st.toggle("Show Player Detections", True, key='live_show_players'),
                "show_radar": st.toggle("Show Tactical Map", True, key='live_show_radar')
            }
            st.subheader("Output")
            save_output = st.checkbox("Save annotated video", False, key='live_save')
            output_filename = st.text_input("Output filename", "live_annotated_video.mp4", key='live_fname') if save_output else ""

        st.markdown("---")
        
        start_button_col, stop_button_col = st.columns(2)
        start_detection = start_button_col.button("Start Live Analysis", use_container_width=True)
        stop_detection = stop_button_col.button("Stop Live Analysis", use_container_width=True)

        if 'stop_flag' not in st.session_state:
            st.session_state.stop_flag = [False]
        if 'detection_running' not in st.session_state:
            st.session_state.detection_running = False

        def progress_callback(value, text=""):
            progress_bar.progress(value)
            progress_text.text(text)

        def frame_callback(frame_rgb):
            stframe_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        if start_detection and not st.session_state.detection_running:
            st.session_state.detection_running = True
            st.session_state.stop_flag[0] = False
            st.toast("Live detection started!")
            
            detect_and_annotate_video(
                video_path, output_filename, save_output, models,
                detection_hyper_params, ball_track_hyperparams, plot_hyperparams,
                st.session_state.stop_flag, progress_callback, frame_callback
            )
            st.session_state.detection_running = False
            progress_callback(100, "Detection finished or stopped.")
            st.toast("Detection completed!")

        if stop_detection and st.session_state.detection_running:
            st.session_state.stop_flag[0] = True
            st.toast("Stopping detection...")

    with tab2:
        st.header("Automated Goal Event Detection & Analysis")
        st.markdown("""
        This tool scans the entire video to automatically detect goal events. For each goal, it backtracks to the moment the shot was taken to provide a tactical snapshot, including:
        - The video frame of the shot.
        - The tactical map showing the positions of all players.
        - The raw coordinates of every player on the pitch.
        
        This data is crucial for advanced xG (Expected Goals) modeling.
        """)
        
        col1, col2 = st.columns([1,2])
        with col1:
            st.subheader("Analysis Parameters")
            goal_params = {
                "player_conf": st.slider('Player Confidence', 0.0, 1.0, 0.5, key='goal_player_conf'),
                "keypoint_conf": st.slider('Keypoint Confidence', 0.0, 1.0, 0.5, key='goal_kp_conf'),
                "shot_frame_offset": st.slider('Shot Frame Offset (frames before goal)', 5, 60, 25, key='goal_offset', help="How many frames to look back from a goal to find the shot frame."),
                "goal_area_leeway_m": st.slider('Goal Area Leeway (meters)', 0.0, 2.0, 0.5, key='goal_leeway', help="Tolerance for detecting the ball crossing the goal line.")
            }

        with col2:
            st.info("ℹ️ **How It Works**\n\n1.  **Pitch Detection:** The system first identifies the pitch keypoints (corners, penalty spots, etc.) to understand the field geometry.\n2.  **Ball Tracking:** It tracks the ball's position throughout the video, converting its screen coordinates to real-world pitch coordinates.\n3.  **Goal Detection:** A 'goal' is registered when the ball's coordinates cross the goal line.\n4.  **Backtracking:** Upon detecting a goal, the system retrieves the frame and player locations from moments *before* the goal, providing the context at the time of the shot.")
        
        st.markdown("---")

        if st.button("🚀 Find Goal Events", use_container_width=True):
            st.session_state.goal_results = None
            
            with st.spinner("Analyzing video for goal events... This may take a while. Grab a coffee! ☕"):
                goal_events = analyze_goal_events(
                    video_path=video_path,
                    models=models,
                    params=goal_params
                )
                st.session_state.goal_results = goal_events

        if 'goal_results' in st.session_state and st.session_state.goal_results is not None:
            results = st.session_state.goal_results
            st.markdown("---")
            if not results:
                st.warning("No goal events were detected in the video with the current settings.")
            else:
                st.success(f"🎉 Analysis Complete! Found {len(results)} goal event(s).")
                for i, event in enumerate(results):
                    with st.expander(f"🥅 **Goal Event #{i+1}** (Shot at frame {event['shot_frame_index']})", expanded=i==0):
                        col_shot, col_map = st.columns(2)
                        with col_shot:
                            st.image(event['shot_frame_img'], caption="Frame: Moment of the Shot", use_column_width=True)
                        with col_map:
                            st.image(event['tactical_map_img'], caption="Tactical Map: Player Positions at Shot", use_column_width=True)
                        
                        st.write("##### Player Positions (x, y meters from top-left corner)")
                        st.json(event['player_positions_at_shot'])

if __name__ == '__main__':
    main()