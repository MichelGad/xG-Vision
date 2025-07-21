# detection.py
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
from skimage.color import rgb2lab
from sklearn.cluster import MiniBatchKMeans
import time
import os
import traceback
from collections import deque
from typing import Optional, List, Dict

# --- Constants and Configuration ---
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

class SoccerPitchConfiguration:
    """
    Configuration class for the soccer pitch dimensions and keypoints.
    """
    def __init__(self):
        self.pitch_width_meters = 105
        self.pitch_height_meters = 68
        goal_width = 7.32
        
        self.goal_posts = {
            "left": {
                "top": np.array([0, self.pitch_height_meters / 2 - goal_width / 2]),
                "bottom": np.array([0, self.pitch_height_meters / 2 + goal_width / 2])
            },
            "right": {
                "top": np.array([self.pitch_width_meters, self.pitch_height_meters / 2 - goal_width / 2]),
                "bottom": np.array([self.pitch_width_meters, self.pitch_height_meters / 2 + goal_width / 2])
            }
        }

        self.vertices = np.array([
            [0, 0], [self.pitch_width_meters / 2, 0], [self.pitch_width_meters, 0],
            [self.pitch_width_meters, self.pitch_height_meters / 2], [self.pitch_width_meters, self.pitch_height_meters],
            [self.pitch_width_meters / 2, self.pitch_height_meters], [0, self.pitch_height_meters],
            [0, self.pitch_height_meters / 2], [self.pitch_width_meters / 2, self.pitch_height_meters / 2],
            [16.5, 13.85], [16.5, 54.15], [self.pitch_width_meters - 16.5, 54.15], [self.pitch_width_meters - 16.5, 13.85],
            [5.5, 24.85], [5.5, 43.15], [self.pitch_width_meters - 5.5, 43.15], [self.pitch_width_meters - 5.5, 24.85],
            [11, self.pitch_height_meters / 2], [self.pitch_width_meters - 11, self.pitch_height_meters / 2],
            [0, 30.35], [0, 37.65], [self.pitch_width_meters, 37.65], [self.pitch_width_meters, 30.35],
            [self.pitch_width_meters / 2 - goal_width / 2, 0], [self.pitch_width_meters / 2 + goal_width / 2, 0],
            [self.pitch_width_meters / 2 + goal_width / 2, self.pitch_height_meters], [self.pitch_width_meters / 2 - goal_width / 2, self.pitch_height_meters],
            [16.5, self.pitch_height_meters / 2], [self.pitch_width_meters - 16.5, self.pitch_height_meters / 2],
            [5.5, self.pitch_height_meters / 2], [self.pitch_width_meters - 5.5, self.pitch_height_meters / 2],
            [self.pitch_width_meters/2, self.pitch_height_meters/2 + 9.15]
        ], dtype=np.float32)

        self.labels = [
            'top_left_corner', 'top_center', 'top_right_corner',
            'right_center', 'bottom_right_corner', 'bottom_center',
            'bottom_left_corner', 'left_center', 'center_circle',
            'left_penalty_top', 'left_penalty_bottom', 'right_penalty_bottom', 'right_penalty_top',
            'left_six_yard_top', 'left_six_yard_bottom', 'right_six_yard_bottom', 'right_six_yard_top',
            'left_penalty_spot', 'right_penalty_spot',
            'left_goal_top', 'left_goal_bottom', 'right_goal_bottom', 'right_goal_top',
            'top_left_goal', 'top_right_goal', 'bottom_right_goal', 'bottom_left_goal',
            'left_penalty_arc', 'right_penalty_arc',
            'left_six_yard_center', 'right_six_yard_center', 'center_circle_bottom'
        ]

CONFIG = SoccerPitchConfiguration()
SUPERVISION_COLORS = sv.ColorPalette.from_hex(['#00BFFF', '#FF6347', '#FFD700', '#ADFF2F']) # Team 1, Team 2, Referee, Ball

def get_crops(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def resolve_goalkeepers_team_id(players: sv.Detections, players_team_id: np.ndarray, goalkeepers: sv.Detections) -> np.ndarray:
    if len(goalkeepers) == 0: return np.array([])
    if len(players) == 0: return np.zeros(len(goalkeepers), dtype=int)
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = np.mean(players_xy[players_team_id == 0], axis=0) if np.any(players_team_id == 0) else None
    team_1_centroid = np.mean(players_xy[players_team_id == 1], axis=0) if np.any(players_team_id == 1) else None
    ids = []
    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid) if team_0_centroid is not None else float('inf')
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid) if team_1_centroid is not None else float('inf')
        ids.append(0 if dist_0 < dist_1 else 1)
    return np.array(ids)

def get_dominant_color_lab(image: np.ndarray) -> np.ndarray:
    if image.size == 0: return np.array([0, 0, 0])
    pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
    if len(pixels) < 1: return np.array([0, 0, 0])
    clt = MiniBatchKMeans(n_clusters=1, n_init='auto', random_state=42)
    clt.fit(rgb2lab(pixels))
    return clt.cluster_centers_[0]

class TeamClassifier:
    def __init__(self):
        self.kmeans = MiniBatchKMeans(n_clusters=2, n_init='auto', random_state=42)
        self.team_colors_lab = None
        self.team_annotation_colors_map = {0: 0, 1: 1, 2: 2, 3: 3}

    def fit(self, crops: list[np.ndarray]):
        dominant_colors_lab = [get_dominant_color_lab(crop) for crop in crops if crop.size > 0]
        if not dominant_colors_lab:
            self.team_colors_lab = np.array([[80, -10, -10], [80, 10, 10]])
            return
        self.kmeans.fit(dominant_colors_lab)
        self.team_colors_lab = self.kmeans.cluster_centers_

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if self.team_colors_lab is None: raise RuntimeError("Classifier has not been fitted.")
        if not crops: return np.array([])
        dominant_colors_lab = [get_dominant_color_lab(crop) for crop in crops]
        return self.kmeans.predict(dominant_colors_lab)

class BallTracker:
    def __init__(self):
        self.track_history = []
        self.last_pos = None
        self.no_detection_frames = 0

    def update(self, detections: sv.Detections, no_ball_thresh: int, dist_thresh: int, max_len: int) -> sv.Detections:
        balls = detections[detections.class_id == BALL_CLASS_ID]
        if len(balls) > 0:
            best_ball_idx = np.argmax(balls.confidence)
            best_ball = balls[best_ball_idx : best_ball_idx + 1]
            center = best_ball.get_anchors_coordinates(sv.Position.CENTER)[0]
            if self.last_pos is not None and np.linalg.norm(center - self.last_pos) >= dist_thresh: self.track_history = []
            self.track_history.append(center)
            self.last_pos, self.no_detection_frames = center, 0
            if len(self.track_history) > max_len: self.track_history.pop(0)
            return best_ball
        self.no_detection_frames += 1
        if self.no_detection_frames > no_ball_thresh: self.last_pos, self.track_history = None, []
        return sv.Detections.empty()

class BallAnnotator:
    def __init__(self, radius: int = 6): self.radius = radius

    def annotate(self, scene: np.ndarray, ball: sv.Detections, history: list) -> np.ndarray:
        if len(ball) > 0:
            center = ball.get_anchors_coordinates(sv.Position.CENTER)[0]
            ball_color = SUPERVISION_COLORS.colors[BALL_CLASS_ID]
            cv2.circle(scene, tuple(map(int, center)), self.radius, ball_color.as_bgr(), -1)
        for i, pos in enumerate(history):
            if not np.isnan(pos).any():
                alpha = (i + 1) / len(history)
                overlay = scene.copy()
                ball_color = SUPERVISION_COLORS.colors[BALL_CLASS_ID]
                cv2.circle(overlay, tuple(map(int, pos)), max(1, int(self.radius * alpha)), ball_color.as_bgr(), -1)
                scene = cv2.addWeighted(overlay, alpha, scene, 1 - alpha, 0)
        return scene

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.homography, _ = cv2.findHomography(source, target) if len(source) >= 4 else (None, None)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if self.homography is None or len(points) == 0: return np.array([])
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), self.homography).reshape(-1, 2)

def draw_pitch() -> np.ndarray:
    field_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field_map.jpg')
    if not os.path.exists(field_map_path):
        return np.zeros((450, 700, 3), dtype=np.uint8)
    pitch_image = cv2.imread(field_map_path)
    return cv2.resize(pitch_image, (700, 450))

def draw_points_on_pitch(xy: np.ndarray, color: sv.Color, pitch: np.ndarray) -> np.ndarray:
    if len(xy) == 0: return pitch
    scale_x, scale_y = pitch.shape[1] / CONFIG.pitch_width_meters, pitch.shape[0] / CONFIG.pitch_height_meters
    for p in xy:
        if not np.isnan(p).any(): cv2.circle(pitch, (int(p[0] * scale_x), int(p[1] * scale_y)), 10, color.as_bgr(), -1)
    return pitch

def generate_tactical_map(keypoints: sv.KeyPoints, all_detections: sv.Detections, all_team_ids: np.ndarray, team_classifier: TeamClassifier, current_ball: sv.Detections, keypoint_conf_threshold: float) -> np.ndarray:
    pitch = draw_pitch()
    
    if len(keypoints.xy) == 0 or len(keypoints.xy[0]) == 0:
        text = "Awaiting keypoints for tactical map..."
        cv2.putText(pitch, text, (50, pitch.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return pitch

    if len(keypoints.confidence) > 0 and keypoints.confidence[0] is not None:
        mask = keypoints.confidence[0] > keypoint_conf_threshold
    else:
        mask = np.zeros(len(keypoints.xy[0]), dtype=bool)
    
    source_points = keypoints.xy[0][mask]
    target_points = CONFIG.vertices[mask]

    if len(source_points) < 4:
        text = f"Awaiting 4+ keypoints for tactical map (found {len(source_points)})"
        cv2.putText(pitch, text, (50, pitch.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return pitch

    transformer = ViewTransformer(source_points, target_points)
    
    if transformer.homography is None:
        cv2.putText(pitch, "Homography calculation failed", (50, pitch.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return pitch

    for team_id in np.unique(all_team_ids):
        mask = all_team_ids == team_id
        team_detections = all_detections[mask]
        color_idx = team_classifier.team_annotation_colors_map.get(int(team_id), 0)
        color = SUPERVISION_COLORS.colors[color_idx]
        
        team_xy = transformer.transform_points(team_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        pitch = draw_points_on_pitch(team_xy, color, pitch)

    if len(current_ball) > 0:
        ball_color = SUPERVISION_COLORS.colors[BALL_CLASS_ID]
        ball_xy = transformer.transform_points(current_ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        pitch = draw_points_on_pitch(ball_xy, ball_color, pitch)
        
    return pitch

def process_frame(frame, models, trackers, team_classifier, annotators, hyperparams):
    player_conf, keypoint_conf = hyperparams['detection'].values()
    
    player_results = models['players'](frame, conf=player_conf, verbose=False)[0]
    keypoint_results = models['keypoints'](frame, conf=keypoint_conf, verbose=False)[0]
    ball_results = models['ball'](frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(player_results)
    keypoints = sv.KeyPoints.from_ultralytics(keypoint_results)
    ball_detections = sv.Detections.from_ultralytics(ball_results)

    tracked_detections = trackers['player'].update_with_detections(detections)
    current_ball = trackers['ball'].update(ball_detections, **hyperparams['ball_track'])
    
    players = tracked_detections[tracked_detections.class_id == PLAYER_CLASS_ID]
    goalkeepers = tracked_detections[tracked_detections.class_id == GOALKEEPER_CLASS_ID]
    referees = tracked_detections[tracked_detections.class_id == REFEREE_CLASS_ID]

    annotated_frame = frame.copy()
    players_team_ids = np.array([])
    goalkeepers_team_ids = np.array([])
    
    if len(players) > 0 and hyperparams['plot']['show_players']:
        players_team_ids = team_classifier.predict(get_crops(frame, players))
        player_labels = {tid: f"P{tid}" for tid in players.tracker_id} if players.tracker_id is not None else {}
        for team_id in np.unique(players_team_ids):
            mask = players_team_ids == team_id
            team_players = players[mask]
            team_labels = [player_labels.get(tid, "") for tid in team_players.tracker_id] if team_players.tracker_id is not None else []
            color_idx = team_classifier.team_annotation_colors_map.get(team_id, 0)
            color = SUPERVISION_COLORS.colors[color_idx]
            sv.EllipseAnnotator(color=color).annotate(annotated_frame, team_players)
            sv.LabelAnnotator(color=color, text_color=sv.Color.WHITE).annotate(annotated_frame, team_players, labels=team_labels)

    if len(goalkeepers) > 0 and hyperparams['plot']['show_players']:
        goalkeepers_team_ids = resolve_goalkeepers_team_id(players, players_team_ids, goalkeepers)
        goalkeeper_labels = {tid: f"GK{tid}" for tid in goalkeepers.tracker_id} if goalkeepers.tracker_id is not None else {}
        for team_id in np.unique(goalkeepers_team_ids):
            mask = goalkeepers_team_ids == team_id
            team_goalkeepers = goalkeepers[mask]
            team_labels = [goalkeeper_labels.get(tid, "") for tid in team_goalkeepers.tracker_id] if team_goalkeepers.tracker_id is not None else []
            color_idx = team_classifier.team_annotation_colors_map.get(team_id, 0)
            color = SUPERVISION_COLORS.colors[color_idx]
            sv.EllipseAnnotator(color=color).annotate(annotated_frame, team_goalkeepers)
            sv.LabelAnnotator(color=color, text_color=sv.Color.WHITE).annotate(annotated_frame, team_goalkeepers, labels=team_labels)

    if len(referees) > 0 and hyperparams['plot']['show_players']:
        ref_color_idx = team_classifier.team_annotation_colors_map[REFEREE_CLASS_ID]
        ref_color = SUPERVISION_COLORS.colors[ref_color_idx]
        ref_labels = [f"Ref{tid}" for tid in referees.tracker_id] if referees.tracker_id is not None else []
        sv.EllipseAnnotator(color=ref_color).annotate(annotated_frame, referees)
        sv.LabelAnnotator(color=ref_color, text_color=sv.Color.WHITE).annotate(annotated_frame, referees, labels=ref_labels)

    if len(current_ball) > 0 and hyperparams['plot']['show_ball_tracks']:
        annotators['ball'].annotate(annotated_frame, current_ball, trackers['ball'].track_history)

    if hyperparams['plot']['show_keypoints']:
        if len(keypoints.confidence) > 0 and keypoints.confidence[0] is not None:
            keypoint_conf_threshold = hyperparams['detection']['keypoint_conf']
            confident_mask = keypoints.confidence[0] > keypoint_conf_threshold
            for x, y in keypoints.xy[0][confident_mask]:
                cv2.circle(annotated_frame, (int(x), int(y)), 5, sv.Color.WHITE.as_bgr(), -1)

    pitch = np.zeros((450, 700, 3), dtype=np.uint8)
    if hyperparams['plot']['show_radar']:
        all_detections = sv.Detections.merge([players, goalkeepers, referees])
        all_team_ids = np.concatenate([
            players_team_ids, 
            goalkeepers_team_ids, 
            np.full(len(referees), REFEREE_CLASS_ID)
        ])
        pitch = generate_tactical_map(
            keypoints, 
            all_detections, 
            all_team_ids, 
            team_classifier, 
            current_ball, 
            hyperparams['detection']['keypoint_conf']
        )
    
    h, w, _ = annotated_frame.shape
    pitch_resized = cv2.resize(pitch, (int(h * (pitch.shape[1]/pitch.shape[0])), h))
    final_frame = cv2.hconcat([annotated_frame, pitch_resized])

    return final_frame

def check_if_goal(ball_pitch_pos: np.ndarray, prev_ball_pitch_pos: np.ndarray, leeway_m: float) -> Optional[str]:
    """
    Checks if a ball has crossed a goal line based on its current and previous positions.
    """
    if ball_pitch_pos is None or prev_ball_pitch_pos is None:
        return None

    left_goal_posts = (CONFIG.goal_posts["left"]["top"][1], CONFIG.goal_posts["left"]["bottom"][1])
    if prev_ball_pitch_pos[0] > leeway_m and ball_pitch_pos[0] <= leeway_m:
        if left_goal_posts[0] < ball_pitch_pos[1] < left_goal_posts[1]:
            return "left"

    right_goal_posts = (CONFIG.goal_posts["right"]["top"][1], CONFIG.goal_posts["right"]["bottom"][1])
    if prev_ball_pitch_pos[0] < (CONFIG.pitch_width_meters - leeway_m) and ball_pitch_pos[0] >= (CONFIG.pitch_width_meters - leeway_m):
        if right_goal_posts[0] < ball_pitch_pos[1] < right_goal_posts[1]:
            return "right"
            
    return None

def analyze_goal_events(video_path: str, models: Dict, params: Dict) -> List[Dict]:
    """
    Analyzes a video to find goal events and extracts data from the moment of the shot.
    """
    player_conf = params['player_conf']
    keypoint_conf = params['keypoint_conf']
    shot_frame_offset = params['shot_frame_offset']
    goal_leeway = params['goal_area_leeway_m']

    frame_generator = sv.get_video_frames_generator(source_path=video_path)
    
    data_buffer = deque(maxlen=shot_frame_offset + 5)
    
    team_classifier = TeamClassifier()
    crop_frames_generator = sv.get_video_frames_generator(source_path=video_path, stride=25)
    all_crops = []
    for frame in crop_frames_generator:
        result = models['players'](frame, conf=player_conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        all_crops.extend(get_crops(frame, player_detections))
    if not all_crops:
        print("Warning: No players detected for team classification.")
        return []
    team_classifier.fit(all_crops)

    goal_events = []
    prev_ball_pitch_pos = None

    for frame_idx, frame in enumerate(frame_generator):
        player_results = models['players'](frame, conf=player_conf, verbose=False)[0]
        keypoint_results = models['keypoints'](frame, conf=keypoint_conf, verbose=False)[0]
        ball_results = models['ball'](frame, verbose=False)[0]

        all_detections = sv.Detections.from_ultralytics(player_results)
        keypoints = sv.KeyPoints.from_ultralytics(keypoint_results)
        ball_detections = sv.Detections.from_ultralytics(ball_results)
        
        transformer = None
        if len(keypoints.xy) > 0 and len(keypoints.xy[0]) > 0 and keypoints.confidence is not None and len(keypoints.confidence[0]) > 0:
            mask = keypoints.confidence[0] > keypoint_conf
            source_points = keypoints.xy[0][mask]
            target_points = CONFIG.vertices[mask]
            if len(source_points) >= 4:
                transformer = ViewTransformer(source_points, target_points)

        ball_pitch_pos = None
        if transformer and transformer.homography is not None and len(ball_detections) > 0:
            ball_center_screen = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
            ball_pitch_pos = transformer.transform_points(ball_center_screen)[0]

        current_data = {
            "frame_index": frame_idx,
            "frame_bgr": frame,
            "detections": all_detections,
            "keypoints": keypoints,
            "transformer": transformer,
            "ball_pitch_pos": ball_pitch_pos
        }
        data_buffer.append(current_data)
        
        goal_side = check_if_goal(ball_pitch_pos, prev_ball_pitch_pos, goal_leeway)
        
        if goal_side and len(data_buffer) >= shot_frame_offset:
            shot_data = data_buffer[0]
            
            shot_frame_img_bgr = shot_data['frame_bgr']
            shot_detections = shot_data['detections']
            shot_keypoints = shot_data['keypoints']

            players = shot_detections[shot_detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = shot_detections[shot_detections.class_id == GOALKEEPER_CLASS_ID]

            if len(players) > 0:
                players_team_ids = team_classifier.predict(get_crops(shot_frame_img_bgr, players))
                goalkeepers_team_ids = resolve_goalkeepers_team_id(players, players_team_ids, goalkeepers)
            else:
                players_team_ids = np.array([])
                goalkeepers_team_ids = np.array([])

            player_positions = {}
            if shot_data['transformer'] and shot_data['transformer'].homography is not None:
                for team_id in range(2):
                    player_mask = players_team_ids == team_id
                    gk_mask = goalkeepers_team_ids == team_id
                    
                    team_players_screen = players[player_mask].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    team_gks_screen = goalkeepers[gk_mask].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    
                    team_players_pitch = shot_data['transformer'].transform_points(team_players_screen)
                    team_gks_pitch = shot_data['transformer'].transform_points(team_gks_screen)

                    player_positions[f'team_{team_id+1}'] = [p.tolist() for p in team_players_pitch]
                    player_positions[f'goalkeeper_{team_id+1}'] = [gk.tolist() for gk in team_gks_pitch]
            
            all_shot_detections = sv.Detections.merge([players, goalkeepers])
            all_team_ids = np.concatenate([players_team_ids, goalkeepers_team_ids])
            
            tactical_map_img = generate_tactical_map(
                keypoints=shot_keypoints,
                all_detections=all_shot_detections,
                all_team_ids=all_team_ids,
                team_classifier=team_classifier,
                current_ball=sv.Detections.empty(),
                keypoint_conf_threshold=keypoint_conf
            )
            
            goal_events.append({
                "shot_frame_index": shot_data['frame_index'],
                "goal_frame_index": frame_idx,
                "shot_frame_img": cv2.cvtColor(shot_frame_img_bgr, cv2.COLOR_BGR2RGB),
                "tactical_map_img": cv2.cvtColor(tactical_map_img, cv2.COLOR_BGR2RGB),
                "player_positions_at_shot": player_positions
            })
            print(f"Goal detected at frame {frame_idx}! Shot frame analysis at {shot_data['frame_index']}.")
            data_buffer.clear()

        prev_ball_pitch_pos = ball_pitch_pos
    
    return goal_events

def detect_and_annotate_video(video_path: str, output_file_name: str, save_output: bool, models: dict, detection_hyper_params: dict, ball_track_hyperparams: dict, plot_hyperparams: dict, stop_flag: list, progress_callback=None, st_frame_callback=None) -> bool:
    trackers = {'player': sv.ByteTrack(), 'ball': BallTracker()}
    annotators = {'ball': BallAnnotator()}
    hyperparams = {'detection': detection_hyper_params, 'ball_track': ball_track_hyperparams, 'plot': plot_hyperparams}

    if progress_callback: progress_callback(0, "Step 1/2: Analyzing team colors...")
    
    frame_generator_for_crops = sv.get_video_frames_generator(source_path=video_path, stride=25)
    
    all_crops = []
    for frame in frame_generator_for_crops:
        result = models['players'](frame, conf=detection_hyper_params['player_conf'], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        all_crops.extend(get_crops(frame, player_detections))

    team_classifier = TeamClassifier()
    team_classifier.fit(all_crops)
    
    if progress_callback: progress_callback(0, "Step 2/2: Annotating video...")

    try:
        frame_generator = sv.get_video_frames_generator(source_path=video_path)
        video_info = sv.VideoInfo.from_video_path(video_path)
        total_frames = video_info.total_frames if video_info.total_frames is not None and video_info.total_frames > 0 else 1

        first_frame = next(frame_generator)
        annotated_frame = process_frame(first_frame, models, trackers, team_classifier, annotators, hyperparams)
        if st_frame_callback: st_frame_callback(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        
        if save_output:
            os.makedirs("outputs", exist_ok=True)
            output_path = os.path.join("outputs", output_file_name if output_file_name and output_file_name.endswith('.mp4') else f"annotated_{int(time.time())}.mp4")
            
            with sv.VideoSink(output_path, sv.VideoInfo(width=annotated_frame.shape[1], height=annotated_frame.shape[0], fps=video_info.fps)) as sink:
                sink.write_frame(annotated_frame)
                
                for frame_idx, frame in enumerate(frame_generator, 1):
                    if stop_flag[0]: break
                    annotated_frame = process_frame(frame, models, trackers, team_classifier, annotators, hyperparams)
                    if st_frame_callback: st_frame_callback(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    sink.write_frame(annotated_frame)
                    if progress_callback:
                        progress_val = min(100, int(((frame_idx + 1) / total_frames) * 100))
                        progress_callback(progress_val, f"Processing... {progress_val}%")
        else:
            for frame_idx, frame in enumerate(frame_generator, 1):
                if stop_flag[0]: break
                annotated_frame = process_frame(frame, models, trackers, team_classifier, annotators, hyperparams)
                if st_frame_callback: st_frame_callback(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                if progress_callback:
                    progress_val = min(100, int(((frame_idx + 1) / total_frames) * 100))
                    progress_callback(progress_val, f"Processing... {progress_val}%")

    except Exception as e:
        if progress_callback: progress_callback(0, f"An error occurred: {e}")
        traceback.print_exc()
        return False
    finally:
        if progress_callback: progress_callback(100, "Processing complete.")
    return True