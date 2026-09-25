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
            # Figure 01-06: left goal line, top to bottom
            [0, 0], [0, 13.85], [0, 24.85], [0, 43.15], [0, 54.15], [0, self.pitch_height_meters],
            # Figure 07-08: left six-yard inner corners (top, bottom)
            [5.5, 24.85], [5.5, 43.15],
            # Figure 09: left penalty spot
            [11, self.pitch_height_meters / 2],
            # Figure 10-13: left penalty-box right edge (top, upper-mid, lower-mid, bottom)
            [16.5, 13.85], [16.5, 24.85], [16.5, 43.15], [16.5, 54.15],
            # Model indices 13-16 = midfield vertical (Figure 15-18).
            # NOTE: NOT Figure 14/19 here — the model's output order places
            # the circle edges at indices 30/31 (see dataset flip_idx).
            [self.pitch_width_meters / 2, 0],
            [self.pitch_width_meters / 2, self.pitch_height_meters / 2 - 9.15],
            [self.pitch_width_meters / 2, self.pitch_height_meters / 2 + 9.15],
            [self.pitch_width_meters / 2, self.pitch_height_meters],
            # Model indices 17-20 = right penalty-box left edge (Fig 20-23)
            [self.pitch_width_meters - 16.5, 13.85], [self.pitch_width_meters - 16.5, 24.85],
            [self.pitch_width_meters - 16.5, 43.15], [self.pitch_width_meters - 16.5, 54.15],
            # Model index 21 = right penalty spot (Figure 24)
            [self.pitch_width_meters - 11, self.pitch_height_meters / 2],
            # Model indices 22-23 = right six-yard inner corners (Fig 25-26)
            [self.pitch_width_meters - 5.5, 24.85], [self.pitch_width_meters - 5.5, 43.15],
            # Model indices 24-29 = right goal line, top to bottom (Fig 27-32)
            [self.pitch_width_meters, 0], [self.pitch_width_meters, 13.85],
            [self.pitch_width_meters, 24.85], [self.pitch_width_meters, 43.15],
            [self.pitch_width_meters, 54.15], [self.pitch_width_meters, self.pitch_height_meters],
            # Model indices 30-31 = center circle left/right (Fig 14/19)
            [self.pitch_width_meters / 2 - 9.15, self.pitch_height_meters / 2],
            [self.pitch_width_meters / 2 + 9.15, self.pitch_height_meters / 2],
        ], dtype=np.float32)

        self.labels = [
            'top_left_corner', 'left_box_top', 'left_six_top', 'left_six_bottom', 'left_box_bottom', 'bottom_left_corner',
            'left_six_yard_top', 'left_six_yard_bottom',
            'left_penalty_spot',
            'left_penalty_top', 'left_penalty_upper_mid', 'left_penalty_lower_mid', 'left_penalty_bottom',
            'top_center', 'center_circle_top', 'center_circle_bottom', 'bottom_center',
            'right_penalty_top', 'right_penalty_upper_mid', 'right_penalty_lower_mid', 'right_penalty_bottom',
            'right_penalty_spot',
            'right_six_yard_top', 'right_six_yard_bottom',
            'top_right_corner', 'right_box_top', 'right_six_top', 'right_six_bottom', 'right_box_bottom', 'bottom_right_corner',
            'center_circle_left', 'center_circle_right',
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

JERSEY_LABELS = ['black', 'blue', 'dark-blue', 'red', 'red-sky-blue',
                 'red-white', 'sky-blue', 'white-blue', 'white-dark',
                 'white-red', 'yellow-black']

class JerseyClassifier:
    """MobileNetV3 jersey color-pattern classifier (11 classes, see JERSEY_LABELS).

    Predicts a discrete jersey pattern per crop, which separates teams far
    more reliably than raw-color KMeans (e.g. white vs white-red).
    Loads lazily; predict() falls back to None entries when unavailable.
    """
    def __init__(self, ckpt_path: Optional[str] = None):
        self.ckpt_path = ckpt_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'models', 'football-jersey-classification.pth')
        self.model = None
        self.transform = None
        self.available = False
        self._try_load()

    def _try_load(self):
        try:
            import torch
            from torchvision.models import mobilenet_v3_small
            from torchvision import transforms
            if not os.path.exists(self.ckpt_path):
                return
            ckpt = torch.load(self.ckpt_path, map_location='cpu')
            self.model = mobilenet_v3_small(num_classes=len(JERSEY_LABELS))
            state = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.available = True
        except Exception:
            self.available = False

    def predict_labels(self, crops: list[np.ndarray]) -> list[Optional[int]]:
        return [label for label, _ in self.predict_with_conf(crops)]

    def predict_with_conf(self, crops: list[np.ndarray]) -> list[tuple[Optional[int], float]]:
        """Per-crop (jersey_label, softmax_confidence); torso-cropped first."""
        if not self.available or not crops:
            return [(None, 0.0)] * len(crops)
        import torch
        out = []
        with torch.no_grad():
            for crop in crops:
                torso = torso_crop(crop)
                if torso is None:
                    out.append((None, 0.0))
                    continue
                rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
                t = self.transform(rgb).unsqueeze(0)
                probs = self.model(t).softmax(dim=1)[0]
                conf, label = float(probs.max()), int(probs.argmax())
                out.append((label, conf))
        return out

def torso_crop(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Upper-torso window: avoids grass, legs, and neighboring players."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if min(h, w) < 8:
        return None
    y0, y1 = int(h * 0.10), int(h * 0.60)
    x0, x1 = int(w * 0.15), int(w * 0.85)
    torso = crop[y0:y1, x0:x1]
    return torso if torso.size > 0 else None

class TeamClassifier:
    # Below this softmax confidence the jersey vote is ignored in favor of
    # the LAB nearest-anchor fallback (blurry/small/occluded crops).
    JERSEY_MIN_CONF = 0.5

    def __init__(self, jersey_classifier: Optional[JerseyClassifier] = None):
        self.kmeans = MiniBatchKMeans(n_clusters=2, n_init='auto', random_state=42)
        self.team_colors_lab = None
        self.team_annotation_colors_map = {0: 0, 1: 1, 2: 2, 3: 3}
        self.jersey = jersey_classifier or JerseyClassifier()
        # jersey_label -> team_id for the two dominant match jerseys
        self.jersey_to_team: Dict[int, int] = {}
        # tracker_id -> vote Counter; makes team identity stable over time
        # even when single-frame crops are ambiguous.
        self.tid_votes: Dict[int, object] = {}

    def fit(self, crops: list[np.ndarray]):
        dominant_colors_lab = [get_dominant_color_lab(crop) for crop in crops if crop.size > 0]
        if not dominant_colors_lab:
            self.team_colors_lab = np.array([[80, -10, -10], [80, 10, 10]])
            return
        self.kmeans.fit(dominant_colors_lab)
        self.team_colors_lab = self.kmeans.cluster_centers_
        self.tid_votes = {}

        # Anchor the two teams to the two most common *jersey patterns*.
        # Jersey labels are discrete and lighting-stable; LAB centroids stay
        # as the fallback for unseen patterns (subs, keeper kits).
        self.jersey_to_team = {}
        if self.jersey.available and crops:
            votes = self.jersey.predict_with_conf(crops)
            order: Dict[int, int] = {}
            for v, c in votes:
                if v is not None and c >= self.JERSEY_MIN_CONF:
                    order[v] = order.get(v, 0) + 1
            top = sorted(order, key=order.get, reverse=True)[:2]
            self.jersey_to_team = {label: tid for tid, label in enumerate(top)}

    def predict(self, crops: list[np.ndarray], tracker_ids: Optional[np.ndarray] = None) -> np.ndarray:
        if self.team_colors_lab is None: raise RuntimeError("Classifier has not been fitted.")
        if not crops: return np.array([])
        fresh = self._fresh_predict(crops)
        if tracker_ids is None:
            return fresh
        # Temporal majority vote per tracker id.
        from collections import Counter
        out = np.empty(len(fresh), dtype=int)
        for k, (team, tid) in enumerate(zip(fresh, tracker_ids)):
            tid = int(tid) if tid is not None else -k - 1
            counter = self.tid_votes.get(tid)
            if counter is None:
                counter = self.tid_votes[tid] = Counter()
            counter[int(team)] += 1
            out[k] = int(counter.most_common(1)[0][0])
        # Bound memory: drop ids unseen for a long time is handled by the
        # caller creating one classifier per video run; cap defensively.
        if len(self.tid_votes) > 5000:
            self.tid_votes.clear()
        return out

    def _fresh_predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if self.jersey_to_team:
            votes = self.jersey.predict_with_conf(crops)
            if any(v is not None and c >= self.JERSEY_MIN_CONF for v, c in votes):
                dominant = [get_dominant_color_lab(c) for c in crops]
                teams = []
                for (v, c), col in zip(votes, dominant):
                    if v is not None and c >= self.JERSEY_MIN_CONF and v in self.jersey_to_team:
                        teams.append(self.jersey_to_team[v])
                    else:
                        # Unseen pattern (e.g. keeper/second kit) or unsure:
                        # nearest anchor in LAB space.
                        d = np.linalg.norm(self.team_colors_lab - col, axis=1)
                        teams.append(int(np.argmin(d)))
                return np.array(teams, dtype=int)
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
    """
    Robust image->pitch homography.

    Uses RANSAC so 1-2 mis-detected keypoints don't corrupt the whole frame.
    Two-stage fit (RANSAC + least-squares refit on inliers) plus a geometric
    spread check so clustered keypoints (e.g. only one penalty box visible)
    don't produce unstable transforms.
    Exposes validity signals so callers can fall back to the previous frame
    instead of plotting wildly wrong positions.
    """
    RANSAC_THRESHOLDS_M = (1.0, 2.0, 4.0)
    MAX_MEAN_REPROJ_ERROR_PX = 30.0
    MIN_INLIERS = 4
    # B1: leave-one-out pruning — iteratively eject the inlier that most
    # disagrees with the rest (mislabeled keypoints), down to this floor.
    LOO_MAX_ERR_M = 8.0
    LOO_MIN_KEEP = 5
    # Target-space spread required for a stable fit (meters).
    MIN_TARGET_RANGE_M = 8.0
    MIN_TARGET_AREA_M2 = 200.0
    # Image-space NMS distance for duplicate keypoints (px).
    NMS_DIST_PX = 15.0

    def __init__(self, source: np.ndarray, target: np.ndarray, label_idx: Optional[np.ndarray] = None, prune_loocv: bool = True, rng_seed: int = 42):
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        self.homography = None
        self.inlier_mask = None
        self.n_inliers = 0
        self.reprojection_error_px = float('inf')
        self.is_valid = False
        self.reject_reason = "fewer than 4 points"
        self.aniso_ratio = float('inf')
        self.mirrored = False
        self.inlier_targets = np.zeros((0, 2), np.float32)
        # C1 line evidence (filled by _best_candidate when field_lines given).
        self.line_score = float('inf')
        self.line_support = 0.0
        self.line_n = 0
        # CONFIG label indices aligned with the *input* arrays, for trust stats.
        self.used_label_idx = np.asarray(label_idx).ravel() if label_idx is not None else None
        self.inlier_label_idx = np.array([], dtype=int)
        # Post-NMS inlier correspondences, for joint keypoint checks.
        self.fit_src = np.zeros((0, 2), np.float32)
        self.fit_tgt = np.zeros((0, 2), np.float32)

        if len(source) >= 4 and len(source) == len(target):
            cv2.setRNGSeed(rng_seed)  # deterministic RANSAC: same frame -> same map
            src, tgt, kept = self._nms(source, target)
            if len(src) < 4:
                self.reject_reason = f"only {len(src)} points after NMS"
                return
            if not self._has_spread(tgt):
                self.reject_reason = "keypoints too clustered"
                # Still fit below so callers can use it as low-confidence fallback.
            # A3: try tight-to-loose RANSAC thresholds; rank below prefers
            # larger tight consensuses. LOO pruning is skipped when line
            # evidence will judge instead (far cheaper and more decisive).
            best = None
            for thresh in self.RANSAC_THRESHOLDS_M:
                cand = self._fit_at_threshold(src, tgt, thresh, prune=prune_loocv)
                if cand is None:
                    continue
                key = (cand["n_inliers"], -cand["aniso"])
                if best is None or key > best[0]:
                    best = (key, cand)
            if best is None:
                self.reject_reason = "RANSAC found no solution"
                return
            fit = best[1]
            self.homography = fit["H"]
            self.inlier_mask = fit["inliers"]
            self.n_inliers = fit["n_inliers"]
            self.reprojection_error_px = fit["err"]
            self.aniso_ratio = fit["aniso"]
            self.inlier_targets = fit.get("inlier_tgt", np.zeros((0, 2), np.float32))
            self.fit_src = src[fit["inliers"]].astype(np.float32)
            self.fit_tgt = tgt[fit["inliers"]].astype(np.float32)
            if self.used_label_idx is not None and len(self.used_label_idx) == len(kept):
                kept_labels = self.used_label_idx[kept]
                self.used_label_idx = kept_labels
                self.inlier_label_idx = kept_labels[fit["inliers"]].astype(int)
            spread_ok = self._has_spread(tgt[fit["inliers"]] if self.n_inliers else tgt)
            if self.n_inliers < self.MIN_INLIERS:
                self.reject_reason = f"only {self.n_inliers} inliers"
            elif self.reprojection_error_px > self.MAX_MEAN_REPROJ_ERROR_PX:
                self.reject_reason = f"reproj err {self.reprojection_error_px:.1f}px"
            elif not spread_ok:
                self.reject_reason = "inliers too clustered"
            elif not self._is_plausible(fit["H"]):
                self.reject_reason = "degenerate H"
            else:
                self.is_valid = True
                self.reject_reason = ""

    @classmethod
    def _fit_at_threshold(cls, src: np.ndarray, tgt: np.ndarray, thresh: float, prune: bool = True):
        H, mask = cv2.findHomography(src, tgt, cv2.RANSAC, thresh)
        if H is None or mask is None:
            return None
        inliers = mask.ravel().astype(bool)
        # Stage 2: refit on inliers only for a tighter solution.
        if int(inliers.sum()) >= 4:
            H_refit, _ = cv2.findHomography(src[inliers], tgt[inliers], 0)
            if H_refit is not None:
                H = H_refit
        # B1: LOO pruning ejects high-leverage mislabels that RANSAC's
        # loose threshold absorbed (e.g. 15-25m-off points counted as
        # inliers at 8m). Refits from scratch on the survivors.
        if prune:
            inliers, H = cls._prune_loocv(src, tgt, inliers)
            if H is None:
                return None
        err = cls._mean_reproj_error(src, tgt, H, inliers)
        aniso = cls._aniso(H)
        return {"H": H, "inliers": inliers, "n_inliers": int(inliers.sum()), "err": err, "aniso": aniso,
                "inlier_tgt": tgt[inliers].copy()}

    @classmethod
    def _prune_loocv(cls, src: np.ndarray, tgt: np.ndarray, inliers: np.ndarray):
        """Drop the worst leave-one-out inlier until all agree (or floor)."""
        inliers = inliers.copy()
        H, _ = cv2.findHomography(src[inliers], tgt[inliers], 0)
        if H is None:
            return inliers, None
        while int(inliers.sum()) > cls.LOO_MIN_KEEP:
            idx = np.where(inliers)[0]
            worst, worst_err = -1, -1.0
            for k in idx:
                sub = inliers.copy()
                sub[k] = False
                Hk, _ = cv2.findHomography(src[sub], tgt[sub], 0)
                if Hk is None:
                    continue
                try:
                    p = cv2.perspectiveTransform(src[k:k + 1].reshape(-1, 1, 2), Hk).reshape(-1, 2)[0]
                except cv2.error:
                    continue
                e = float(np.linalg.norm(p - tgt[k]))
                if e > worst_err:
                    worst, worst_err = k, e
            if worst < 0 or worst_err <= cls.LOO_MAX_ERR_M:
                break
            inliers[worst] = False
            H, _ = cv2.findHomography(src[inliers], tgt[inliers], 0)
            if H is None:
                return inliers, None
        return inliers, H

    @staticmethod
    def _aniso(H: np.ndarray) -> float:
        if H is None or not np.isfinite(H).all() or abs(float(H[2, 2])) < 1e-9:
            return float('inf')
        A = H[:2, :2] / H[2, 2]
        try:
            s = np.linalg.svd(A, compute_uv=False)
        except np.linalg.LinAlgError:
            return float('inf')
        if not np.isfinite(s).all() or s.min() < 1e-12:
            return float('inf')
        return float(s.max() / s.min())

    @classmethod
    def _nms(cls, source: np.ndarray, target: np.ndarray):
        """Drop near-duplicate image points, keeping first occurrence."""
        keep = np.ones(len(source), dtype=bool)
        for i in range(len(source)):
            if not keep[i]:
                continue
            dists = np.linalg.norm(source[i + 1:] - source[i], axis=1)
            keep[i + 1:][dists < cls.NMS_DIST_PX] = False
        return source[keep], target[keep], keep

    @classmethod
    def _has_spread(cls, tgt: np.ndarray) -> bool:
        if len(tgt) < 4:
            return False
        rx = float(tgt[:, 0].max() - tgt[:, 0].min())
        ry = float(tgt[:, 1].max() - tgt[:, 1].min())
        return rx >= cls.MIN_TARGET_RANGE_M and ry >= cls.MIN_TARGET_RANGE_M and (rx * ry) >= cls.MIN_TARGET_AREA_M2

    @classmethod
    def _is_plausible(cls, H: np.ndarray) -> bool:
        """Scale sanity for a px->meters H (raw det is ~1e-11 at this scale).

        Bounds are intentionally wide: broadcast perspective makes the
        linear part span ~0.01-15 m/px across near/far field. Tight
        degenerate-collapse protection lives in the line-support veto,
        trust-region and keypoint-agreement guards instead.
        """
        if H is None or not np.isfinite(H).all() or abs(float(H[2, 2])) < 1e-9:
            return False
        A = H[:2, :2] / H[2, 2]
        try:
            s = np.linalg.svd(A, compute_uv=False)
        except np.linalg.LinAlgError:
            return False
        if not np.isfinite(s).all():
            return False
        # Typical: ~1920px per ~60m visible width -> ~0.03 m/px near field.
        return bool(s.min() >= 0.0005 and s.max() <= 30.0 and cls._aniso(H) < 50.0)

    @staticmethod
    def _mean_reproj_error(source: np.ndarray, target: np.ndarray, H: np.ndarray, inliers: np.ndarray) -> float:
        try:
            projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), H).reshape(-1, 2)
            errs = np.linalg.norm(projected[inliers] - target[inliers], axis=1)
            return float(np.mean(errs)) if len(errs) else float('inf')
        except cv2.error:
            return float('inf')

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if self.homography is None or points is None or len(points) == 0:
            return np.array([])
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if not np.isfinite(pts).all():
            return np.array([])
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)


class HomographySmoother:
    """
    Temporal smoothing / fallback for per-frame homographies.

    Broadcast footage has jittery keypoints. Smoothing H over time removes
    high-frequency jumps; falling back to the last good H avoids radar
    flicker when keypoints briefly drop below 4 or fail RANSAC validation.
    """
    def __init__(self, alpha: float = 0.35, max_stale_frames: int = 45):
        self.alpha = alpha
        self.max_stale_frames = max_stale_frames
        self.last_good_H: Optional[np.ndarray] = None
        self.stale_count = 0
        self.is_stale = False
        # Phase 1: per-label trust. Laplace prior (2 pseudo-successes of 3)
        # keeps new labels near 0.67 until evidence accumulates.
        n_labels = 32
        self.label_success = np.full(n_labels, 2.0)
        self.label_total = np.full(n_labels, 3.0)
        # Last image position per label, for the teleport gate.
        self.last_kp_xy: Dict[int, np.ndarray] = {}
        self.n_gated_teleport = 0

    # A1: a mirrored-but-valid fit must agree with recent history, else it
    # would poison the smoother and swing the radar across the pitch.
    # Compared in image space via H^-1 so the threshold is in pixels.
    MAX_CORNER_SHIFT_PX = 250.0
    # B2: weak fits may render LOW CONF but only nudge good history
    # (alpha * 0.3) instead of overwriting it.
    WEAK_FIT_MIN_INLIERS = 5

    # Labels teleporting further than this between calls are dropped for
    # that frame (false positive masquerading as a known landmark).
    TELEPORT_PX = 60.0
    # Candidates whose inlier set is dominated by distrusted labels are
    # rejected unless nothing better exists (handled by caller fallback).
    MIN_MEAN_INLIER_TRUST = 0.35

    def trust(self, idx: int) -> float:
        if 0 <= idx < len(self.label_success):
            return float(self.label_success[idx] / self.label_total[idx])
        return 0.67

    def mean_trust(self, idxs: np.ndarray) -> float:
        idxs = np.asarray(idxs, dtype=int).ravel()
        idxs = idxs[(idxs >= 0) & (idxs < len(self.label_success))]
        if len(idxs) == 0:
            return 0.67
        return float(np.mean(self.label_success[idxs] / self.label_total[idxs]))

    def gate_teleport(self, xy_all: np.ndarray, label_idx_all: np.ndarray) -> np.ndarray:
        """Keep mask dropping labels that jumped impossibly far (Phase 1)."""
        keep = np.ones(len(xy_all), dtype=bool)
        for k, (p, li) in enumerate(zip(np.asarray(xy_all), np.asarray(label_idx_all))):
            prev = self.last_kp_xy.get(int(li))
            if prev is not None and float(np.linalg.norm(np.asarray(p) - prev)) > self.TELEPORT_PX:
                keep[k] = False
                self.n_gated_teleport += 1
        return keep

    def observe_keypoints(self, xy_all: np.ndarray, label_idx_all: np.ndarray):
        for p, li in zip(np.asarray(xy_all), np.asarray(label_idx_all)):
            self.last_kp_xy[int(li)] = np.asarray(p, dtype=np.float32).copy()

    def _observe_fit(self, candidate: ViewTransformer):
        used = getattr(candidate, "used_label_idx", None)
        inl = getattr(candidate, "inlier_label_idx", None)
        if used is None:
            return
        used = np.asarray(used, dtype=int).ravel()
        valid = (used >= 0) & (used < len(self.label_success))
        self.label_total[used[valid]] += 1.0
        if inl is not None:
            inl = np.asarray(inl, dtype=int).ravel()
            inl = inl[(inl >= 0) & (inl < len(self.label_success))]
            self.label_success[inl] += 1.0

    def update(self, candidate: Optional[ViewTransformer]) -> Optional[np.ndarray]:
        if candidate is not None and candidate.is_valid:
            if candidate.inlier_label_idx is not None and len(candidate.inlier_label_idx):
                if self.mean_trust(candidate.inlier_label_idx) < self.MIN_MEAN_INLIER_TRUST:
                    return self._hold("distrusted labels")
            if self.last_good_H is None:
                self.last_good_H = candidate.homography
                self._observe_fit(candidate)
            else:
                shift = _homography_shift_px(candidate.homography, self.last_good_H)
                if shift > self.MAX_CORNER_SHIFT_PX:
                    return self._hold(f"mirror-jump {shift:.0f}px")
                # Guard against sudden jumps (wrong-side-of-pitch fits):
                # blend cautiously when the candidate differs wildly; weak
                # (few-inlier) fits only nudge so they can't swing the radar.
                diff = float(np.mean(np.abs(candidate.homography - self.last_good_H)))
                scale = float(np.mean(np.abs(self.last_good_H)) + 1e-6)
                alpha = self.alpha
                if diff / scale > 1.0:
                    alpha *= 0.5
                if candidate.n_inliers < self.WEAK_FIT_MIN_INLIERS:
                    alpha *= 0.3
                self.last_good_H = (
                    alpha * candidate.homography + (1.0 - alpha) * self.last_good_H
                )
                self._observe_fit(candidate)
            self.stale_count = 0
            self.is_stale = False
            return self.last_good_H
        return self._hold("invalid")

    def _hold(self, _reason: str = "") -> Optional[np.ndarray]:
        self.stale_count += 1
        self.is_stale = True
        if self.last_good_H is not None and self.stale_count <= self.max_stale_frames:
            return self.last_good_H
        return None

    def wrap(self, H: Optional[np.ndarray], stale: Optional[bool] = None) -> Optional[ViewTransformer]:
        """Wrap a (possibly smoothed) matrix in a lightweight transformer shim."""
        if H is None:
            return None
        obj = ViewTransformer.__new__(ViewTransformer)
        obj.homography = H
        obj.inlier_mask = None
        obj.n_inliers = ViewTransformer.MIN_INLIERS
        obj.reprojection_error_px = 0.0
        obj.is_valid = True
        obj.reject_reason = ""
        obj.aniso_ratio = 1.0
        obj.mirrored = False
        obj.inlier_targets = np.zeros((0, 2), np.float32)
        obj.line_score = float('inf')
        obj.line_support = 0.0
        obj.line_n = 0
        obj.used_label_idx = None
        obj.inlier_label_idx = np.array([], dtype=int)
        obj.fit_src = np.zeros((0, 2), np.float32)
        obj.fit_tgt = np.zeros((0, 2), np.float32)
        obj.is_stale = self.is_stale if stale is None else stale
        return obj


def mirror_targets(tgt: np.ndarray) -> np.ndarray:
    """Mirror pitch targets across the halfway line (x -> 105 - x).

    Used for the A2 mirror hypothesis: when the keypoint model swaps the
    left/right halves, the mirrored fit explains the same image points.
    """
    out = np.asarray(tgt, dtype=np.float32).copy()
    out[:, 0] = CONFIG.pitch_width_meters - out[:, 0]
    return out


def _homography_shift_px(H_new: np.ndarray, H_old: np.ndarray) -> float:
    """Mean image-space displacement between two image->pitch homographies.

    Projects reference pitch points (corners + center) through both
    inverses; a mirrored/wrong-side fit jumps by thousands of px, normal
    camera motion by tens. Returns inf when either H is unusable.
    """
    try:
        H_new = np.asarray(H_new, dtype=np.float64)
        H_old = np.asarray(H_old, dtype=np.float64)
        inv_new = np.linalg.inv(H_new)
        inv_old = np.linalg.inv(H_old)
        W, Hh = CONFIG.pitch_width_meters, CONFIG.pitch_height_meters
        ref = np.array(
            [[0, 0], [W, 0], [W, Hh], [0, Hh], [W / 2, Hh / 2]],
            dtype=np.float64,
        ).reshape(-1, 1, 2)
        p_new = cv2.perspectiveTransform(ref.astype(np.float32), inv_new.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        p_old = cv2.perspectiveTransform(ref.astype(np.float32), inv_old.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        if not (np.isfinite(p_new).all() and np.isfinite(p_old).all()):
            return float('inf')
        return float(np.mean(np.linalg.norm(p_new - p_old, axis=1)))
    except (np.linalg.LinAlgError, cv2.error, ValueError, TypeError):
        return float('inf')


def refine_with_prior(
    xy_all: np.ndarray,
    conf_all: Optional[np.ndarray],
    prior_H: np.ndarray,
    accept_px: float = 40.0,
    min_conf: float = 0.2,
    label_idx_all: Optional[np.ndarray] = None,
) -> Optional[ViewTransformer]:
    """B1: guided matching against the last good H.

    Reprojects every pitch label into the image via prior_H^-1 and keeps
    detections within accept_px, then refits. Recovers frames where the
    blind fit locks onto a mirrored consensus.
    """
    try:
        if xy_all is None or len(xy_all) == 0 or conf_all is None:
            return None
        xy_all = np.asarray(xy_all, dtype=np.float32)
        conf_all = np.asarray(conf_all, dtype=np.float32)
        lab = np.asarray(label_idx_all, dtype=int).ravel() if label_idx_all is not None else np.arange(len(xy_all), dtype=int)
        verts = CONFIG.vertices[lab]
        inv = np.linalg.inv(np.asarray(prior_H, dtype=np.float64))
        pred = cv2.perspectiveTransform(
            verts.reshape(-1, 1, 2).astype(np.float32),
            inv.astype(np.float32),
        ).reshape(-1, 2)
        dists = np.linalg.norm(xy_all - pred, axis=1)
        mask = (dists < accept_px) & (conf_all > min_conf)
        if int(mask.sum()) < ViewTransformer.MIN_INLIERS:
            return None
        return ViewTransformer(xy_all[mask], verts[mask], label_idx=lab[mask])
    except (np.linalg.LinAlgError, cv2.error, ValueError, TypeError, IndexError):
        return None

# Radar canvas keeps the 3:2 aspect of field_map.jpg (1470x980) so the
# texture is never stretched. Previously 700x450 (1.56:1) distorted it.
RADAR_WIDTH = 750
RADAR_HEIGHT = 500

# Measured touchline rectangle inside field_map.jpg (white boundary lines).
# Original resolution 1470x980: left/right x ~94/1375, top/bottom y ~62/916.
# Stored normalized so it survives any resize. (0,0)m maps to the touchline,
# NOT to the image corner (which includes the outer green margin + goals).
PITCH_BOUNDS_NORM = {
    "x0": 94.0 / 1470.0,
    "x1": 1375.0 / 1470.0,
    "y0": 62.0 / 980.0,
    "y1": 916.0 / 980.0,
}

# Points marginally outside (goal nets, throw-in run-up) are still drawable.
PITCH_DRAW_LEEWAY_M = 3.0

def _pitch_rect_px(pitch: np.ndarray):
    h, w = pitch.shape[:2]
    x0 = PITCH_BOUNDS_NORM["x0"] * w
    x1 = PITCH_BOUNDS_NORM["x1"] * w
    y0 = PITCH_BOUNDS_NORM["y0"] * h
    y1 = PITCH_BOUNDS_NORM["y1"] * h
    return x0, y0, x1, y1

def meters_to_pixel(xy_m: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Map pitch meters [0,105]x[0,68] to pixels inside the touchline rect."""
    x0, y0, x1, y1 = _pitch_rect_px(pitch)
    xy_m = np.asarray(xy_m, dtype=np.float32)
    px = np.empty_like(xy_m)
    px[..., 0] = x0 + (xy_m[..., 0] / CONFIG.pitch_width_meters) * (x1 - x0)
    px[..., 1] = y0 + (xy_m[..., 1] / CONFIG.pitch_height_meters) * (y1 - y0)
    return px

def draw_pitch() -> np.ndarray:
    field_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field_map.jpg')
    if not os.path.exists(field_map_path):
        return np.zeros((RADAR_HEIGHT, RADAR_WIDTH, 3), dtype=np.uint8)
    pitch_image = cv2.imread(field_map_path)
    return cv2.resize(pitch_image, (RADAR_WIDTH, RADAR_HEIGHT))

def foot_anchor_points(detections: sv.Detections, inset_frac: float = 0.04) -> np.ndarray:
    """Bottom-center raised by a fraction of box height.

    Raw BOTTOM_CENTER sits on the box edge, which includes shadow/grass
    bias below the feet. Raising it slightly centers the anchor on contact.
    """
    if len(detections) == 0:
        return np.array([])
    pts = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(np.float32)
    heights = (detections.xyxy[:, 3] - detections.xyxy[:, 1]).astype(np.float32)
    pts[:, 1] -= inset_frac * heights
    return pts

def draw_points_on_pitch(xy: np.ndarray, color: sv.Color, pitch: np.ndarray) -> np.ndarray:
    if xy is None or len(xy) == 0:
        return pitch
    xy = np.asarray(xy, dtype=np.float32)
    for p in xy:
        if p is None or not np.isfinite(p).all():
            continue
        # Drop wild homography outputs instead of plotting them mid-pitch.
        if not (-PITCH_DRAW_LEEWAY_M <= p[0] <= CONFIG.pitch_width_meters + PITCH_DRAW_LEEWAY_M
                and -PITCH_DRAW_LEEWAY_M <= p[1] <= CONFIG.pitch_height_meters + PITCH_DRAW_LEEWAY_M):
            continue
        u, v = meters_to_pixel(p.reshape(1, 2), pitch).ravel()
        h, w = pitch.shape[:2]
        cv2.circle(pitch, (int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))), 10, color.as_bgr(), -1)
    return pitch

# Pitch line polylines in meters, for line-alignment scoring/refinement.
def _pitch_line_polys():
    W, Hh = CONFIG.pitch_width_meters, CONFIG.pitch_height_meters
    polys = [
        [(0, 0), (W, 0), (W, Hh), (0, Hh), (0, 0)],
        [(W / 2, 0), (W / 2, Hh)],
        [(0, 13.85), (16.5, 13.85), (16.5, 54.15), (0, 54.15)],
        [(0, 24.85), (5.5, 24.85), (5.5, 43.15), (0, 43.15)],
        [(W - 16.5, 13.85), (W, 13.85), (W, 54.15), (W - 16.5, 54.15)],
        [(W - 5.5, 24.85), (W, 24.85), (W, 43.15), (W - 5.5, 43.15)],
        [(11, Hh / 2), (11, Hh / 2)],
        [(W - 11, Hh / 2), (W - 11, Hh / 2)],
    ]
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    polys.append([(W / 2 + 9.15 * np.cos(a), Hh / 2 + 9.15 * np.sin(a)) for a in th])
    return polys

PITCH_LINE_POLYS_M = _pitch_line_polys()

class FieldLines:
    """White-line evidence for one frame: scoring + ICP refinement of H.

    Scoring projects dense pitch-line samples into the image (via H^-1)
    and reads a distance transform: mean px distance to the nearest white
    pixel. Refinement iterates image->pitch DLT fits from nearest-white
    correspondences. Player/ball boxes are blacked out so kits don't vote.
    """
    WIDTH = 640
    MIN_SAMPLES = 80
    # Best-polyline support below this vetoes: a good H aligns at least
    # the detected structures; undetected ones must not count against it.
    # (Degenerate all-lines-onto-one-line collapses are caught by the
    # keypoint-agreement + trust-region guards in _maybe_refine.)
    MIN_SUPPORT = 0.30
    MIN_GROUP_SAMPLES = 15
    SUPPORT_PX = 6.0
    SUPPORT_MAX_ANGLE_DEG = 25.0
    REFINE_FROM_SUPPORT = 0.75
    REFINE_ITERS = 6
    REFINE_RADII_PX = (25.0, 20.0, 15.0, 12.0, 10.0, 8.0)
    # M-step RANSAC threshold (px @ WIDTH): nearest-white matches include
    # boards/text clutter, so the DLT refit itself must be robust.
    REFINE_RANSAC_PX = 3.0
    # Trimmed mean fraction: occluded/missing segments must not dominate.
    SCORE_QUANTILE = 0.7

    def __init__(self, frame_bgr: np.ndarray, exclude_boxes: Optional[np.ndarray] = None):
        h, w = frame_bgr.shape[:2]
        self.scale = self.WIDTH / float(w)
        small = cv2.resize(frame_bgr, (self.WIDTH, int(h * self.scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Flat-field: sun/shade splits the pitch; normalize local contrast
        # so shaded markings respond like lit ones.
        flat = cv2.GaussianBlur(gray, (0, 0), 15)
        norm = np.clip(gray.astype(np.float32) / (flat.astype(np.float32) + 10) * 100, 0, 255).astype(np.uint8)
        norm = cv2.GaussianBlur(norm, (3, 3), 0)
        # White top-hat with line kernels: thin bright ridges (markings)
        # regardless of absolute brightness (floodlit footage washes out).
        resp = np.zeros_like(norm, dtype=np.float32)
        for ang in (0, 45, 90, 135):
            k = np.zeros((15, 15), np.uint8)
            cv2.line(k, (0, 7), (14, 7), 1, 1)
            if ang:
                k = cv2.warpAffine(k, cv2.getRotationMatrix2D((7, 7), ang, 1), (15, 15))
            th = cv2.morphologyEx(norm, cv2.MORPH_TOPHAT, k)
            np.maximum(resp, th.astype(np.float32), out=resp)
        mask = (resp > 22).astype(np.uint8)
        # Markings lie on grass: require proximity to green (kills stands,
        # boards, sky). Drop the top strip (never pitch).
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        near_green = cv2.dilate(green, np.ones((7, 7), np.uint8))
        mask = cv2.bitwise_and(mask, mask, mask=near_green)
        mask[:int(mask.shape[0] * 0.12), :] = 0
        if exclude_boxes is not None:
            for x0, y0, x1, y1 in np.asarray(exclude_boxes).reshape(-1, 4):
                cv2.rectangle(mask,
                              (int(x0 * self.scale), int(y0 * self.scale)),
                              (int(x1 * self.scale), int(y1 * self.scale)), 0, -1)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 30:
                mask[lab == i] = 0
        self.mask = (mask > 0).astype(np.uint8)
        # Distance to nearest line pixel: lines->0, background->nonzero.
        inv = np.where(self.mask > 0, 0, 255).astype(np.uint8)
        self.dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        ys, xs = np.nonzero(self.mask)
        self.white_xy = np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) else np.zeros((0, 2), np.float32)
        # Edge direction at each white pixel (for orientation agreement).
        gray_f = cv2.resize(frame_bgr, (self.WIDTH, int(h * self.scale)))
        gray_f = cv2.cvtColor(gray_f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        self.white_ang = np.arctan2(gy[ys, xs], gx[ys, xs]).astype(np.float32) if len(xs) else np.zeros(0, np.float32)
        self._samples_m = self._dense_samples()
        self._samples_d = self._dense_dirs()
        self._samples_g = self._dense_groups()

    @staticmethod
    def _dense_samples(step_m: float = 1.0) -> np.ndarray:
        pts = []
        for poly in PITCH_LINE_POLYS_M:
            poly = np.asarray(poly, dtype=np.float32)
            for a, b in zip(poly[:-1], poly[1:]):
                seg_len = float(np.linalg.norm(b - a))
                n = max(1, int(seg_len / step_m))
                for t in np.linspace(0, 1, n, endpoint=False):
                    pts.append(a + (b - a) * t)
            pts.append(poly[-1])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _dense_dirs(step_m: float = 1.0) -> np.ndarray:
        dirs = []
        for poly in PITCH_LINE_POLYS_M:
            poly = np.asarray(poly, dtype=np.float32)
            for a, b in zip(poly[:-1], poly[1:]):
                seg_len = float(np.linalg.norm(b - a))
                n = max(1, int(seg_len / step_m))
                d = (b - a) / (seg_len + 1e-9)
                dirs.extend([d] * n)
            seg_len = float(np.linalg.norm(poly[-1] - poly[-2])) + 1e-9
            dirs.append((poly[-1] - poly[-2]) / seg_len)
        return np.array(dirs, dtype=np.float32)

    @staticmethod
    def _dense_groups(step_m: float = 1.0) -> np.ndarray:
        grp = []
        for gi, poly in enumerate(PITCH_LINE_POLYS_M):
            poly = np.asarray(poly, dtype=np.float32)
            for a, b in zip(poly[:-1], poly[1:]):
                seg_len = float(np.linalg.norm(b - a))
                n = max(1, int(seg_len / step_m))
                grp.extend([gi] * n)
            grp.append(gi)
        return np.array(grp, dtype=int)

    def _project(self, H_img2pitch: np.ndarray) -> Optional[np.ndarray]:
        try:
            inv = np.linalg.inv(np.asarray(H_img2pitch, dtype=np.float64))
        except (np.linalg.LinAlgError, ValueError, TypeError):
            return None
        img = cv2.perspectiveTransform(
            self._samples_m.reshape(-1, 1, 2), inv.astype(np.float32)).reshape(-1, 2)
        return (img * self.scale).astype(np.float32)

    # Line samples are only trusted near RANSAC inlier keypoints: mow
    # stripes and boards respond to the ridge filter far from truth.
    FOCUS_RADIUS_M = 20.0

    def _focused(self, focus_m: Optional[np.ndarray]) -> np.ndarray:
        if focus_m is None or len(focus_m) == 0:
            return np.ones(len(self._samples_m), dtype=bool)
        d = np.linalg.norm(
            self._samples_m[:, None, :] - np.asarray(focus_m, dtype=np.float32)[None, :, :], axis=2)
        return d.min(axis=1) <= self.FOCUS_RADIUS_M

    def _sample_dirs_img(self, H_img2pitch: np.ndarray) -> Optional[np.ndarray]:
        """Image-space direction of each dense sample's own pitch segment."""
        try:
            inv = np.linalg.inv(np.asarray(H_img2pitch, dtype=np.float64))
        except (np.linalg.LinAlgError, ValueError, TypeError):
            return None
        a = cv2.perspectiveTransform(
            self._samples_m.reshape(-1, 1, 2), inv.astype(np.float32)).reshape(-1, 2)
        b = cv2.perspectiveTransform(
            (self._samples_m + 0.5 * self._samples_d).reshape(-1, 1, 2),
            inv.astype(np.float32)).reshape(-1, 2)
        return (b - a) * self.scale

    def score(self, H_img2pitch: np.ndarray, focus_m: Optional[np.ndarray] = None) -> tuple[float, float, int]:
        """(trimmed-mean px, best-polyline support, n valid samples).

        Support needs a sample near a marking AND parallel to it, so ad
        boards/text (random orientations) can't fake alignment. Reported as
        the best single polyline so undetected structures (shade, occlusion)
        don't punish an otherwise good fit.
        """
        proj = self._project(H_img2pitch)
        if proj is None or len(self.white_xy) == 0:
            return float('inf'), 0.0, 0
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(self.white_xy)
        except ImportError:
            return float('inf'), 0.0, 0
        h, w = self.dist.shape
        valid = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
        valid &= self._focused(focus_m)
        n = int(valid.sum())
        if n < self.MIN_SAMPLES:
            return float('inf'), 0.0, n
        dd, ii = tree.query(proj[valid], k=1)
        dirs = self._sample_dirs_img(H_img2pitch)
        agree = np.ones(n, dtype=bool)
        if dirs is not None:
            seg = dirs[valid]
            want = np.arctan2(seg[:, 1], seg[:, 0])
            got = self.white_ang[ii] + np.pi / 2  # gradient perp = edge dir
            diff = np.abs((want - got + np.pi / 2) % np.pi - np.pi / 2)
            agree = diff <= np.deg2rad(self.SUPPORT_MAX_ANGLE_DEG)
        near = dd <= self.SUPPORT_PX
        good = near & agree
        gv = self._samples_g[valid]
        support = 0.0
        for gi in np.unique(gv):
            m = gv == gi
            if int(m.sum()) >= self.MIN_GROUP_SAMPLES:
                support = max(support, float(good[m].mean()))
        dc = np.minimum(dd, 40.0)
        ds = np.sort(dc)
        k = max(1, int(len(ds) * self.SCORE_QUANTILE))
        return float(np.mean(ds[:k])), support, n

    # Keypoints must sit on (near) a detected marking; far-off ones are
    # false positives. At WIDTH px scale.
    KP_SUPPORT_PX = 8.0

    def keypoint_support(self, xy_fullres: np.ndarray) -> np.ndarray:
        """Boolean mask: which image points lie near a marking."""
        pts = np.asarray(xy_fullres, dtype=np.float32).reshape(-1, 2) * self.scale
        h, w = self.dist.shape
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        ok = np.zeros(len(pts), dtype=bool)
        ok[inside] = self.dist[pts[inside, 1].astype(int), pts[inside, 0].astype(int)] <= self.KP_SUPPORT_PX
        return ok

    def _projected_area(self, H_img2pitch: np.ndarray) -> float:
        proj = self._project(H_img2pitch)
        if proj is None or len(proj) < 3:
            return 0.0
        try:
            return float(cv2.contourArea(proj.astype(np.float32)))
        except cv2.error:
            return 0.0

    def refine(self, H_img2pitch: np.ndarray, focus_m: Optional[np.ndarray] = None) -> tuple[Optional[np.ndarray], float, float]:
        """ICP, coarse-to-fine radii: nearest-white correspondences -> DLT."""
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(self.white_xy) if len(self.white_xy) else None
        except ImportError:
            tree = None
        if tree is None:
            return None, float('inf')
        best_H = np.asarray(H_img2pitch, dtype=np.float64)
        best_mean, best_sup, _ = self.score(best_H, focus_m)
        base_area = self._projected_area(best_H)
        H = best_H.copy()
        focus_sel = self._focused(focus_m)
        samples = self._samples_m[focus_sel]
        for radius in self.REFINE_RADII_PX[:self.REFINE_ITERS]:
            proj = self._project(H)
            if proj is None:
                break
            proj = proj[focus_sel]
            h, w = self.dist.shape
            inside = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
            if int(inside.sum()) < 4:
                break
            dd, ii = tree.query(proj[inside], k=1)
            use = dd <= radius
            if int(use.sum()) < 4:
                continue
            img_corr = (self.white_xy[ii[use]] / self.scale).astype(np.float32)
            pitch_corr = samples[inside][use]
            Hn, _ = cv2.findHomography(img_corr, pitch_corr, cv2.RANSAC, self.REFINE_RANSAC_PX)
            if Hn is None or not np.isfinite(Hn).all():
                continue
            # Anti-collapse: the projected pitch footprint must not shrink
            # or explode relative to the seed (degenerate slides ace support
            # by piling all lines onto one white blob).
            area = self._projected_area(Hn)
            if not (0.25 * base_area <= area <= 4.0 * base_area):
                continue
            s_mean, s_sup, _ = self.score(Hn, focus_m)
            if (s_sup, -s_mean) > (best_sup, -best_mean):
                best_mean, best_sup, best_H = s_mean, s_sup, Hn.astype(np.float64)
                H = best_H.copy()
        return (best_H if np.isfinite(best_mean) else None), best_mean, best_sup

def _best_candidate(xy: np.ndarray, conf: Optional[np.ndarray], primary_thresh: float, prior_H: Optional[np.ndarray] = None, smoother: Optional[HomographySmoother] = None, field_lines: Optional[FieldLines] = None):
    """Try the requested conf plus fallbacks; return (best, n_kp_at_primary).

    The keypoint model mislabels symmetric pitch points, so the threshold
    with the largest inlier consensus wins. A2: each threshold also fits a
    mirror hypothesis (x -> 105 - x); a valid non-mirrored fit is always
    preferred, mirror is fallback only (flagged). B1: when a prior H exists,
    guided matching is tried first. Phase 1: label teleport gate drops
    jumping false positives; trust scores rank otherwise-tied candidates.
    C1: with field_lines, candidates are gated/refined by white-line
    alignment, which mirror confusion cannot fool. A valid candidate is
    preferred; otherwise the one with the most inliers (LOW CONF fallback).
    """
    if conf is None:
        return None, 0
    xy = np.asarray(xy, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    label_all = np.arange(len(xy), dtype=int)
    n_primary = int((conf > primary_thresh).sum())

    prior = prior_H if prior_H is not None else (smoother.last_good_H if smoother is not None else None)

    def apply_lines(cand: Optional[ViewTransformer]) -> Optional[ViewTransformer]:
        """Score a candidate against white lines (veto happens post-refine)."""
        if cand is None or field_lines is None or cand.homography is None:
            return cand
        s, sup, n = field_lines.score(cand.homography, cand.inlier_targets)
        cand.line_score, cand.line_support, cand.line_n = s, sup, n
        return cand

    # Phase 1 teleport gate (only once history exists).
    gated = np.ones(len(xy), dtype=bool)
    if smoother is not None and smoother.last_good_H is not None and len(smoother.last_kp_xy):
        gated = smoother.gate_teleport(xy, label_all)
    xy_g, conf_g, lab_g = xy[gated], conf[gated], label_all[gated]
    if smoother is not None:
        smoother.observe_keypoints(xy, label_all)

    # NOTE: no line-support pre-filter here. The mask misses shaded/faint
    # markings, so filtering starves good structures (e.g. the left box)
    # and leaves degenerate clusters. Ghosts are outvoted by RANSAC count
    # and caught by the line-support veto instead.

    # B1: guided refit against history beats any blind fit.
    if prior is not None and len(xy_g) >= ViewTransformer.MIN_INLIERS:
        guided = refine_with_prior(xy_g, conf_g, prior, label_idx_all=lab_g)
        if guided is not None and guided.is_valid:
            if _homography_shift_px(guided.homography, prior) <= HomographySmoother.MAX_CORNER_SHIFT_PX:
                guided = apply_lines(guided)
                if guided.is_valid:
                    return _maybe_refine(guided, field_lines), n_primary

    def trust_of(c: ViewTransformer) -> float:
        if smoother is not None and c.inlier_label_idx is not None and len(c.inlier_label_idx):
            return smoother.mean_trust(c.inlier_label_idx)
        return 0.67

    # NOTE: line support is a veto ( Misaligned gross failures), never a
    # preference: focused support inflates for clustered fits, so ranking
    # by it selects degenerate consensuses over good large ones.

    best_normal = None   # valid, non-mirrored: always preferred
    best_mirror = None   # valid, mirrored: fallback only (flagged)
    best_invalid = None  # most inliers: LOW CONF fallback
    threshes = [primary_thresh] + [t for t in (0.3, 0.5, 0.7) if t != primary_thresh]
    # Distinct RNG seeds per (threshold, mirror) fit: RANSAC sampling
    # diversity finds different consensuses deterministically; ranking
    # then picks the largest valid one instead of the first lucky draw.
    for ti, th in enumerate(threshes):
        mask = conf_g > th
        src, tgt = xy_g[mask], CONFIG.vertices[lab_g[mask]]
        lab = lab_g[mask]
        if len(src) < 4:
            continue
        for mi, (tgt_hyp, mirrored) in enumerate(((tgt, False), (mirror_targets(tgt), True))):
            cand = ViewTransformer(src, tgt_hyp, label_idx=lab, prune_loocv=field_lines is None,
                                   rng_seed=1000 + ti * 10 + mi)
            cand.mirrored = mirrored
            cand = apply_lines(cand)
            if cand.is_valid:
                key = (cand.n_inliers, trust_of(cand), -cand.aniso_ratio)
                if not mirrored:
                    cur_key = (best_normal.n_inliers, trust_of(best_normal), -best_normal.aniso_ratio) if best_normal is not None else None
                    if best_normal is None or key > cur_key:
                        best_normal = cand
                else:
                    cur_key = (best_mirror.n_inliers, trust_of(best_mirror), -best_mirror.aniso_ratio) if best_mirror is not None else None
                    if best_mirror is None or key > cur_key:
                        best_mirror = cand
            elif best_invalid is None or cand.n_inliers > best_invalid.n_inliers:
                best_invalid = cand
    # A valid non-mirrored fit already returned above. Fall back to a valid
    # mirror (symmetric confusion), else the largest invalid consensus.
    if best_normal is not None:
        return _maybe_refine(best_normal, field_lines), n_primary
    if best_mirror is not None:
        return _maybe_refine(best_mirror, field_lines), n_primary
    return best_invalid, n_primary

def _maybe_refine(cand: Optional[ViewTransformer], field_lines: Optional[FieldLines]) -> Optional[ViewTransformer]:
    """ICP-refine a valid fit against white lines (C1), then gate (C1).

    Refinement first: a close-but-tilted fit (typical 8-inlier compromise)
    snaps onto the markings; only a still-unsupported result is vetoed.
    Support fraction (not distance mean) decides: missing segments can't
    inflate it, and degenerate slides into boards/seats keep keypoint
    agreement via the joint check below.
    """
    if cand is None or field_lines is None or not cand.is_valid:
        return cand
    if cand.line_n >= FieldLines.MIN_SAMPLES and cand.line_support < FieldLines.REFINE_FROM_SUPPORT:
        H_new, s_mean, s_sup = field_lines.refine(cand.homography, cand.inlier_targets)
        if H_new is not None and s_sup >= cand.line_support + 0.03:
            # Trust region: refinement snaps (tens of px), never teleports.
            # On divergence revert to the point fit (still judged by the
            # final gate below) instead of killing a usable candidate.
            if _homography_shift_px(H_new, cand.homography) > 80.0:
                cand.reject_reason = "refine diverged (kept points)"
                return cand
            # Joint objective: the snap must keep keypoint agreement.
            if len(cand.fit_src) >= 4:
                try:
                    proj = cv2.perspectiveTransform(
                        cand.fit_src.reshape(-1, 1, 2), H_new).reshape(-1, 2)
                    kp_err = float(np.mean(np.linalg.norm(proj - cand.fit_tgt, axis=1)))
                except cv2.error:
                    kp_err = float('inf')
                if kp_err > max(3.0, cand.reprojection_error_px * 1.5):
                    cand.reject_reason = f"refine kept (kp {kp_err:.1f}m)"
                    return cand
            cand.homography = H_new
            cand.line_score, cand.line_support = s_mean, s_sup
            cand.aniso_ratio = ViewTransformer._aniso(H_new)
            if not ViewTransformer._is_plausible(H_new):
                cand.is_valid = False
                cand.reject_reason = "refined H implausible"
                return cand
    if cand.is_valid and cand.line_n >= FieldLines.MIN_SAMPLES and cand.line_support < FieldLines.MIN_SUPPORT:
        cand.is_valid = False
        cand.reject_reason = f"line-unsupported {cand.line_support:.2f}"
    return cand

def generate_tactical_map(keypoints: sv.KeyPoints, all_detections: sv.Detections, all_team_ids: np.ndarray, team_classifier: TeamClassifier, current_ball: sv.Detections, keypoint_conf_threshold: float, smoother: Optional[HomographySmoother] = None, field_lines: Optional[FieldLines] = None) -> np.ndarray:
    pitch = draw_pitch()

    def overlay(text, color=(255, 255, 255)):
        cv2.putText(pitch, text, (30, pitch.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def badge(stale: bool = False, low_conf: bool = False, detail: str = ""):
        if stale or low_conf:
            label = []
            if low_conf:
                label.append("LOW CONF")
            if stale:
                label.append(f"HOLDING LAST MAP ({detail})" if detail else "HOLDING LAST MAP")
            elif detail:
                label.append(detail)
            cv2.putText(pitch, " | ".join(label), (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    transformer = None
    low_conf = False
    stale = False

    if len(keypoints.xy) == 0 or len(keypoints.xy[0]) == 0:
        n_kp = 0
        candidate = None
    else:
        raw_conf = keypoints.confidence[0] if len(keypoints.confidence) > 0 and keypoints.confidence[0] is not None else None
        prior = smoother.last_good_H if smoother is not None else None
        candidate, n_kp = _best_candidate(keypoints.xy[0], raw_conf, keypoint_conf_threshold, prior_H=prior, smoother=smoother, field_lines=field_lines)

    if smoother is not None:
        H = smoother.update(candidate)
        transformer = smoother.wrap(H)
        stale = smoother.is_stale and transformer is not None
        if transformer is None:
            # Cold start with no history: use the raw candidate even if it
            # failed validation, flagged LOW CONF, instead of a blank map.
            if candidate is not None and candidate.homography is not None:
                transformer = candidate
                low_conf = True
            else:
                overlay(f"No homography yet ({n_kp} keypoints)", (0, 0, 255))
                return pitch
    else:
        transformer = candidate
        if transformer is None:
            overlay(f"Awaiting 4+ keypoints (found {n_kp})")
            return pitch
        if not transformer.is_valid:
            if transformer.homography is not None:
                # Single-frame path (goal snapshots): still draw, flagged.
                low_conf = True
            else:
                overlay(f"Homography failed: {transformer.reject_reason}", (0, 0, 255))
                return pitch

    for team_id in np.unique(all_team_ids):
        mask = all_team_ids == team_id
        team_detections = all_detections[mask]
        color_idx = team_classifier.team_annotation_colors_map.get(int(team_id), 0)
        color = SUPERVISION_COLORS.colors[color_idx]

        team_xy = transformer.transform_points(foot_anchor_points(team_detections))
        pitch = draw_points_on_pitch(team_xy, color, pitch)

    if len(current_ball) > 0:
        ball_color = SUPERVISION_COLORS.colors[BALL_CLASS_ID]
        # Ball is often airborne: CENTER is the correct projection. BOTTOM_CENTER
        # previously biased it toward the feet plane.
        ball_xy = transformer.transform_points(current_ball.get_anchors_coordinates(sv.Position.CENTER))
        pitch = draw_points_on_pitch(ball_xy, ball_color, pitch)

    detail = ""
    if candidate is not None:
        detail = f"{candidate.n_inliers}/{n_kp} inliers err {candidate.reprojection_error_px:.1f}px"
        if candidate.mirrored:
            detail += " mirrored"
        if candidate.line_n >= FieldLines.MIN_SAMPLES:
            detail += f" line {candidate.line_score:.0f}px sup {candidate.line_support:.2f}"
    elif smoother is not None and stale:
        detail = f"stale {smoother.stale_count}"
    badge(stale=stale, low_conf=low_conf, detail=detail)

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
        players_team_ids = team_classifier.predict(
            get_crops(frame, players),
            tracker_ids=players.tracker_id if players.tracker_id is not None else None,
        )
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

    pitch = np.zeros((RADAR_HEIGHT, RADAR_WIDTH, 3), dtype=np.uint8)
    if hyperparams['plot']['show_radar']:
        all_detections = sv.Detections.merge([players, goalkeepers, referees])
        all_team_ids = np.concatenate([
            players_team_ids,
            goalkeepers_team_ids,
            np.full(len(referees), REFEREE_CLASS_ID)
        ])
        if 'homography' not in trackers or trackers['homography'] is None:
            trackers['homography'] = HomographySmoother()
        # White-line evidence; person/ball boxes blacked out so kits don't vote.
        excl = [d.xyxy for d in (all_detections, current_ball) if len(d) > 0]
        try:
            field_lines = FieldLines(
                frame, np.concatenate(excl, axis=0) if excl else None)
        except Exception:
            field_lines = None
        pitch = generate_tactical_map(
            keypoints,
            all_detections,
            all_team_ids,
            team_classifier,
            current_ball,
            hyperparams['detection']['keypoint_conf'],
            smoother=trackers['homography'],
            field_lines=field_lines
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
    try:
        cur = np.asarray(ball_pitch_pos, dtype=float).ravel()
        prev = np.asarray(prev_ball_pitch_pos, dtype=float).ravel()
        if cur.size < 2 or prev.size < 2 or not np.isfinite(cur).all() or not np.isfinite(prev).all():
            return None
    except (ValueError, TypeError):
        return None

    left_goal_posts = (CONFIG.goal_posts["left"]["top"][1], CONFIG.goal_posts["left"]["bottom"][1])
    if prev[0] > leeway_m and cur[0] <= leeway_m:
        if left_goal_posts[0] < cur[1] < left_goal_posts[1]:
            return "left"

    right_goal_posts = (CONFIG.goal_posts["right"]["top"][1], CONFIG.goal_posts["right"]["bottom"][1])
    if prev[0] < (CONFIG.pitch_width_meters - leeway_m) and cur[0] >= (CONFIG.pitch_width_meters - leeway_m):
        if right_goal_posts[0] < cur[1] < right_goal_posts[1]:
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
    smoother = HomographySmoother()

    for frame_idx, frame in enumerate(frame_generator):
        player_results = models['players'](frame, conf=player_conf, verbose=False)[0]
        keypoint_results = models['keypoints'](frame, conf=keypoint_conf, verbose=False)[0]
        ball_results = models['ball'](frame, verbose=False)[0]

        all_detections = sv.Detections.from_ultralytics(player_results)
        keypoints = sv.KeyPoints.from_ultralytics(keypoint_results)
        ball_detections = sv.Detections.from_ultralytics(ball_results)

        candidate = None
        if len(keypoints.xy) > 0 and len(keypoints.xy[0]) > 0 and keypoints.confidence is not None and len(keypoints.confidence[0]) > 0:
            excl = [d.xyxy for d in (all_detections, ball_detections) if len(d) > 0]
            try:
                frame_lines = FieldLines(
                    frame, np.concatenate(excl, axis=0) if excl else None)
            except Exception:
                frame_lines = None
            candidate, _ = _best_candidate(
                keypoints.xy[0], keypoints.confidence[0], keypoint_conf,
                prior_H=smoother.last_good_H, smoother=smoother,
                field_lines=frame_lines,
            )

        H_smooth = smoother.update(candidate)
        transformer = smoother.wrap(H_smooth)

        ball_pitch_pos = None
        if transformer is not None and len(ball_detections) > 0:
            ball_center_screen = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
            pts = transformer.transform_points(ball_center_screen)
            if len(pts) > 0 and np.isfinite(pts[0]).all():
                # Ignore wild ball projections (common when ball is airborne);
                # they otherwise create false goal triggers.
                if (-PITCH_DRAW_LEEWAY_M <= pts[0][0] <= CONFIG.pitch_width_meters + PITCH_DRAW_LEEWAY_M
                        and -PITCH_DRAW_LEEWAY_M <= pts[0][1] <= CONFIG.pitch_height_meters + PITCH_DRAW_LEEWAY_M):
                    ball_pitch_pos = pts[0]

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
            shot_transformer = shot_data['transformer']
            if shot_transformer is not None and shot_transformer.is_valid:
                for team_id in range(2):
                    player_mask = players_team_ids == team_id
                    gk_mask = goalkeepers_team_ids == team_id

                    team_players_screen = foot_anchor_points(players[player_mask])
                    team_gks_screen = foot_anchor_points(goalkeepers[gk_mask])

                    team_players_pitch = shot_transformer.transform_points(team_players_screen)
                    team_gks_pitch = shot_transformer.transform_points(team_gks_screen)

                    def _clean(pts):
                        out = []
                        for p in pts:
                            if p is None or not np.isfinite(p).all():
                                continue
                            if (-PITCH_DRAW_LEEWAY_M <= p[0] <= CONFIG.pitch_width_meters + PITCH_DRAW_LEEWAY_M
                                    and -PITCH_DRAW_LEEWAY_M <= p[1] <= CONFIG.pitch_height_meters + PITCH_DRAW_LEEWAY_M):
                                out.append([float(p[0]), float(p[1])])
                        return out

                    player_positions[f'team_{team_id+1}'] = _clean(team_players_pitch)
                    player_positions[f'goalkeeper_{team_id+1}'] = _clean(team_gks_pitch)
            
            all_shot_detections = sv.Detections.merge([players, goalkeepers])
            all_team_ids = np.concatenate([players_team_ids, goalkeepers_team_ids])

            excl = [d.xyxy for d in (all_shot_detections,) if len(d) > 0]
            try:
                shot_lines = FieldLines(
                    shot_frame_img_bgr, np.concatenate(excl, axis=0) if excl else None)
            except Exception:
                shot_lines = None
            tactical_map_img = generate_tactical_map(
                keypoints=shot_keypoints,
                all_detections=all_shot_detections,
                all_team_ids=all_team_ids,
                team_classifier=team_classifier,
                current_ball=sv.Detections.empty(),
                keypoint_conf_threshold=keypoint_conf,
                field_lines=shot_lines
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