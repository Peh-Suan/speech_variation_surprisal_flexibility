from scipy.interpolate import interp1d
import numpy as np
from . import mviewer

# ============================================================
# EMA data
# ============================================================

def process_ema_mat(mat_f):

    viewer = mviewer.Viewer(ignore_meta=True)
    viewer.load(mat_f)

    data = {}

    for idx in range(len(viewer.data_orig)):
        articulator = str(viewer.data_orig[idx][0][0])

        if articulator=="AUDIO": continue
        data[articulator] = {
            "fps": None,
            "pos": [],
        }
        fps = float(viewer.data_orig[idx][1][0][0])
        pos = viewer.data_orig[idx][2]

        data[articulator]["fps"] = fps

        for x, y, z in pos:
            data[articulator]["pos"].append((float(x), float(y), float(z)))

    return data

def interpolate_missing_frames(features):
    """
    Linearly interpolate NaN/Inf values along time independently
    for every feature column.

    features: (T, D)

    Interior gaps:
        linearly interpolated between preceding and following values

    Leading/trailing gaps:
        filled using the nearest valid value

    Returns:
        interpolated array, or None if any feature column has no
        valid observations at all
    """
    features = np.asarray(features, dtype=np.float64).copy()
    time_indices = np.arange(features.shape[0])

    for feature_idx in range(features.shape[1]):
        values = features[:, feature_idx]
        valid = np.isfinite(values)

        # Cannot interpolate a column that has no valid values
        if valid.sum() == 0:
            return None

        # Nothing to repair
        if valid.all():
            continue

        # np.interp:
        # - linearly interpolates interior missing values
        # - uses nearest valid value at the boundaries
        features[:, feature_idx] = np.interp(
            time_indices,
            time_indices[valid],
            values[valid],
        )

    return features

def get_aeema_articulatory_features(mat_f, start, end):
    data = process_ema_mat(mat_f)

    start_time = start  # in seconds
    end_time = end  # in seconds

    features = []
    for articulator in ["UL", "LL", "JAW", "TD", "TB", "TT"]:

        fps = data[articulator]["fps"]
        start_idx = round(start_time*fps)
        end_idx = round(end_time*fps)

        pos = data[articulator]["pos"][start_idx:end_idx+1]

        features.append(np.array([[x, y, z] for x, y, z in pos]))
    if len(features) == 0 or len(features[0]) == 0:
        return None

    features = np.concatenate(features, axis=1)

    features = interpolate_missing_frames(features)

    if features is None:
        return None

    x = features

    if not np.isfinite(x).all():
        return None
    
    return x

# ============================================================
# Annotated MRI data
# ============================================================
import os
import numpy as np


TARGET_CONTOUR = {
    1: 10,
    2: 10,
    3: 15,
    4: 5,
}


def extract_contours(contour_data):
    """
    Convert one MATLAB contour frame into a list of dictionaries.

    Each dictionary contains:
        points: (N, 2)
        labels: (N,)
        mu: scalar
    """

    seg = contour_data["contours"][0][0]["segment"][0]
    container = seg[0]

    clean = []

    for k in range(container.shape[1]):

        contour_struct_array = container[0, k]
        entry = contour_struct_array[0]

        pts = np.asarray(
            entry["v"][0],
            dtype=float,
        )

        labels = np.asarray(
            entry["i"][0]
        ).reshape(-1)

        mu = float(
            np.real(
                entry["mu"][0].item()
            )
        )

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"Expected contour points with shape (N, 2), "
                f"got {pts.shape}."
            )

        if len(pts) != len(labels):
            raise ValueError(
                "Contour point/label mismatch: "
                f"{len(pts)} points vs {len(labels)} labels."
            )

        clean.append({
            "points": pts,
            "labels": labels,
            "mu": mu,
        })

    return clean


def resample_polyline(points, target_n):
    """
    Resample a 2D polyline to exactly target_n points using
    arc-length parameterization.
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    if target_n <= 0:
        raise ValueError(
            "target_n must be positive."
        )

    if len(points) == 0:
        raise ValueError(
            "Cannot resample an empty polyline."
        )

    if len(points) == target_n:
        return points.copy()

    if len(points) == 1:
        return np.repeat(
            points,
            target_n,
            axis=0,
        )

    diffs = np.diff(
        points,
        axis=0,
    )

    dists = np.linalg.norm(
        diffs,
        axis=1,
    )

    cumulative = np.concatenate([
        [0.0],
        np.cumsum(dists),
    ])

    total_len = cumulative[-1]

    if total_len == 0:
        return np.repeat(
            points[:1],
            target_n,
            axis=0,
        )

    # Remove duplicate arc-length positions caused by
    # consecutive identical points.
    keep = np.concatenate([
        [True],
        np.diff(cumulative) > 0,
    ])

    cumulative = cumulative[keep]
    points = points[keep]

    target_pos = np.linspace(
        0,
        total_len,
        target_n,
    )

    x = np.interp(
        target_pos,
        cumulative,
        points[:, 0],
    )

    y = np.interp(
        target_pos,
        cumulative,
        points[:, 1],
    )

    return np.column_stack([
        x,
        y,
    ])


def frame_to_feature_vector(contours_clean):
    """
    Convert one frame's contours to a fixed point-coordinate matrix.

    Returns
    -------
    np.ndarray
        Shape (P, 2).

    Contour index 1 is resampled according to TARGET_CONTOUR.
    Other contours retain their original points.
    """

    rows = []

    for contour_id, contour in enumerate(
        contours_clean
    ):

        pts = contour["points"]
        labels = contour["labels"]

        if contour_id == 1:

            for label, target_n in TARGET_CONTOUR.items():

                pts_label = pts[
                    labels == label
                ]

                if len(pts_label) == 0:
                    raise ValueError(
                        f"Contour {contour_id} "
                        f"has no points for label {label}."
                    )

                pts_resampled = resample_polyline(
                    pts_label,
                    target_n,
                )

                rows.extend(
                    pts_resampled
                )

        else:

            rows.extend(pts)

    out = np.asarray(
        rows,
        dtype=float,
    )

    if out.ndim != 2 or out.shape[1] != 2:
        raise ValueError(
            f"Expected output shape (P, 2), got {out.shape}."
        )

    return out


def mat_to_mp4(mat_f):
    return (
        mat_f
        .replace(
            "/USC-75SpeakerAnnot16/",
            "/",
        )
        .replace(
            "/track/",
            "/2drt/video/",
        )
        .replace(
            "track.mat",
            "video.mp4",
        )
    )


def interpolate(X, n=5):
    """
    Temporally interpolate (T, P, C) to (n, P, C).
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    if X.ndim != 3:
        raise ValueError(
            f"Expected X with shape (T, P, C), "
            f"got {X.shape}."
        )

    if n <= 0:
        raise ValueError(
            "n must be positive."
        )

    T, P, C = X.shape

    if T == 0:
        return None

    if T == n:
        return X.copy()

    # np.interp also handles T == 1 by repeating
    # the sole observation.
    t = np.linspace(
        0,
        1,
        T,
    )

    t_new = np.linspace(
        0,
        1,
        n,
    )

    new_X = np.empty(
        (n, P, C),
        dtype=float,
    )

    for j in range(P):
        for k in range(C):
            new_X[:, j, k] = np.interp(
                t_new,
                t,
                X[:, j, k],
            )

    return new_X


def get_aemri75anno_articulatory_features(
    mat_f,
    start,
    end
):
    from scipy.io import loadmat
    import cv2

    if end <= start:
        raise ValueError(
            f"end ({end}) must be greater than start ({start})."
        )

    data = loadmat(mat_f)

    if "trackdata" not in data:
        raise KeyError(
            f"'trackdata' not found in {mat_f}"
        )

    contour_datas = data["trackdata"][0]

    mp4_f = mat_to_mp4(mat_f)

    if not os.path.exists(mp4_f):
        raise FileNotFoundError(
            f"Could not find corresponding video: {mp4_f}"
        )

    cap = cv2.VideoCapture(mp4_f)

    if not cap.isOpened():
        cap.release()
        raise ValueError(
            f"Could not open video: {mp4_f}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    video_num_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"Invalid FPS {fps} for {mp4_f}"
        )

    # Retaining your current frame-selection convention.
    start_frame_idx = round(
        start * fps
    )

    end_frame_idx = round(
        end * fps
    )

    # Prevent negative/out-of-range indexing.
    start_frame_idx = max(
        0,
        start_frame_idx,
    )

    end_frame_idx = min(
        len(contour_datas) - 1,
        end_frame_idx,
    )

    if start_frame_idx > end_frame_idx:
        return None

    all_contours = []
    expected_shape = None

    for contour_data in contour_datas[
        start_frame_idx:
        end_frame_idx + 1
    ]:

        contours = extract_contours(
            contour_data
        )

        frame_features = frame_to_feature_vector(
            contours
        )

        if expected_shape is None:
            expected_shape = frame_features.shape

        elif frame_features.shape != expected_shape:
            raise ValueError(
                "Inconsistent contour shape within segment: "
                f"expected {expected_shape}, "
                f"got {frame_features.shape}."
            )

        all_contours.append(
            frame_features
        )

    if not all_contours:
        return None

    features = np.stack(
        all_contours,
        axis=0,
    )

    if features is None:
        return None

    # (T, P, 2) -> (T, D)
    features = features.reshape(
        features.shape[0],
        -1,
    )

    if not np.isfinite(features).all():
        return None

    return features

# ============================================================
# MRI video data
# ============================================================

import os
import cv2
import numpy as np
from scipy.interpolate import interp1d


# ============================================================
# Shared utilities
# ============================================================

def _normalize_frames(frames):
    """
    Z-normalize a video segment as in the original Dataset classes.
    """
    return (
        (frames - frames.mean()) /
        (frames.std() + 1e-5)
    ).astype(np.float32)


def _read_video_segment(file_path, start, end):
    """
    Read frames corresponding to start:end seconds from an AVI/MP4 file.

    Returns:
        (T, H, W), using channel 0 as in the original code.
    """
    if file_path is None or not os.path.exists(file_path):
        return None

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        cap.release()
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        return None

    start_frame = round(start * fps)
    end_frame = round(end * fps)

    if end_frame < start_frame:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []

    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()

        if not ret:
            break

        # Same behavior as your original datasets:
        # take channel 0 rather than cv2.cvtColor(...)
        frames.append(frame[:, :, 0])

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.stack(frames, axis=0).astype(np.float32)

    # Defensive check
    if not np.isfinite(frames).all():
        return None

    return frames


def _get_mri_articulatory_features(
    file_path,
    start,
    end,
    stats=None
):
    """
    Shared implementation for all three MRI datasets.

    Returns:
        (T, H, W)
    """
    frames = _read_video_segment(
        file_path,
        start,
        end
    )

    if frames is None:
        return None

    if stats is not None:
        mean, sd = stats
        frames = ((frames - mean) / sd).astype(np.float32)

    return frames.astype(np.float32)

def get_aemri_articulatory_features(
    file_path,
    start,
    end,
    stats=None
):
    """
    USC-TIMIT MRI.

    file_path: path to .avi
    start/end: seconds

    Returns:
        (T, H, W)
    """
    return _get_mri_articulatory_features(
        file_path=file_path,
        start=start,
        end=end,
        stats=stats
    )

def get_frmri_articulatory_features(
    file_path,
    start,
    end,
    stats=None
):
    """
    French MRI.

    file_path: path to .avi
    start/end: seconds

    Returns:
        (T, H, W)
    """
    return _get_mri_articulatory_features(
        file_path=file_path,
        start=start,
        end=end,
        stats=stats
    )

def get_aemri75_articulatory_features(
    file_path,
    start,
    end,
    stats=None
):
    """
    USC-75 MRI.

    file_path: path to .mp4
    start/end: seconds

    Returns:
        (T, H, W)
    """
    return _get_mri_articulatory_features(
        file_path=file_path,
        start=start,
        end=end,
        stats=stats
    )

# ============================================================
# Ultrasound + lip imaging video data
# ============================================================

def _mask_ultrasound_image_bool(h, w):
    """The fan mask as a boolean array, so it can be reused rather than re-derived."""
    cx, cy = w // 2, h + 6
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    dx, dy = X - cx, cy - Y
    radius = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    return ((radius >= 25) & (radius <= 60)
            & (angle >= (np.pi / 3) * 0.7) & (angle <= (np.pi / 3) * 2.35))

from functools import lru_cache
import numpy as np
import cv2
import os


@lru_cache(maxsize=32)
def _load_beus_frames(file_path):
    # Does NOT load the whole recording into RAM.
    return np.load(file_path, mmap_mode="r")


@lru_cache(maxsize=256)
def _load_beus_fps(fps_path):
    return float(np.asarray(np.load(fps_path)).squeeze())


@lru_cache(maxsize=16)
def _cached_ultrasound_mask(h, w):
    return _mask_ultrasound_image_bool(h, w)


def _beus_streams(file_path, spatial_size=112, start=None, end=None):

    if file_path is None or not os.path.exists(file_path):
        return None

    if not file_path.endswith("_frames.npy"):
        raise ValueError("file_path should point to a *_frames.npy file.")

    fps_path = file_path.replace("_frames.npy", "_fps.npy")

    if not os.path.exists(fps_path):
        raise FileNotFoundError(f"FPS file not found: {fps_path}")

    # -----------------------------
    # Memory-map instead of loading
    # -----------------------------
    frames_all = _load_beus_frames(file_path)
    fps = _load_beus_fps(fps_path)

    if frames_all.ndim != 3:
        raise ValueError(
            f"Expected video shape (T, H, W), got {frames_all.shape}"
        )

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {fps}")

    # -----------------------------
    # Slice FIRST
    # -----------------------------
    if start is not None and end is not None:

        a = max(0, round(start * fps))
        b = min(round(end * fps), frames_all.shape[0] - 1)

        if b < a:
            return None

        frames = frames_all[a:b + 1]

    else:
        frames = frames_all

    if frames.shape[0] == 0:
        return None

    # -----------------------------
    # Convert only requested frames
    # -----------------------------
    frames = np.asarray(frames, dtype=np.float32)

    if not np.isfinite(frames).all():
        return None

    # -----------------------------
    # Resize only requested frames
    # -----------------------------
    if spatial_size is not None:
        frames = np.stack([
            cv2.resize(
                f,
                (spatial_size, spatial_size),
                interpolation=cv2.INTER_AREA,
            )
            for f in frames
        ])

    height = frames.shape[1]

    if height % 2 != 0:
        raise ValueError(
            f"Frame height must be divisible by 2, got {height}"
        )

    half = height // 2

    ultrasound = frames[:, :half, :].copy()
    lips = frames[:, half:, :].copy()

    mask = _cached_ultrasound_mask(
        ultrasound.shape[1],
        ultrasound.shape[2],
    )

    ultrasound[:, ~mask] = 0.0

    return ultrasound, lips, mask

def get_beus_articulatory_features(file_path, start, end, spatial_size=112,
                                   stats=None):
    """
    Extract one segment from a BE ultrasound recording.
 
    Returns
    -------
    (T, 2, H/2, W) float32, TIME-FIRST.
        view 0 = ultrasound (fan-masked), view 1 = lips.
        With spatial_size=112: (T, 2, 56, 112).
 
    `stats`: ((us_mean, us_sd), (lip_mean, lip_sd)) from beus_recording_stats
        for THIS recording. Required — see the note below.
    """
    out = _beus_streams(file_path, spatial_size, start, end)
    if out is None:
        return None
    us, lp, mask = out
 
    if stats is None:
        # Deliberately not falling back to per-token normalisation: that forces
        # every segment to mean 0 / sd 1 and erases all between-token intensity
        # variation, which is part of what distinguishes the tokens. Silently
        # doing it on a forgotten argument is worse than stopping.
        raise ValueError(
            "stats is required: pass beus_recording_stats(file_path, spatial_size) "
            "for this recording. Without it these are raw pixel values.")
 
    (us_mean, us_sd), (lip_mean, lip_sd) = stats
    us = (us - us_mean) / us_sd
    us[:, ~mask] = 0.0          # keep the out-of-fan region at exactly 0
    lp = (lp - lip_mean) / lip_sd
 
    x = np.stack([us, lp], axis=1).astype(np.float32)    # (T, 2, H/2, W)
    if not np.isfinite(x).all():
        return None
    return x