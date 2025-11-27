import os
import numpy as np
import pandas as pd
import logging
from typing import Optional


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def filter_outliers(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Replace outliers in AGA/Left/Right angles with NaN if the point does not satisfy the original neighbor-difference condition."""
    data = data.copy()
    aga = list(data['Anterior Glottic Angle'])
    la = list(data['Angle of Left Cord'])
    ra = list(data['Angle of Right Cord'])

    for i in range(1, len(aga) - 1):
        cond = (
            abs(aga[i] - aga[i - 1]) < threshold and
            abs(aga[i] - aga[i + 1]) < threshold and
            abs(la[i] - la[i - 1]) < threshold and
            abs(la[i] - la[i + 1]) < threshold and
            abs(ra[i] - ra[i - 1]) < threshold and
            abs(ra[i] - ra[i + 1]) < threshold
        )
        if not cond:
            data.at[i, 'Anterior Glottic Angle'] = np.nan
            data.at[i, 'Angle of Left Cord'] = np.nan
            data.at[i, 'Angle of Right Cord'] = np.nan

    return data


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """Extract a frame from a video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning(f"Error opening video: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logging.warning(f"Error extracting frame {frame_number} from {video_path}")
        return None

    return frame
