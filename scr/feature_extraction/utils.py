import os
import logging
from typing import Optional

import cv2
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def filter_outliers(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Replace outliers in AGA/Left/Right angles with NaN if the point does not
    satisfy the original neighbor-difference condition.

    The condition is applied to:
        - 'Anterior Glottic Angle'
        - 'Angle of Left Cord'
        - 'Angle of Right Cord'

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with the three angle columns.
    threshold : float
        Maximum allowed difference with both neighbors for a point
        to be considered non-outlier.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with outliers replaced by NaN.
    """
    data = data.copy()

    # Convert to numpy arrays for robustness and speed
    aga = data['Anterior Glottic Angle'].to_numpy(copy=True)
    la = data['Angle of Left Cord'].to_numpy(copy=True)
    ra = data['Angle of Right Cord'].to_numpy(copy=True)

    n = len(aga)
    if n < 3:
        # Troppo pochi punti per applicare la logica sui vicini
        data['Anterior Glottic Angle'] = aga
        data['Angle of Left Cord'] = la
        data['Angle of Right Cord'] = ra
        return data

    for i in range(1, n - 1):
        cond = (
            abs(aga[i] - aga[i - 1]) < threshold and
            abs(aga[i] - aga[i + 1]) < threshold and
            abs(la[i] - la[i - 1]) < threshold and
            abs(la[i] - la[i + 1]) < threshold and
            abs(ra[i] - ra[i - 1]) < threshold and
            abs(ra[i] - ra[i + 1]) < threshold
        )
        if not cond:
            aga[i] = np.nan
            la[i] = np.nan
            ra[i] = np.nan

    data['Anterior Glottic Angle'] = aga
    data['Angle of Left Cord'] = la
    data['Angle of Right Cord'] = ra

    return data


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract a frame from a video using OpenCV.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_number : int
        Index of the frame to extract (0-based).

    Returns
    -------
    Optional[np.ndarray]
        The extracted frame (BGR image) if successful, otherwise None.
    """
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

