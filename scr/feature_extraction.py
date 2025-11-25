"""
Feature extraction pipeline for laryngeal paralysis angle time series.

Functions
---------
- ensure_dir       : Create directory if it does not exist.
- filter_outliers  : Neighbor-based spike removal with NaN imputation
                     (as implemented in the original analysis).
- check_consecutives : Detect sustained excursions above/below GMM means
                       (not used in the current CLI pipeline).
- create_exc_frame : Export grouped exceptional frames (optional, currently unused).
- extract_frame    : Get a specific frame from a video.
- ggm              : Robust 2-component GMM after outlier rejection.
- feature_extract  : Compute statistical / causal / velocity features, save difference image.
- find_angle_file  : Locate the angle CSV for a given subject.

CLI
---
Example usage:

    python src/feature_extraction.py \
        --videos /path/to/videos \
        --angles /path/to/angles_csvs \
        --patients /path/to/Patients_df.xlsx \
        --outdir results \
        --images-subdir images \
        --threshold 30 \
        --contamination 0.05

Outputs
-------
- features.xlsx in outdir
- Optional difference images in outdir/images_subdir
"""

import os
import sys
import argparse
import logging

from typing import List, Optional

import numpy as np
import pandas as pd
import cv2

from scipy.stats import kendalltau
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from moviepy.editor import VideoFileClip


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

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

    IMPORTANT
    ---------
    The logic here is kept exactly as in the original analysis, even if it may
    not match the narrative in the docstring. This preserves numerical results.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns:
        ['Anterior Glottic Angle', 'Angle of Left Cord', 'Angle of Right Cord'].
    threshold : float
        Neighbor difference threshold.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with NaNs set for detected outliers.
    """
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


def check_consecutives(means: np.ndarray, aga_try: np.ndarray) -> bool:
    """
    Check if there are >=10 consecutive frames above max(mean) and >=10
    consecutive frames above min(mean).

    NOTE
    ----
    This function is retained for completeness and backwards compatibility.
    It is not used in the main CLI pipeline.

    Parameters
    ----------
    means : np.ndarray
        Array of GMM means (shape (2, 1) or (2,)).
    aga_try : np.ndarray
        1D array segment of Anterior Glottic Angle values.

    Returns
    -------
    bool
        True if both conditions are satisfied.
    """
    means = np.array(means).reshape(-1)
    max_mean = np.max(means)
    min_mean = np.min(means)

    # Count max runs
    cons = 0
    max_run = 0
    for v in aga_try:
        if v > max_mean:
            cons += 1
            max_run = max(max_run, cons)
        else:
            cons = 0

    # Count min runs (note: original checks > min(mean))
    cons = 0
    min_run = 0
    for v in aga_try:
        if v > min_mean:
            cons += 1
            min_run = max(min_run, cons)
        else:
            cons = 0

    return (max_run >= 10) and (min_run >= 10)


def create_exc_frame(aga_ok: list, out_path: str = "frames_normal.xlsx") -> None:
    """
    Group patient IDs and frames into rows, pad with NaNs, and export to Excel.

    Parameters
    ----------
    aga_ok : list
        List containing alternating patient IDs (str) and frame indices (int),
        as produced by the original analysis code.
    out_path : str, optional
        Excel file path to save, by default "frames_normal.xlsx".
    """
    frames_list = []
    pat_list = []

    for item in aga_ok:
        if isinstance(item, str):
            frames_list.append(pat_list)
            pat_list = [item]
        elif isinstance(item, int):
            pat_list.append(item)

    if pat_list not in frames_list:
        frames_list.append(pat_list)

    max_length = max(len(sublist) for sublist in frames_list)
    padded = [sub + [np.nan] * (max_length - len(sub)) for sub in frames_list]
    df = pd.DataFrame(padded)
    df.to_excel(out_path, index=False, header=False)


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extract a frame from a video using OpenCV.

    Parameters
    ----------
    video_path : str
        Path to video file.
    frame_number : int
        0-based frame index.

    Returns
    -------
    np.ndarray or None
        BGR frame as a NumPy array, or None if extraction failed.
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


def ggm(time_series: np.ndarray, contamination: float) -> np.ndarray:
    """
    Robustly fit a 2-component Gaussian Mixture to non-NaN values after
    elliptic envelope outlier rejection.

    Parameters
    ----------
    time_series : array-like
        1D array-like of values (can include NaN).
    contamination : float
        Expected outlier proportion in (0, 0.5).

    Returns
    -------
    np.ndarray
        Means array of shape (2, 1).
    """
    ts = np.array(time_series).reshape(-1)
    aga_nonan = ts[~np.isnan(ts)].reshape(-1, 1)

    envelope = EllipticEnvelope(contamination=contamination)
    outliers = envelope.fit_predict(aga_nonan)
    inliers = aga_nonan[outliers == 1].reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(inliers)

    return gmm.means_


def feature_extract(
    name: str,
    data: pd.DataFrame,
    t0: int,
    video_path: str,
    target: int,
    images_outdir: Optional[str] = None
) -> list:
    """
    Compute statistical, velocity, and causality features; optionally save
    a difference image (max vs min AGA frames).

    NOTE
    ----
    The logic and parameter values (including the fixed contamination=0.05
    inside this function) are kept exactly as in the original analysis.

    Returns
    -------
    list
        [Name, Maximum Angle, Minimum Angle, Correlation, Granger Causality,
         Minimum Std, Diff Std, Velocity Correlation, Velocity Granger Causality,
         Velocity Minimum Std, Velocity Diff Std, Means Diff, Bimodal Std,
         Class, Path (optional)]
    """
    # Percentiles on AGA
    max_ang = np.nanpercentile(data['Anterior Glottic Angle'], 97)
    min_ang = np.nanpercentile(data['Anterior Glottic Angle'], 3)

    # Note: original code swaps labels; keep behavior as-is
    la = np.array(data['Angle of Right Cord'])
    ra = np.array(data['Angle of Left Cord'])
    idx = ~np.isnan(ra) & ~np.isnan(la)

    # Correlation
    corr, _ = kendalltau(la[idx], ra[idx])

    # Granger causality (angles)
    try:
        gc1 = grangercausalitytests(
            np.column_stack((la[idx], ra[idx])),
            maxlag=1,
            verbose=False
        )
        gc2 = grangercausalitytests(
            np.column_stack((ra[idx], la[idx])),
            maxlag=1,
            verbose=False
        )
        grang = max(
            gc1[1][0]['ssr_chi2test'][0],
            gc2[1][0]['ssr_chi2test'][0]
        )
    except Exception:
        grang = np.nan

    # Dispersion
    min_std = min(np.std(la[idx]), np.std(ra[idx]))
    diff_std = abs(np.std(la[idx]) - np.std(ra[idx]))

    # Velocities
    vel_la = la[idx][1:] - la[idx][:-1]
    vel_ra = ra[idx][1:] - ra[idx][:-1]
    vel_corr, _ = kendalltau(vel_la, vel_ra)

    # Granger on velocities
    try:
        gc1v = grangercausalitytests(
            np.column_stack((vel_la, vel_ra)),
            maxlag=1,
            verbose=False
        )
        gc2v = grangercausalitytests(
            np.column_stack((vel_ra, vel_la)),
            maxlag=1,
            verbose=False
        )
        vel_grang = max(
            gc1v[1][0]['ssr_chi2test'][0],
            gc2v[1][0]['ssr_chi2test'][0]
        )
    except Exception:
        vel_grang = np.nan

    vel_min_std = min(np.std(vel_la), np.std(vel_ra))
    vel_diff_std = abs(np.std(vel_la) - np.std(vel_ra))

    # GMM-based bimodality metrics
    means = ggm(data['Anterior Glottic Angle'], contamination=0.05)
    means_diff = max(means) - min(means)
    means_diff = float(np.array(means_diff).reshape(-1)[0])  # original indexing style
    std_bimod = np.nanstd(np.array(data['Anterior Glottic Angle']))

    # Optional difference image between min/max AGA frames
    path_out: Optional[str] = None
    try:
        if images_outdir is not None:
            ensure_dir(images_outdir)

            frame_min = int(np.nanargmin(data['Anterior Glottic Angle'])) + int(t0)
            frame_max = int(np.nanargmax(data['Anterior Glottic Angle'])) + int(t0)

            f1 = extract_frame(video_path, frame_min)
            f2 = extract_frame(video_path, frame_max)

            if f1 is not None and f2 is not None:
                g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

                # Z-score each frame
                g1 = (g1 - np.mean(g1)) / (np.std(g1) + 1e-8)
                g2 = (g2 - np.mean(g2)) / (np.std(g2) + 1e-8)

                diff = np.abs(g1 - g2)
                # Median thresholding; original used np.concatenate(rs1) which
                # would be invalid for 2D arrays. This keeps the fixed behavior.
                thr = np.median(diff)
                diff = diff * (diff > thr)

                # Normalize to 0-255
                maxv = diff.max() if diff.max() > 0 else 1.0
                diff = (diff / maxv) * 255.0
                diff = diff.astype(np.uint8)

                path_out = os.path.join(images_outdir, f"{name}.jpg")
                cv2.imwrite(path_out, diff)
    except Exception as e:
        logging.warning(f"Difference image failed for {name}: {e}")

    result = [
        name, max_ang, min_ang, corr, grang, min_std, diff_std,
        vel_corr, vel_grang, vel_min_std, vel_diff_std, means_diff,
        std_bimod, target
    ]
    if path_out is not None:
        result.append(path_out)

    return result


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def find_angle_file(angle_files: List[str], sub_id: str) -> Optional[str]:
    """
    Return the first angle file name containing the subject id, or None.

    Parameters
    ----------
    angle_files : list of str
        List of available angle CSV filenames.
    sub_id : str
        Subject identifier to search for.

    Returns
    -------
    str or None
        Matching filename or None if no match is found.
    """
    for f in angle_files:
        if sub_id in f:
            return f
    return None


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description="Feature extraction for laryngeal paralysis dataset"
    )
    p.add_argument("--videos", required=True,
                   help="Directory containing original videos")
    p.add_argument("--angles", required=True,
                   help="Directory containing AGATI angle CSV files")
    p.add_argument("--patients", required=True,
                   help="Path to Patients_df.xlsx")
    p.add_argument("--outdir", default="results",
                   help="Output directory for features")
    p.add_argument(
        "--images-subdir",
        default=None,
        help=("Subdirectory inside outdir to save difference images "
              "(e.g., 'images/normal')")
    )
    p.add_argument("--threshold", type=float, default=30.0,
                   help="Outlier neighbor threshold")
    p.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="EllipticEnvelope contamination for GMM (used in gating GMM)"
    )
    p.add_argument("--split-name", default="bilateral",
                   help="Label to use in optional frames filename")
    p.add_argument("--target", type=int, default=0,
                   help="Class label to store with features")
    p.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return p.parse_args()


def main() -> None:
    """Main entry point for the CLI feature extraction."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel))

    ensure_dir(args.outdir)

    images_outdir: Optional[str] = None
    if args.images_subdir:
        images_outdir = os.path.join(args.outdir, args.images_subdir)
        ensure_dir(images_outdir)

    # Load patients and angle file list
    patients_df = pd.read_excel(args.patients)
    angle_ts_files = sorted(
        [f for f in os.listdir(args.angles) if f.lower().endswith(".csv")]
    )

    features = []
    aga_ok = []

    for i in range(len(patients_df)):
        sub_id = str(patients_df.loc[i, 'sub_id'])
        video_name = str(patients_df.loc[i, 'name'])
        t0_val = patients_df.loc[i, 't0']
        tf_val = patients_df.loc[i, 'tf']

        ts_file = find_angle_file(angle_ts_files, sub_id)
        if ts_file is None:
            logging.info(f"{sub_id}: angle file not found (remember to run AGATI)")
            continue

        # Load and sanitize angles
        data = pd.read_csv(os.path.join(args.angles, ts_file))

        # Clip invalid AGA values
        data.loc[data['Anterior Glottic Angle'] > 90,
                 ['Anterior Glottic Angle']] = np.nan
        data.loc[data['Anterior Glottic Angle'] < 0,
                 ['Anterior Glottic Angle']] = np.nan

        # Filter outliers only on non-NaN rows, then write back
        valid_mask = data['Anterior Glottic Angle'].notna()
        filtered = filter_outliers(
            data.loc[valid_mask, [
                'Anterior Glottic Angle',
                'Angle of Left Cord',
                'Angle of Right Cord'
            ]],
            args.threshold
        )
        data.loc[valid_mask, [
            'Anterior Glottic Angle',
            'Angle of Left Cord',
            'Angle of Right Cord'
        ]] = filtered

        aga = np.array(data['Anterior Glottic Angle'])

        # Video frame rate for time slicing
        video_path = os.path.join(args.videos, video_name)
        try:
            clip = VideoFileClip(video_path)
            frame_rate = clip.fps
        except Exception as e:
            logging.warning(
                f"{sub_id}: could not read video '{video_path}' ({e})"
            )
            frame_rate = None

        # Apply time window if available
        t0 = 0
        if frame_rate is not None and pd.notna(t0_val) and str(t0_val) != 'all':
            try:
                t0 = int(frame_rate * float(t0_val))
                tf = int(frame_rate * float(tf_val))
                aga = aga[t0:tf]
            except Exception as e:
                logging.warning(
                    f"{sub_id}: invalid t0/tf values ({e}); using full series"
                )
                t0 = 0

        # Descriptive logging
        logging.info(
            f"{sub_id}: Max {np.nanmax(aga):.3f}, "
            f"Min {np.nanmin(aga):.3f}, "
            f"Mean {np.nanmean(aga):.3f}, "
            f"Std {np.nanstd(aga):.3f}"
        )

        # GMM means for gate (gating GMM uses CLI contamination)
        try:
            means = ggm(aga, contamination=args.contamination)
        except Exception as e:
            logging.warning(f"{sub_id}: GMM failed ({e}); skipping")
            continue

        aga_ok.append(sub_id)
        count = 0
        idx_starts = []

        # Scan windows of length 100
        if len(aga) >= 100:
            for j in range(0, len(aga) - 100):
                window = aga[j:j + 100]

                # Original selection logic
                if (np.sum(window > np.max(means)) > 10) and \
                   (np.sum(window < np.min(means)) > 10):

                    # Ensure windows are separated by at least 100 frames
                    if not idx_starts or (j - idx_starts[-1] > 100):
                        idx_starts.append(j)
                        aga_ok.append(j + t0)

                        # Slice original data aligning with window
                        # (for multi-angle columns)
                        data_slice = data.iloc[j + t0:j + t0 + 100].copy()

                        feat = feature_extract(
                            name=f"{sub_id}_trial{count}",
                            data=data_slice,
                            t0=t0,
                            video_path=video_path,
                            target=args.target,
                            images_outdir=images_outdir
                        )
                        features.append(feat)
                        count += 1
        else:
            logging.info(f"{sub_id}: series too short (<100)")

    # Build output DataFrame
    cols = [
        'Name', 'Maximum Angle', 'Minimum Angle', 'Correlation',
        'Granger Causality', 'Minimum Std', 'Diff Std',
        'Velocity Correlation', 'Velocity Granger Causality',
        'Velocity Minimum Std', 'Velocity Diff Std', 'Means Diff',
        'Bimodal Std', 'Class'
    ]

    # Append Path column if any image path was produced
    has_paths = any(len(row) == len(cols) + 1 for row in features)
    if has_paths:
        cols.append('Path')

    df = pd.DataFrame(features, columns=cols)

    # Save features
    out_xlsx = os.path.join(args.outdir, "features.xlsx")
    df.to_excel(out_xlsx, index=False)
    logging.info(f"Saved features to {out_xlsx}")

    # Optional: save exceptional frames list (kept commented as in original)
    # create_exc_frame(
    #     aga_ok,
    #     out_path=os.path.join(args.outdir, f"frames_{args.split_name}.xlsx")
    # )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        sys.exit(1)
