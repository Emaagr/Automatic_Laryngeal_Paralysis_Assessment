import os
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from moviepy.editor import VideoFileClip
from src.utils import ensure_dir, filter_outliers, extract_frame


def ggm(time_series: np.ndarray, contamination: float) -> np.ndarray:
    """Fit a 2-component Gaussian Mixture model to the time series."""
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
    """Compute statistical, velocity, and causality features; optionally save a difference image."""
    max_ang = np.nanpercentile(data['Anterior Glottic Angle'], 97)
    min_ang = np.nanpercentile(data['Anterior Glottic Angle'], 3)

    la = np.array(data['Angle of Right Cord'])
    ra = np.array(data['Angle of Left Cord'])
    idx = ~np.isnan(ra) & ~np.isnan(la)

    corr, _ = kendalltau(la[idx], ra[idx])

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

    min_std = min(np.std(la[idx]), np.std(ra[idx]))
    diff_std = abs(np.std(la[idx]) - np.std(ra[idx]))

    vel_la = la[idx][1:] - la[idx][:-1]
    vel_ra = ra[idx][1:] - ra[idx][:-1]
    vel_corr, _ = kendalltau(vel_la, vel_ra)

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

    means = ggm(data['Anterior Glottic Angle'], contamination=0.05)
    means_diff = max(means) - min(means)
    means_diff = float(np.array(means_diff).reshape(-1)[0]) 
    std_bimod = np.nanstd(np.array(data['Anterior Glottic Angle']))

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

                g1 = (g1 - np.mean(g1)) / (np.std(g1) + 1e-8)
                g2 = (g2 - np.mean(g2)) / (np.std(g2) + 1e-8)

                diff = np.abs(g1 - g2)
                thr = np.median(diff)
                diff = diff * (diff > thr)

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

