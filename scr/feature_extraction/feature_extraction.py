import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import cv2

from scipy.stats import kendalltau
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture

from src.utils import ensure_dir, extract_frame


def ggm(time_series: np.ndarray, contamination: float) -> np.ndarray:
    """
    Fit a 2-component Gaussian Mixture model to the time series after removing outliers.

    Parameters
    ----------
    time_series : np.ndarray
        1D array-like time series.
    contamination : float
        Proportion of outliers for EllipticEnvelope.

    Returns
    -------
    np.ndarray
        Array of shape (2,) containing the two component means.
        Returns np.array([np.nan, np.nan]) in degenerate cases.
    """
    ts = np.asarray(time_series).reshape(-1)
    ts_nonan = ts[~np.isnan(ts)]

    # Gestione casi degeneri
    if ts_nonan.size < 2:
        return np.array([np.nan, np.nan])

    ts_nonan = ts_nonan.reshape(-1, 1)

    # Per serie molto corte evito l'outlier detection, che può rompersi
    if ts_nonan.shape[0] < 10:
        data_in = ts_nonan
    else:
        try:
            envelope = EllipticEnvelope(contamination=contamination, random_state=42)
            labels = envelope.fit_predict(ts_nonan)
            inliers = ts_nonan[labels == 1]
            if inliers.size == 0:
                data_in = ts_nonan
            else:
                data_in = inliers
        except Exception:
            # Se qualcosa va storto, uso comunque i dati grezzi
            data_in = ts_nonan

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(data_in)

    # Ritorno un vettore 1D (2,)
    means = gmm.means_.reshape(-1)
    return means


def feature_extract(
    name: str,
    data: pd.DataFrame,
    t0: int,
    video_path: str,
    target: int,
    images_outdir: Optional[str] = None
) -> list:
    """
    Compute statistical, velocity and causality features; optionally save a difference image.

    Parameters
    ----------
    name : str
        Identifier for the sample (used also as image name).
    data : pd.DataFrame
        Dataframe with at least the columns:
        - 'Anterior Glottic Angle'
        - 'Angle of Right Cord'
        - 'Angle of Left Cord'
    t0 : int
        Starting frame index for this segment in the original video.
    video_path : str
        Path to the original video file.
    target : int
        Integer label (class).
    images_outdir : Optional[str]
        If provided, directory in which to save the difference image.

    Returns
    -------
    list
        If images_outdir is None:
            [
                name, max_ang, min_ang, corr, grang, min_std, diff_std,
                vel_corr, vel_grang, vel_min_std, vel_diff_std, means_diff,
                std_bimod, target
            ]
        else:
            [
                ..., target, path_out
            ]
        where path_out is the path of the saved difference image.
    """
    # --- Static features on Anterior Glottic Angle ---
    max_ang = np.nanpercentile(data['Anterior Glottic Angle'], 97)
    min_ang = np.nanpercentile(data['Anterior Glottic Angle'], 3)

    la = np.array(data['Angle of Right Cord'])
    ra = np.array(data['Angle of Left Cord'])
    idx = ~np.isnan(ra) & ~np.isnan(la)

    # Correlazione angoli
    if np.sum(idx) > 1:
        corr, _ = kendalltau(la[idx], ra[idx])
    else:
        corr = np.nan

    # Granger causality sugli angoli
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

    # Varianze angoli
    if np.sum(idx) > 1:
        std_la = np.std(la[idx])
        std_ra = np.std(ra[idx])
        min_std = min(std_la, std_ra)
        diff_std = abs(std_la - std_ra)
    else:
        min_std = np.nan
        diff_std = np.nan

    # --- Velocità ---
    vel_la = la[idx][1:] - la[idx][:-1]
    vel_ra = ra[idx][1:] - ra[idx][:-1]

    if vel_la.size > 1 and vel_ra.size > 1:
        vel_corr, _ = kendalltau(vel_la, vel_ra)
    else:
        vel_corr = np.nan

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

    if vel_la.size > 0 and vel_ra.size > 0:
        vel_min_std = min(np.std(vel_la), np.std(vel_ra))
        vel_diff_std = abs(np.std(vel_la) - np.std(vel_ra))
    else:
        vel_min_std = np.nan
        vel_diff_std = np.nan

    # --- Bimodalità Anterior Glottic Angle ---
    means = ggm(data['Anterior Glottic Angle'], contamination=0.05)
    if np.all(np.isnan(means)):
        means_diff = np.nan
    else:
        means_diff = float(np.nanmax(means) - np.nanmin(means))

    std_bimod = np.nanstd(np.array(data['Anterior Glottic Angle']))

    # --- Difference image ---
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

                # Normalizzazione
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


