import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# ---------------- Defaults ----------------
DEFAULT_SR = 16000
DEFAULT_N_MFCC = 20
DEFAULT_FRAME_MS = 25
DEFAULT_HOP_MS = 10
DEFAULT_STRIDE = 0.10
DEFAULT_MIN_SEP = 0.35
DEFAULT_THRESHOLD = 0.60
DEFAULT_TOL = 0.30        # tolerance for GT matching (s)

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)

# ---------------- Audio / DSP helpers ----------------
def load_audio(file, target_sr=DEFAULT_SR):
    """Load audio as mono float32 at target_sr (WAV/FLAC/OGG preferred)."""
    y, sr = librosa.load(file, sr=target_sr, mono=True)
    # mild RMS normalization for stability (prevents wildly different levels)
    rms = np.sqrt(np.mean(y**2) + 1e-10)
    if rms > 0:
        y = np.clip(y / (rms * 10.0), -1.0, 1.0)
    return y.astype(np.float32), sr

def ms_to_samples(ms, sr):
    return int(round((ms/1000.0) * sr))

def compute_mfcc(y, sr, n_mfcc, frame_ms, hop_ms):
    n_fft = 1 << (ms_to_samples(frame_ms, sr) - 1).bit_length()  # next pow2
    hop_length = ms_to_samples(hop_ms, sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann'))**2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=128, power=1.0)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel + 1e-10), n_mfcc=n_mfcc)
    return mfcc, hop_length

# ---------------- Preprocessing: filters ----------------
def butter_filter(y, sr, mode="Band-pass", low_hz=100.0, high_hz=4000.0, order=4):
    """Apply zero-phase Butterworth filtering with filtfilt."""
    nyq = sr / 2.0
    # clamp cutoffs into valid range
    low = max(1.0, min(low_hz, nyq - 10.0)) / nyq
    high = max(1.0, min(high_hz, nyq - 1.0)) / nyq
    if mode == "High-pass":
        b, a = butter(order, low, btype='highpass')
    elif mode == "Low-pass":
        b, a = butter(order, high, btype='lowpass')
    elif mode == "Band-pass":
        if low >= high:
            # fallback to a narrow band if misconfigured
            w1 = 100.0 / nyq
            w2 = 3000.0 / nyq
            b, a = butter(order, [w1, w2], btype='bandpass')
        else:
            b, a = butter(order, [low, high], btype='bandpass')
    else:
        return y
    # filtfilt avoids phase distortion
    return filtfilt(b, a, y).astype(np.float32)

# ---------------- Preprocessing: noise reduction ----------------
def noise_reduce(y, sr, frame_ms, hop_ms, ref_duration_s=0.5, strength=0.8):
    """
    Simple spectral-subtraction noise reduction:
    - Estimate noise magnitude spectrum from first ref_duration_s of audio.
    - Subtract (strength * noise_mag) from each frame's magnitude.
    - Reconstruct with original phase.
    """
    n_fft = 1 << (ms_to_samples(frame_ms, sr) - 1).bit_length()
    hop_length = ms_to_samples(hop_ms, sr)

    Z = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    S = np.abs(Z)

    # Determine noise segment (from head of clip)
    ref_samples = int(round(ref_duration_s * sr))
    ref_samples = max(hop_length, min(ref_samples, len(y)))
    if ref_samples < hop_length * 2:
        # Fallback: use per-bin 10th percentile if the head is too short
        noise_mag = np.percentile(S, 10, axis=1, keepdims=True)
    else:
        Zref = librosa.stft(y[:ref_samples], n_fft=n_fft, hop_length=hop_length, window='hann')
        noise_mag = np.maximum(np.mean(np.abs(Zref), axis=1, keepdims=True), 1e-6)

    # Spectral subtraction with flooring to avoid musical noise
    S_clean = np.maximum(S - strength * noise_mag, 0.1 * noise_mag)
    Z_clean = S_clean * np.exp(1j * np.angle(Z))
    y_clean = librosa.istft(Z_clean, hop_length=hop_length, window='hann', length=len(y))

    # Preserve rough loudness
    rms_in  = np.sqrt(np.mean(y**2) + 1e-10)
    rms_out = np.sqrt(np.mean(y_clean**2) + 1e-10)
    if rms_out > 0:
        y_clean = y_clean * (rms_in / rms_out)
    return np.clip(y_clean, -1.0, 1.0).astype(np.float32)

# ---------------- DTW path cost ----------------
def dtw_cost(template_mfcc, window_mfcc, metric="cosine", band_ratio=0.2):
    m, n = template_mfcc.shape[1], window_mfcc.shape[1]
    band = int(max(1, band_ratio * max(m, n)))
    D, wp = librosa.sequence.dtw(
        X=template_mfcc, Y=window_mfcc,
        metric=metric, subseq=False, backtrack=True, band_rad=band
    )
    acc_cost = D[-1, -1]
    path_len = max(1, len(wp))
    return float(acc_cost) / float(path_len)

# ---------------- Sliding scans (DTW & Correlation) ----------------
def zscore(x, axis=None, eps=1e-8):
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)

def cosine_similarity(a, b, eps=1e-8):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)

def flatten_mfcc(mfcc):
    return mfcc.T.reshape(-1)

def sliding_dtw_scan(y_templ, y_target, sr, n_mfcc, frame_ms, hop_ms, stride_s):
    T = len(y_templ)
    step = int(round(stride_s * sr))
    if T <= 0 or len(y_target) < T:
        return np.array([]), np.array([]), np.array([]), T, step

    templ_mfcc, _ = compute_mfcc(y_templ, sr, n_mfcc, frame_ms, hop_ms)

    centers_sec, costs = [], []
    for start in range(0, len(y_target) - T + 1, step):
        win = y_target[start:start+T]
        win_mfcc, _ = compute_mfcc(win, sr, n_mfcc, frame_ms, hop_ms)
        c = dtw_cost(templ_mfcc, win_mfcc)
        centers_sec.append((start + T//2) / sr)
        costs.append(c)

    costs = np.array(costs, dtype=np.float32)
    cmin, cmax = np.percentile(costs, 5), np.percentile(costs, 95)
    scale = max(1e-6, (cmax - cmin))
    norm_costs = (costs - cmin) / scale
    similarity = 1.0 / (1.0 + norm_costs.clip(0, None))  # higher is better
    return np.array(centers_sec), costs, similarity, T, step

def sliding_corr_scan(y_templ, y_target, sr, n_mfcc, frame_ms, hop_ms, stride_s):
    T = len(y_templ)
    step = int(round(stride_s * sr))
    if T <= 0 or len(y_target) < T:
        return np.array([]), np.array([]), np.array([]), T, step

    templ_mfcc, _ = compute_mfcc(y_templ, sr, n_mfcc, frame_ms, hop_ms)
    templ_vec = flatten_mfcc(zscore(templ_mfcc))

    centers_sec, sims = [], []
    for start in range(0, len(y_target) - T + 1, step):
        win = y_target[start:start+T]
        win_mfcc, _ = compute_mfcc(win, sr, n_mfcc, frame_ms, hop_ms)
        win_vec = flatten_mfcc(zscore(win_mfcc))
        sim = cosine_similarity(templ_vec, win_vec)
        sim01 = 0.5 * (sim + 1.0)  # [-1,1] -> [0,1]
        centers_sec.append((start + T//2) / sr)
        sims.append(sim01)

    similarity = np.array(sims, dtype=np.float32)
    costs = 1.0 - similarity
    return np.array(centers_sec), costs, similarity, T, step

# ---------------- Post-processing ----------------
def non_max_suppression(times, scores, min_sep_s):
    keep_times, keep_scores = [], []
    if len(times) == 0:
        return np.array([]), np.array([])
    idxs = np.argsort(-scores)  # by score desc
    taken = np.zeros_like(times, dtype=bool)
    for i in idxs:
        if taken[i]:
            continue
        t = times[i]
        keep_times.append(t)
        keep_scores.append(scores[i])
        taken |= (np.abs(times - t) < (min_sep_s / 2.0))
    order = np.argsort(keep_times)
    return np.array(keep_times)[order], np.array(keep_scores)[order]

def estimate_intervals(centers, templ_len_s):
    starts = centers - 0.5 * templ_len_s
    ends   = centers + 0.5 * templ_len_s
    return starts.clip(min=0), centers, ends

# ---------------- Metrics vs Ground Truth ----------------
def parse_ground_truth(df_gt):
    cols = [c.lower().strip() for c in df_gt.columns]
    colmap = {c.lower().strip(): c for c in df_gt.columns}

    centers = None
    if "center (s)" in cols:
        centers = df_gt[colmap["center (s)"]].astype(float).values
    elif "start (s)" in cols and "end (s)" in cols:
        starts = df_gt[colmap["start (s)"]].astype(float).values
        ends   = df_gt[colmap["end (s)"]].astype(float).values
        centers = (starts + ends) / 2.0

    if centers is None:
        raise ValueError("Ground truth must have 'Center (s)' or both 'Start (s)' and 'End (s)'.")
    return np.array(centers, dtype=float)

def match_detections_to_gt(dets, gts, tol_s):
    dets = np.sort(dets)
    gts  = np.sort(gts)
    used_det = np.zeros(len(dets), dtype=bool)
    used_gt  = np.zeros(len(gts), dtype=bool)
    matches = []
    for j, t in enumerate(gts):
        best_i = -1
        best_dt = 1e9
        for i, d in enumerate(dets):
            if used_det[i]:
                continue
            dt = abs(d - t)
            if dt <= tol_s and dt < best_dt:
                best_dt = dt
                best_i = i
        if best_i >= 0:
            used_det[best_i] = True
            used_gt[j] = True
            matches.append((dets[best_i], t, best_dt))
    TP = int(used_gt.sum())
    FP = int((~used_det).sum())
    FN = int((~used_gt).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2*precision*recall / (precision + recall) if (precision+recall) > 0 else 0.0
    return TP, FP, FN, precision, recall, f1, matches

# ---------------- UI ----------------
st.set_page_config(page_title="MFCC + DTW / Correlation Detector", layout="wide")
st.title("🔎 Repeated Phrase Detector: MFCC + DTW / Correlation (with Filters & Noise Reduction)")

with st.sidebar:
    st.header("Parameters")
    method = st.selectbox("Method", ["DTW (robust)", "Correlation (baseline)"])
    sr = st.number_input("Target Sample Rate (Hz)", min_value=8000, max_value=48000, step=1000, value=DEFAULT_SR)
    n_mfcc = st.slider("Number of MFCCs", 8, 40, value=DEFAULT_N_MFCC, step=1)
    frame_ms = st.slider("Frame length (ms)", 10, 40, value=DEFAULT_FRAME_MS, step=1)
    hop_ms = st.slider("Hop length (ms)", 5, 20, value=DEFAULT_HOP_MS, step=1)
    stride_s = st.slider("Detection stride (s)", 0.02, 0.50, value=DEFAULT_STRIDE, step=0.01)
    min_sep_s = st.slider("Min separation between detections (s)", 0.10, 1.50, value=DEFAULT_MIN_SEP, step=0.05)
    threshold = st.slider("Similarity threshold (0–1, higher = stricter)", 0.10, 0.95, value=DEFAULT_THRESHOLD, step=0.01)
    st.caption("Raise threshold to reduce false positives; increase separation to merge close peaks.")

    st.markdown("---")
    st.header("Preprocessing")
    enable_filter = st.checkbox("Enable filter", value=False)
    if enable_filter:
        f_mode = st.selectbox("Filter type", ["High-pass", "Low-pass", "Band-pass"])
        f_order = st.slider("Filter order", 2, 8, value=4, step=1)
        colf1, colf2 = st.columns(2)
        if f_mode == "High-pass":
            with colf1:
                f_low = st.number_input("High-pass cutoff (Hz)", min_value=20.0, max_value=float(sr/2 - 50), value=100.0, step=10.0)
            f_high = None
        elif f_mode == "Low-pass":
            with colf1:
                f_high = st.number_input("Low-pass cutoff (Hz)", min_value=200.0, max_value=float(sr/2 - 10), value=float(sr/2 - 100), step=50.0)
            f_low = None
        else:
            with colf1:
                f_low = st.number_input("Band-pass low (Hz)", min_value=20.0, max_value=float(sr/2 - 100), value=100.0, step=10.0)
            with colf2:
                f_high = st.number_input("Band-pass high (Hz)", min_value=200.0, max_value=float(sr/2 - 10), value=4000.0, step=50.0)
    enable_nr = st.checkbox("Noise reduction (spectral subtraction)", value=False)
    if enable_nr:
        nr_ref = st.slider("Noise profile duration at start (s)", 0.10, 3.00, value=0.50, step=0.10)
        nr_strength = st.slider("Reduction strength", 0.40, 1.50, value=0.80, step=0.05)
        st.caption("Tip: 0.6–0.9 works well. If artifacts appear, reduce strength or increase frame length.")

    st.markdown("---")
    st.subheader("Ground Truth (optional)")
    tol_s = st.slider("Matching tolerance (s)", 0.05, 1.00, value=DEFAULT_TOL, step=0.05)
    gt_file = st.file_uploader("Upload Ground Truth CSV", type=["csv"], key="gt")

colL, colR = st.columns(2)
with colL:
    st.subheader("🎙️ Template Audio (Short Word/Phrase)")
    template_file = st.file_uploader("Upload template (WAV/FLAC/OGG)", type=["wav", "flac", "ogg"], key="templ")
    if template_file:
        y_templ, sr_loaded = load_audio(template_file, target_sr=sr)
        st.audio(template_file, format="audio/wav")
        st.write(f"Duration: {len(y_templ)/sr:.2f} s | SR: {sr}")
with colR:
    st.subheader("🗣️ Target Audio (Long Clip)")
    target_file = st.file_uploader("Upload target (WAV/FLAC/OGG)", type=["wav", "flac", "ogg"], key="target")
    if target_file:
        y_target, sr_loaded2 = load_audio(target_file, target_sr=sr)
        st.audio(target_file, format="audio/wav")
        st.write(f"Duration: {len(y_target)/sr:.2f} s | SR: {sr}")

run = st.button("▶️ Run Detection", type="primary", disabled=not (template_file and target_file))

# ---------------- Plot helpers ----------------
def plot_waveform(y, sr, title):
    fig, ax = plt.subplots(figsize=(8, 2.0), dpi=110)
    t = np.arange(len(y)) / sr
    ax.plot(t, y)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3, linestyle=":")
    st.pyplot(fig); plt.close(fig)

def plot_spectrogram(y, sr, title):
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=110)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96, fmax=sr//2)
    Sdb = librosa.power_to_db(S + 1e-10, ref=np.max)
    img = librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig); plt.close(fig)

def plot_similarity(centers, similarity, peaks_idx, title):
    fig, ax = plt.subplots(figsize=(12, 2.6), dpi=110)
    ax.plot(centers, similarity, label="similarity")
    if len(peaks_idx) > 0:
        ax.scatter(centers[peaks_idx], similarity[peaks_idx], s=40, marker="o", label="peaks")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Similarity (0–1)")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend()
    st.pyplot(fig); plt.close(fig)

# ---------------- Main ----------------
if run and template_file and target_file:
    st.info("Processing… (Increase stride / lower SR if the audio is long and DTW feels heavy.)")
    t0 = time.time()

    # ----- Preprocessing -----
    # Work on copies so the original buffers remain for st.audio playback
    y_t = np.copy(y_templ)
    y_x = np.copy(y_target)

    # Filters
    if enable_filter:
        y_t = butter_filter(y_t, sr, mode=f_mode, low_hz=(f_low or 100.0), high_hz=(f_high or sr/2 - 100), order=f_order)
        y_x = butter_filter(y_x, sr, mode=f_mode, low_hz=(f_low or 100.0), high_hz=(f_high or sr/2 - 100), order=f_order)

    # Noise reduction
    if enable_nr:
        y_t = noise_reduce(y_t, sr, frame_ms, hop_ms, ref_duration_s=nr_ref, strength=nr_strength)
        y_x = noise_reduce(y_x, sr, frame_ms, hop_ms, ref_duration_s=nr_ref, strength=nr_strength)

    # ----- Visuals -----
    st.subheader("📈 Visuals")
    c1, c2 = st.columns(2)
    with c1:
        plot_waveform(y_t, sr, "Template waveform")
        plot_spectrogram(y_t, sr, "Template mel-spectrogram")
    with c2:
        plot_waveform(y_x, sr, "Target waveform")
        plot_spectrogram(y_x, sr, "Target mel-spectrogram")

    # ----- Detection -----
    if method.startswith("DTW"):
        centers, costs, similarity, T, step = sliding_dtw_scan(
            y_t, y_x, sr, n_mfcc, frame_ms, hop_ms, stride_s
        )
        method_label = "DTW"
    else:
        centers, costs, similarity, T, step = sliding_corr_scan(
            y_t, y_x, sr, n_mfcc, frame_ms, hop_ms, stride_s
        )
        method_label = "Correlation"

    if len(centers) == 0:
        st.error("Target is shorter than template; choose a longer target or a shorter template.")
        st.stop()

    # Peak picking on similarity
    distance_pts = max(1, int(round(min_sep_s / max(1e-6, (step / sr)))))
    peaks_idx, props = find_peaks(similarity, height=threshold, distance=distance_pts)

    # Extra NMS in seconds
    keep_times, keep_scores = non_max_suppression(centers[peaks_idx], similarity[peaks_idx], min_sep_s)

    # Convert to intervals
    templ_len_s = len(y_t) / sr
    starts, centers_out, ends = estimate_intervals(keep_times, templ_len_s)

    # Results table
    df = pd.DataFrame({
        "#": np.arange(1, len(centers_out) + 1, dtype=int),
        "Start (s)": np.round(starts, 2),
        "Center (s)": np.round(centers_out, 2),
        "End (s)": np.round(ends, 2),
        "Score": np.round(keep_scores, 3),
        "Method": method_label,
        "Filter": f_mode if enable_filter else "None",
        "NoiseRed": "On" if enable_nr else "Off"
    })

    # Similarity plot
    st.subheader(f"🔬 Similarity curve ({method_label})")
    plot_similarity(centers, similarity, peaks_idx, f"{method_label} similarity vs. time")

    # Detections
    st.subheader(f"✅ Detections: {len(df)} occurrence(s)")
    st.dataframe(df, use_container_width=True)

    # CSV export
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download detections as CSV", data=csv_buf.getvalue(),
                       file_name="detections.csv", mime="text/csv")

    # ----- Evaluation vs GT (optional) -----
    if gt_file is not None:
        try:
            df_gt = pd.read_csv(gt_file)
            gt_centers = parse_ground_truth(df_gt)
            TP, FP, FN, P, R, F1, matches = match_detections_to_gt(
                dets=centers_out, gts=gt_centers, tol_s=tol_s
            )

            st.subheader("📏 Evaluation vs Ground Truth")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("TP", TP)
            mcol2.metric("FP", FP)
            mcol3.metric("FN", FN)
            mcol4.metric("Tol (s)", tol_s)

            mcol5, mcol6, mcol7 = st.columns(3)
            mcol5.metric("Precision", f"{P:.3f}")
            mcol6.metric("Recall",    f"{R:.3f}")
            mcol7.metric("F1-score",  f"{F1:.3f}")

            if len(matches) > 0:
                dfm = pd.DataFrame({
                    "Detection Center (s)": [round(d, 3) for d, t, dt in matches],
                    "GT Center (s)":        [round(t, 3) for d, t, dt in matches],
                    "|Δt| (s)":             [round(dt, 3) for d, t, dt in matches],
                })
                st.dataframe(dfm, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute metrics: {e}")

    st.caption(f"Computed in {time.time() - t0:.2f} s | stride={stride_s:.2f}s | template={templ_len_s:.2f}s | SR={sr} Hz")
