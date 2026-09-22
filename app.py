"""
Thesis pipeline demo — single subject, two CSV uploads + demographics in,
VO2max prediction (window-by-window + global) out.

Configuration: 60-second Stage 2 windows, trained WITHOUT a stability
filter on the training windows, so no per-window stability threshold is
applied here either — the global estimate is the mean of every valid
window's prediction.

Run with:  streamlit run app.py
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import load_clean, synchronize, filtering, windowing, features_5s, met_stage1, features_60s, vo2_stage2

st.set_page_config(page_title="VO₂ max estimation demo", layout="wide")
st.title("VO₂ max estimation demo")
st.caption("Laura Saldaña A. Thesis")
st.caption("Configuration: 60-second windows without stability filtering.")

MODELS_DIR = Path(__file__).parent / "models"
MET_MODEL_PATH = MODELS_DIR / "met_model_full"
VO2_MODEL_PATH = MODELS_DIR / "final_all_subjects_model.json"
VO2_NORM_PATH = MODELS_DIR / "final_all_subjects_normalization.npz"

# ============================================================
# SIDEBAR — demographics
# ============================================================
st.sidebar.header("Demographics")
gender_label = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=25)
height_m = st.sidebar.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.70, format="%.2f")
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=10.0, max_value=250.0, value=70.0, format="%.1f")
fat_pct = st.sidebar.number_input("Fat %", min_value=0.0, max_value=70.0, value=20.0, format="%.1f")
st.sidebar.caption("Entered as a normal percentage (e.g. 20.0 for 20%) — converted to a fraction internally "
                    "to match how it's stored in the training data (SubjectsInfo.xlsx stores it as a fraction, e.g. 0.112).")
hr_rest = st.sidebar.number_input("HR baseline / resting (bpm)", min_value=30, max_value=150, value=65)
spo2_rest = st.sidebar.number_input("SpO2 baseline (%)", min_value=50, max_value=100, value=97)

bmi = weight_kg / (height_m ** 2)
st.sidebar.metric("BMI (auto-calculated)", f"{bmi:.1f}")

st.sidebar.header("Playback")
playback_mode = st.sidebar.radio("Result reveal mode", ["Animated (visual real-time)", "Instant"])
playback_delay = st.sidebar.slider("Seconds per window (animated mode)", 0.05, 1.0, 0.25, 0.05)

# ============================================================
# MAIN — file uploads
# ============================================================
col1, col2 = st.columns(2)
with col1:
    imu_file = st.file_uploader("IMU CSV", type=["csv"])
with col2:
    bio_file = st.file_uploader("Biomarkers CSV (HR + SpO2)", type=["csv"])

run_clicked = st.button("Run pipeline", type="primary", disabled=not (imu_file and bio_file))

# ============================================================
# Model loading — bundled model files, loaded once and cached
# ============================================================
@st.cache_resource(show_spinner=False)
def _load_met_model():
    return met_stage1.load_met_model(MET_MODEL_PATH, device="cpu")


@st.cache_resource(show_spinner=False)
def _load_vo2_model():
    return vo2_stage2.load_vo2_model(str(VO2_MODEL_PATH), str(VO2_NORM_PATH))


# ============================================================
# RUN
# ============================================================
if run_clicked:
    with st.status("Running pipeline...", expanded=True) as status:

        st.write("Loading & cleaning...")
        imu_raw = pd.read_csv(imu_file)
        bio_raw = pd.read_csv(bio_file)
        cleaned = load_clean.load_and_clean(imu_raw, bio_raw, nan_threshold=0.10)
        for w in cleaned.warnings:
            st.warning(w)
        if cleaned.bio_exceeds_threshold:
            st.error("Biomarker data exceeds the 10% NaN threshold on a critical column. "
                      "Proceeding anyway (first-sketch demo behavior), but treat results with caution.")

        st.write("Synchronizing...")
        try:
            imu_sync, bio_sync, sync_info = synchronize.synchronize(cleaned.imu_df, cleaned.bio_df)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        st.caption(f"Shared window: {sync_info['shared_start']} → {sync_info['shared_end']} "
                   f"({sync_info['duration_sec']/60:.1f} min)")

        st.write("Filtering IMU (high-pass -> low-pass -> z-score)...")
        imu_filt = filtering.filter_imu(imu_sync)

        st.write("Segmenting windows (5s / 50% for stage 1, 60s / 50% for stage 2)...")
        windowed = windowing.segment_windows(imu_filt, bio_sync)
        n_5s = len(windowed["imu_5s"]["windows"])
        n_60s = len(windowed["paired_60s"]["imu"])
        st.caption(f"{n_5s} five-second windows, {n_60s} sixty-second windows.")
        if n_60s == 0:
            st.error("No valid 60s windows were produced — the recording may be too short or has gaps. Stopping.")
            st.stop()

        st.write("Extracting 5s IMU intensity features...")
        X5 = features_5s.extract_5s_features(windowed["imu_5s"]["windows"], windowed["imu_5s"]["feature_columns"])

        st.write("Running Stage 1 model (MET regression)...")
        model1, scaler_mean, scaler_scale = _load_met_model()
        met_matrix = met_stage1.run_met_stage1(X5, model1, scaler_mean, scaler_scale)

        st.write("Building Stage 2 feature vectors (60s windows + MET curve + demographics)...")
        gender_val = 1 if gender_label == "Male" else 0
        demo_vector = np.array([gender_val, age, height_m, weight_kg, fat_pct / 100.0, bmi, hr_rest, spo2_rest])
        X60, window_times, window_stability = features_60s.build_feature_vectors(
            windowed["paired_60s"], met_matrix, windowed["imu_5s"]["t_start"], windowed["imu_5s"]["t_end"],
            demo_vector, hr_rest=hr_rest, age=age,
        )
        if len(X60) == 0:
            st.error("No 60s windows had a valid MET alignment — cannot proceed to Stage 2.")
            st.stop()

        st.write("Running Stage 2 model (VO2max regression)...")
        model2, y_mean, y_std, meta = _load_vo2_model()
        if X60.shape[1] != meta["n_features"]:
            st.error(f"Feature vector has {X60.shape[1]} columns but the model expects "
                      f"{meta['n_features']}. Stopping — do not trust downstream output.")
            st.stop()

        preds = vo2_stage2.run_vo2_stage2(X60, model2, y_mean, y_std)
        status.update(label="Pipeline complete — revealing results", state="complete")

    # --------------------------------------------------------
    # RESULTS — playback
    # --------------------------------------------------------
    st.subheader("Results")
    chart_col, metric_col = st.columns([3, 1])
    chart_placeholder = chart_col.empty()
    metric_placeholder = metric_col.empty()
    table_placeholder = st.empty()

    rows = []
    n = len(preds)

    for i in range(n):
        rows.append({
            "window": i + 1,
            "t_start_min": window_times[i][0] / 60.0,
            "VO2max (ml/kg/min)": preds[i],
        })
        df_so_far = pd.DataFrame(rows)

        chart_placeholder.line_chart(df_so_far.set_index("t_start_min")["VO2max (ml/kg/min)"])

        running = vo2_stage2.global_mean_estimate(preds[:i + 1])
        metric_placeholder.metric("Global VO2max estimate", f"{running:.1f} ml/kg/min" if not np.isnan(running) else "—")

        if playback_mode == "Animated (visual real-time)":
            time.sleep(playback_delay)

    table_placeholder.dataframe(df_so_far, use_container_width=True, hide_index=True)

    final_global = vo2_stage2.global_mean_estimate(preds)
    st.success(f"Final global VO2max estimate: **{final_global:.1f} ml/kg/min** (mean of all {n} windows)")

else:
    st.info("Upload the IMU CSV and Biomarkers CSV above, fill in demographics in the sidebar, "
            "and click **Run pipeline** to estimate VO2max.")
