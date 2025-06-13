# ----------------------------------------------------------------------------------------------------------------------------------
#  Merrypopins Streamlit App — Nano‑indentation pop‑in Analysis
#  A user‑friendly interface for analyzing nanoindentation data.
#  This app allows users to upload indentation data, preprocess it,
#  detect pop-in events using various methods, and visualize the results and produce statistical calculations from detected popins.
#  It also provides options to download the results in CSV format or as a ZIP file containing the data and visualizations.
#  (2025‑06‑13)
# ----------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

from merrypopins.load_datasets import load_txt, load_tdm
from merrypopins.preprocess import (
    default_preprocess,
    finalise_contact_index,
    remove_pre_min_load,
    rescale_data,
)
from merrypopins.locate import (
    detect_popins_cnn,
    detect_popins_fd_fourier,
    detect_popins_iforest,
    detect_popins_savgol,
)

from merrypopins.statistics import (
    default_statistics,
    calculate_stress_strain,
    calculate_stress_strain_statistics,
    default_statistics_stress_strain,
)

# ───────────────────────────────────────────────────────────────
#  1 ∙ PAGE CONFIG & APP‑LEVEL LOGGING
# ───────────────────────────────────────────────────────────────
PAGE_TITLE = "Merrypopins Nano‑indentation pop‑in Analysis"
APP_VERSION = "2025‑06‑13"
DOC_URL = (
    "https://serprateai.github.io/merrypopins/reference/merrypopins.load_datasets/"
)

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# ───────────────────────────────────────────────────────────────
#  2 ∙ SIDEBAR — global controls & upload persistence helpers
# ───────────────────────────────────────────────────────────────

st.sidebar.title(PAGE_TITLE)
st.sidebar.image("streamlit_app/static/logo-transparent.png", use_container_width=True)
# docs link
st.sidebar.markdown(
    f"📚 For detailed documentation about `merrypopins` library and tuning parameters visit our [home page.]({DOC_URL})"
)

# —— upload and png helper ————————————————————————————
# ── ensure PNG export always uses Kaleido ──────────────────────
pio.kaleido.scope.default_format = "png"  # <-- new
pio.kaleido.scope.default_width = 1000  # optional defaults
pio.kaleido.scope.default_height = 600
pio.kaleido.scope.default_scale = 2


def _fig_to_png(fig) -> bytes:
    """Robust PNG export that always uses Kaleido."""
    return pio.to_image(fig, format="png")  # dimensions come from scope defaults


def persist_file_uploader(label: str, key: str, types: Tuple[str, ...]):
    """Return file‑like BytesIO & name, persisting across reruns."""
    stored_key = f"{key}_stored"
    upload_key = f"{key}_upload"

    uploaded = st.sidebar.file_uploader(label, type=list(types), key=upload_key)
    if uploaded is not None:
        # save bytes immediately so we don’t lose them on the next rerun
        st.session_state[stored_key] = {
            "name": uploaded.name,
            "bytes": uploaded.getvalue(),
        }

    if stored_key in st.session_state:
        data = st.session_state[stored_key]
        return io.BytesIO(data["bytes"]), data["name"]
    return None, None


# —— reset button ————————————————————————————————————


def reset_all():
    for k in list(st.session_state.keys()):
        if not k.startswith("_st_"):
            del st.session_state[k]
    st.rerun()


if st.sidebar.button("🚫 Reset session", help="Clear cached data and start over"):
    reset_all()

# ───────────────────────────────────────────────────────────────
#  3 ∙ UPLOAD SECTION
# ───────────────────────────────────────────────────────────────

txt_io, txt_name = persist_file_uploader(
    "📄 Upload indentation *.txt*", "txt_file", ("txt",)
)
tdm_io, tdm_name = persist_file_uploader(
    "📑 Optional metadata (.tdm/.tdx)", "tdm_file", ("tdm", "tdx")
)

if txt_io is None:
    st.info("⬅️ Upload a .txt file to begin …")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as _tmp:
    _tmp.write(txt_io.read())
    TXT_PATH = Path(_tmp.name)

if tdm_io is not None:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(tdm_name).suffix
    ) as _mtmp:
        _mtmp.write(tdm_io.read())
        TDM_PATH = Path(_mtmp.name)
else:
    TDM_PATH = None

# ───────────────────────────────────────────────────────────────
#  4 ∙ CONFIG DATACLASSES
# ───────────────────────────────────────────────────────────────


@dataclass
class PreprocessConfig:
    remove_pre: bool = True
    rescale: bool = True
    finalise: bool = True
    N_baseline: int = 50
    k_sigma: float = 5.0
    smooth_win: int = 11
    polyorder: int = 2


@dataclass
class TrimConfig:
    trim_edges: bool = True
    trim_margin: int = 30
    max_load_cut: bool = True


if "prep_cfg" not in st.session_state:
    st.session_state["prep_cfg"] = PreprocessConfig()
if "trim_cfg" not in st.session_state:
    st.session_state["trim_cfg"] = TrimConfig()

prep_cfg: PreprocessConfig = st.session_state["prep_cfg"]
trim_cfg: TrimConfig = st.session_state["trim_cfg"]

# ───────────────────────────────────────────────────────────────
#  5 ∙ SIDEBAR CONTROLS (update dataclasses in‑place)
# ───────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Pre‑processing")
prep_cfg.remove_pre = st.sidebar.checkbox(
    "Remove data before min(load)", prep_cfg.remove_pre
)
prep_cfg.rescale = st.sidebar.checkbox("Auto‑rescale depth", prep_cfg.rescale)
prep_cfg.finalise = st.sidebar.checkbox("Trim/flag contact point", prep_cfg.finalise)

with st.sidebar.expander("Advanced thresholds"):
    prep_cfg.N_baseline = st.slider("Baseline points", 10, 200, prep_cfg.N_baseline)
    prep_cfg.k_sigma = st.slider("Noise × k", 1.0, 10.0, prep_cfg.k_sigma)
    prep_cfg.smooth_win = st.slider(
        "Smooth window (odd)", 3, 51, prep_cfg.smooth_win, step=2
    )
    prep_cfg.polyorder = st.slider("Poly‑order", 1, 5, prep_cfg.polyorder)

st.sidebar.header("🪚 Edge trimming")
trim_cfg.trim_edges = st.sidebar.checkbox("Trim first points", trim_cfg.trim_edges)
trim_cfg.trim_margin = st.sidebar.number_input(
    "Trim margin (pts)", 0, 500, trim_cfg.trim_margin
)
trim_cfg.max_load_cut = st.sidebar.checkbox(
    "Ignore after max‑load", trim_cfg.max_load_cut
)

want_zip = st.sidebar.checkbox("Bundle outputs as ZIP", value=True)

# ───────────────────────────────────────────────────────────────
#  6 ∙ PREPROCESSING HELPERS (cached)
# ───────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, hash_funcs={Path: lambda p: p.stat().st_mtime})
def _load_and_preprocess(path: Path, cfg: PreprocessConfig) -> pd.DataFrame:
    df0 = load_txt(path)
    if cfg.remove_pre:
        df0 = remove_pre_min_load(df0, load_col="Load (µN)")
    if cfg.rescale:
        df0 = rescale_data(
            df0,
            depth_col="Depth (nm)",
            load_col="Load (µN)",
            N_baseline=cfg.N_baseline,
            k=cfg.k_sigma,
            window_length=cfg.smooth_win,
            polyorder=cfg.polyorder,
        )
    if cfg.finalise:
        df0 = finalise_contact_index(df0, depth_col="Depth (nm)")
    return df0


# detectors are cached individually so tweaks don’t recompute everything
@st.cache_data(show_spinner=False)
def _iforest(df: pd.DataFrame, cont: float, win: int, cfg: TrimConfig):
    return detect_popins_iforest(
        df,
        contamination=cont,
        window=win,
        trim_edges_enabled=cfg.trim_edges,
        trim_margin=cfg.trim_margin,
        max_load_trim_enabled=cfg.max_load_cut,
    )


@st.cache_data(show_spinner=False)
def _cnn(
    df: pd.DataFrame, epochs: int, ws: int, batch: int, thr_mul: float, cfg: TrimConfig
):
    return detect_popins_cnn(
        df,
        window_size=ws,
        epochs=epochs,
        batch_size=batch,
        threshold_multiplier=thr_mul,
        trim_edges_enabled=cfg.trim_edges,
        trim_margin=cfg.trim_margin,
        max_load_trim_enabled=cfg.max_load_cut,
    )


@st.cache_data(show_spinner=False)
def _fd(df: pd.DataFrame, th: float, sp: float, cfg: TrimConfig):
    return detect_popins_fd_fourier(
        df,
        threshold=th,
        spacing=sp,
        trim_edges_enabled=cfg.trim_edges,
        trim_margin=cfg.trim_margin,
        max_load_trim_enabled=cfg.max_load_cut,
    )


@st.cache_data(show_spinner=False)
def _sg(df: pd.DataFrame, wl: int, po: int, th: float, der: int, cfg: TrimConfig):
    return detect_popins_savgol(
        df,
        window_length=wl,
        polyorder=po,
        threshold=th,
        deriv=der,
        trim_edges_enabled=cfg.trim_edges,
        trim_margin=cfg.trim_margin,
        max_load_trim_enabled=cfg.max_load_cut,
    )


# ───────────────────────────────────────────────────────────────
#  7 ∙ MAIN BODY — run or reuse pipeline
# ───────────────────────────────────────────────────────────────

st.title("📊 Merrypopins: Nano‑indentation Pop‑in Detector")

# —— metadata preview ————————————————————————————
df_raw = load_txt(TXT_PATH)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw data (head)")
    st.dataframe(df_raw.head(), use_container_width=True)
with col2:
    st.subheader("File info")
    st.metric("Points", len(df_raw))
    st.metric("Min load µN", f"{df_raw['Load (µN)'].min():.2f}")
    st.metric("Max load µN", f"{df_raw['Load (µN)'].max():.2f}")
    ts = df_raw.attrs.get("timestamp")
    if ts:
        st.write(f"**Timestamp:** {ts}")
    if df_raw.attrs.get("num_points"):
        st.write(f"**Declared points:** {df_raw.attrs['num_points']}")

if TDM_PATH is not None:
    df_root, df_channels = load_tdm(TDM_PATH)
    st.subheader("Metadata (.tdm) preview")
    st.json(df_root.iloc[0].to_dict())
    st.dataframe(df_channels, use_container_width=True)

# —— preprocessing ———————————————————————————————
st.subheader("🔧 Pre‑processing")
mode = st.radio(
    "Mode",
    ["Default", "Custom", "None"],
    index=1 if prep_cfg.remove_pre else 0,
    horizontal=True,
)

if mode == "Default":
    df_pre = default_preprocess(df_raw)
elif mode == "Custom":
    df_pre = _load_and_preprocess(TXT_PATH, prep_cfg)
else:
    df_pre = df_raw.copy()

fig_raw_pre = px.line(
    df_raw,
    x="Depth (nm)",
    y="Load (µN)",
    labels=None,
    title="Raw vs. Pre‑processed",
    color_discrete_sequence=["blue"],
)
# Explicitly set the raw data trace color
fig_raw_pre.update_traces(line=dict(color="blue"))

fig_raw_pre.add_scatter(
    x=df_pre["Depth (nm)"],
    y=df_pre["Load (µN)"],
    mode="lines",
    name="Pre‑processed",
    line=dict(color="orange"),
)

# Add contact point markers
if "contact_point" in df_pre.columns and df_pre["contact_point"].any():
    cp = df_pre[df_pre["contact_point"]]
    fig_raw_pre.add_scatter(
        x=cp["Depth (nm)"],
        y=cp["Load (µN)"],
        mode="markers",
        marker_symbol="circle-open",
        marker_size=12,
        name="Contact",
        marker=dict(color="red"),
    )

fig_raw_pre.update_layout(showlegend=True)
st.plotly_chart(fig_raw_pre, use_container_width=True)

# --- NEW: download raw-vs-preprocessed plot -------------------
raw_png = _fig_to_png(fig_raw_pre)
st.download_button(
    "🖼️ Download PNG (raw vs pre-processed)",
    data=raw_png,
    file_name="raw_vs_preprocessed.png",
    mime="image/png",
)

# cache preprocessed df for detectors
st.session_state["df_pre"] = df_pre

# ───────────────────────────────────────────────────────────────
#  8 ∙ DETECTOR CONTROLS
# ───────────────────────────────────────────────────────────────

st.subheader("⚙️ Detection Parameters")

col_if, col_cnn, col_fd, col_sg = st.columns(4)

with col_if:
    st.markdown("**Isolation Forest**")
    if_enable_if = st.checkbox("Enable IF", True)
    if_cont = st.slider("Contamination", 0.001, 0.3, 0.01, 0.001, key="if_cont_slider")
    if_win = st.slider("Stiffness window", 3, 30, 10, key="if_win_slider")

with col_cnn:
    st.markdown("**CNN Autoencoder**")
    if_enable_cnn = st.checkbox("Enable CNN", True)
    cnn_ep = st.slider("Epochs", 1, 50, 10, key="cnn_ep_slider")
    cnn_ws = st.slider("Window size", 16, 128, 64, step=8, key="cnn_ws_slider")
    cnn_bs = st.slider("Batch size", 8, 128, 32, step=8, key="cnn_bs_slider")
    cnn_tm = st.slider("Threshold ×σ", 1.0, 10.0, 5.0, key="cnn_tm_slider")

with col_fd:
    st.markdown("**Fourier derivative**")
    if_enable_fd = st.checkbox("Enable FD", True)
    fd_th = st.slider("Threshold σ", 1.0, 5.0, 3.0, key="fd_th_slider")
    fd_sp = st.slider("Spacing", 0.1, 5.0, 1.0, 0.1, key="fd_sp_slider")

with col_sg:
    st.markdown("**Savitzky‑Golay**")
    if_enable_sg = st.checkbox("Enable SG", True)
    sg_wl = st.slider("Window length", 3, 21, 11, step=2, key="sg_wl_slider")
    sg_po = st.slider("Poly order", 1, 5, 2, key="sg_po_slider")
    sg_th = st.slider("Threshold σ", 1.0, 5.0, 3.0, key="sg_th_slider")
    sg_der = st.slider("Derivative order", 1, 3, 1, key="sg_der_slider")

run_clicked = st.button("🚀 Run selected detectors")

if run_clicked or "detectors_done" not in st.session_state:

    st.session_state["detectors_done"] = True  # flag so downloads survive reruns

    with st.spinner("Running detectors …"):
        method_frames: Dict[str, pd.DataFrame] = {}

        if if_enable_if:
            method_frames["popin_iforest"] = _iforest(df_pre, if_cont, if_win, trim_cfg)
        if if_enable_cnn:
            method_frames["popin_cnn"] = _cnn(
                df_pre, cnn_ep, cnn_ws, cnn_bs, cnn_tm, trim_cfg
            )
        if if_enable_fd:
            method_frames["popin_fd"] = _fd(df_pre, fd_th, fd_sp, trim_cfg)
        if if_enable_sg:
            method_frames["popin_savgol"] = _sg(
                df_pre, sg_wl, sg_po, sg_th, sg_der, trim_cfg
            )

        # merge all boolean columns onto a shared copy
        df_det = df_pre.copy()
        for col, f in method_frames.items():
            df_det[col] = f[col]
        flag_cols = list(method_frames.keys())
        if flag_cols:
            df_det["popin"] = df_det[flag_cols].any(axis=1)
            df_det["popin_score"] = df_det[flag_cols].sum(axis=1)
            df_det["popin_confident"] = df_det["popin_score"] >= 2

        st.session_state["df_det"] = df_det

# ───────────────────────────────────────────────────────────────
#  9 ∙ RESULTS VISUALISATION & DOWNLOAD
# ───────────────────────────────────────────────────────────────

df_det: pd.DataFrame = st.session_state.get("df_det")
if df_det is not None:
    st.subheader("📊 Detections summary")
    cols = st.columns(6)
    cols[0].metric("Total pop‑ins", int(df_det["popin"].sum()))
    cols[1].metric(
        "Confident ≥2", int(df_det.get("popin_confident", pd.Series()).sum())
    )
    for i, m in enumerate(
        ["popin_iforest", "popin_cnn", "popin_fd", "popin_savgol"], start=2
    ):
        if m in df_det:
            cols[i].metric(f"{m.split('_')[1].upper()} hits", int(df_det[m].sum()))

    fig = px.line(
        df_det, x="Depth (nm)", y="Load (µN)", title="Merrypopins Popin Detections"
    )
    color_map = {
        "popin_iforest": "red",
        "popin_cnn": "purple",
        "popin_fd": "orange",
        "popin_savgol": "green",
        "popin_confident": "black",
    }
    for m, c in color_map.items():
        if m in df_det and df_det[m].any():
            fig.add_scatter(
                x=df_det[df_det[m]]["Depth (nm)"],
                y=df_det[df_det[m]]["Load (µN)"],
                mode="markers",
                marker=dict(color=c),
                name=m.replace("popin_", "").capitalize(),
            )
    st.plotly_chart(fig, use_container_width=True)

    # —— download buttons ————————————————————————————
    csv_bytes = df_det.to_csv(index=False).encode()
    png_bytes = _fig_to_png(fig)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 CSV results",
            data=csv_bytes,
            file_name="merrypopins_annotated.csv",
            mime="text/csv",
        )
        st.download_button(
            "🖼️ Download PNG (Merrypopings Popin Detections Plot)",
            data=png_bytes,
            file_name="detections_plot.png",
            mime="image/png",
        )

    with col_dl2:
        if want_zip:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("merrypopins_annotated.csv", csv_bytes)
                # save figure as html so no kaleido dependency
                zf.writestr("detections_plot.png", png_bytes)
                zf.writestr(
                    "session_config.json",
                    json.dumps(
                        {
                            "preprocess": asdict(prep_cfg),
                            "trim": asdict(trim_cfg),
                        },
                        indent=2,
                    ),
                )
            st.download_button(
                "📦 ZIP (CSV + plot + config)",
                data=zip_buf.getvalue(),
                file_name="merrypopins_results.zip",
            )

    # ───────────────────────────────────────────────────────────────
    # 10 ∙ COMPUTE STATISTICS
    # ───────────────────────────────────────────────────────────────
    # -----------------------------------------------------------
    # 4)  DOWNLOAD BUTTONS
    # -----------------------------------------------------------
    # ––– helper to create & place a pair of buttons
    def _dl_pair(col_csv, col_plot, df, fig, stem):
        csv_bytes = df.to_csv(index=False).encode()
        png_bytes = _fig_to_png(fig)
        with col_csv:
            st.download_button(
                f"📥 Download CSV ({stem})",
                data=csv_bytes,
                file_name=f"{stem}.csv",
                mime="text/csv",
            )
        with col_plot:
            st.download_button(
                f"📥 Download Plot ({stem})",
                data=png_bytes,
                file_name=f"{stem}.png",
                mime="image/png",
            )


st.subheader("📊 Compute Pop-in Statistics")

if st.session_state.get("df_det") is not None:
    # -----------------------------------------------------------
    # 1)  LOAD–DEPTH  statistics
    # -----------------------------------------------------------
    st.markdown("### Load-Depth Pop-in Statistics")
    df_statistics = default_statistics(st.session_state["df_det"])

    st.write("#### Computed Pop-in Statistics:")
    st.dataframe(df_statistics[["start_idx", "end_idx", "depth_jump", "popin_length"]])

    # Depth-jump vs pop-in-length
    fig_statistics = px.scatter(
        df_statistics,
        x="depth_jump",
        y="popin_length",
        title="Depth Jump vs Pop-in Length",
        labels={
            "depth_jump": "Depth Jump (nm)",
            "popin_length": "Pop-in Length (s)",
        },
    )
    st.plotly_chart(fig_statistics, use_container_width=True)
    col_dl1, col_dl2 = st.columns(2)
    _dl_pair(
        col_dl1, col_dl2, df_statistics, fig_statistics, "popin_statistics_load_depth"
    )

    # -----------------------------------------------------------
    # 2)  STRESS–STRAIN  statistics
    # -----------------------------------------------------------
    st.markdown("### Stress–Strain Pop-in Statistics")

    df_stress_strain = calculate_stress_strain(df_statistics)
    df_stress_strain_stats = calculate_stress_strain_statistics(df_stress_strain)

    st.write("#### Computed Stress–Strain Statistics:")
    st.dataframe(
        df_stress_strain_stats[
            ["stress_jump", "strain_jump", "stress_slope", "strain_slope"]
        ]
    )

    # Strain-jump (x) vs stress-jump (y)
    fig_stress_strain = px.scatter(
        df_stress_strain_stats,
        x="strain_jump",
        y="stress_jump",
        title="Stress Jump vs Strain Jump",
        labels={
            "strain_jump": "Strain Jump (–)",
            "stress_jump": "Stress Jump (MPa)",
        },
    )
    st.plotly_chart(fig_stress_strain, use_container_width=True)
    col_dl3, col_dl4 = st.columns(2)
    _dl_pair(
        col_dl3,
        col_dl4,
        df_stress_strain_stats,
        fig_stress_strain,
        "popin_statistics_stress_strain",
    )

    # -----------------------------------------------------------
    # 3)  FULL pipeline (stress–strain time-series)
    # -----------------------------------------------------------
    st.markdown("### Full Stress–Strain Statistics Pipeline")

    df_stats_ss_full = default_statistics_stress_strain(
        st.session_state["df_det"],
        popin_flag_column="popin",
        before_window=0.5,
        after_window=0.5,
        Reff_um=5.323,
        min_load_uN=2000,
        smooth_stress=True,
        stress_col="stress",
        strain_col="strain",
        time_col="Time (s)",
    )

    st.write("#### Full Stress–Strain Statistics:")
    st.dataframe(df_stats_ss_full[["stress", "strain", "stress_slope", "strain_slope"]])

    # Strain (x) vs stress (y)
    fig_full_statistics = px.scatter(
        df_stats_ss_full,
        x="strain",
        y="stress",
        title="Stress vs Strain",
        labels={
            "strain": "Strain (–)",
            "stress": "Stress (MPa)",
        },
    )
    st.plotly_chart(fig_full_statistics, use_container_width=True)
    col_dl5, col_dl6 = st.columns(2)
    _dl_pair(
        col_dl5,
        col_dl6,
        df_stats_ss_full,
        fig_full_statistics,
        "popin_statistics_full_stress_strain",
    )


# ───────────────────────────────────────────────────────────────
# 11 ∙ FOOTER
# ───────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.write(f"**App version:** {APP_VERSION}")
st.sidebar.markdown("Made with ❤️ by the Merrypopins team")
