import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from merrypopins.preprocess import default_preprocess
from merrypopins.make_dataset import merrypopins_pipeline
from merrypopins.locate import (
    detect_popins_iforest,
    detect_popins_cnn,
    detect_popins_fd_fourier,
    detect_popins_savgol,
)
from merrypopins.load_datasets import load_txt

st.set_page_config(page_title="Merrypopins Analyzer", layout="wide")

st.title("📊 Merrypopins: Nanoindentation Pop-In Detector")

st.markdown(
    """Analyze nanoindentation curves with multiple pop-in detection algorithms:
- **Isolation Forest**
- **CNN Autoencoder**
- **Fourier Derivative**
- **Savitzky-Golay Filter**
"""
)

uploaded_file = st.file_uploader("Upload indentation .txt file", type=["txt"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    df = load_txt(tmp_path)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("🔧 Preprocessing")
    if st.checkbox("Run Default Preprocessing", value=True):
        df_pre = default_preprocess(df)

        st.subheader("Raw vs Preprocessed Plot")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Depth (nm)"], df["Load (µN)"], "--", alpha=0.5, label="Raw")
        ax.plot(df_pre["Depth (nm)"], df_pre["Load (µN)"], "-", label="Preprocessed")
        ax.set_xlabel("Depth (nm)")
        ax.set_ylabel("Load (µN)")
        ax.set_title("Raw vs Preprocessed")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        df = df_pre

    with st.expander("⚙️ Pipeline Parameters"):
        use_iforest = st.checkbox("Use Isolation Forest", value=True)
        use_cnn = st.checkbox("Use CNN Autoencoder", value=True)
        use_fd = st.checkbox("Use Fourier Derivative", value=True)
        use_savgol = st.checkbox("Use Savitzky-Golay", value=True)

        iforest_contamination = st.slider("IF Contamination", 0.001, 0.3, 0.01)
        iforest_window = st.slider("IF Window", 3, 30, 10)

        cnn_epochs = st.slider("CNN Epochs", 1, 50, 10)
        cnn_window_size = st.slider("CNN Window Size", 16, 128, 64, step=8)

        fd_threshold = st.slider("FD Threshold (std dev)", 1.0, 5.0, 3.0)
        savgol_window_len = st.slider("SG Window Length", 3, 21, 9, step=2)
        savgol_poly = st.slider("SG Polynomial Order", 1, 5, 2)

    if st.button("Run Pipeline 🚀"):
        with st.spinner("Running Merrypopins pipeline... this may take a moment ⏳"):
            status_text = st.empty()
            status_text.text("Processing...")

            pipeline_df = merrypopins_pipeline(
                txt_path=tmp_path,
                iforest_contamination=iforest_contamination,
                cnn_epochs=cnn_epochs,
                cnn_window_size=cnn_window_size,
                fd_threshold=fd_threshold,
                savgol_window_length=savgol_window_len,
                savgol_polyorder=savgol_poly,
                stiffness_window=iforest_window,
                use_iforest=use_iforest,
                use_cnn=use_cnn,
                use_fd=use_fd,
                use_savgol=use_savgol,
            )

            status_text.text("✅ Pipeline completed!")
            st.success("Pipeline executed successfully!")

            fig = px.line(
                pipeline_df,
                x="Depth (nm)",
                y="Load (µN)",
                title="Detected Pop-ins (Composite)",
            )
            for col in [
                "popin_iforest",
                "popin_cnn",
                "popin_fd",
                "popin_savgol",
                "popin_confident",
            ]:
                if col in pipeline_df:
                    fig.add_scatter(
                        x=pipeline_df[pipeline_df[col]]["Depth (nm)"],
                        y=pipeline_df[pipeline_df[col]]["Load (µN)"],
                        mode="markers",
                        name=col,
                    )
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download Annotated CSV",
                data=pipeline_df.to_csv(index=False),
                file_name="annotated_popins.csv",
            )

    st.subheader("Run Individual Detection Methods")

    with st.expander("🔍 Isolation Forest"):
        contamination = st.slider("Contamination", 0.001, 0.3, 0.03)
        window = st.slider("Stiffness Window", 3, 30, 10)
        if st.button("Run Isolation Forest"):
            with st.spinner("Running Isolation Forest... this may take a moment ⏳"):
                status_text = st.empty()
                status_text.text("Processing...")
                df_iso = detect_popins_iforest(
                    df, contamination=contamination, window=window
                )
                st.plotly_chart(
                    px.scatter(
                        df_iso,
                        x="Depth (nm)",
                        y="Load (µN)",
                        color=df_iso["popin_iforest"].astype(str),
                    )
                )

    with st.expander("🤖 CNN Autoencoder"):
        epochs = st.slider("Epochs", 1, 50, 10)
        window_size = st.slider("Window Size", 16, 128, 64, step=8)
        if st.button("Run CNN Autoencoder"):
            with st.spinner("Running CNN Autoencoder... this may take a moment ⏳"):
                status_text = st.empty()
                status_text.text("Processing...")
                df_cnn = detect_popins_cnn(df, epochs=epochs, window_size=window_size)
                st.plotly_chart(
                    px.scatter(
                        df_cnn,
                        x="Depth (nm)",
                        y="Load (µN)",
                        color=df_cnn["popin_cnn"].astype(str),
                    )
                )

    with st.expander("📈 Fourier Derivative"):
        threshold = st.slider("Threshold (std dev)", 1.0, 5.0, 3.0)
        if st.button("Run Fourier Derivative"):
            with st.spinner("Running Fourier Derivative... this may take a moment ⏳"):
                status_text = st.empty()
                status_text.text("Processing...")
                df_fd = detect_popins_fd_fourier(df, threshold=threshold)
                st.plotly_chart(
                    px.scatter(
                        df_fd,
                        x="Depth (nm)",
                        y="Load (µN)",
                        color=df_fd["popin_fd"].astype(str),
                    )
                )

    with st.expander("🧮 Savitzky-Golay"):
        window_len = st.slider("Window Length", 3, 21, 9, step=2)
        poly = st.slider("Polynomial Order", 1, 5, 2)
        if st.button("Run Savitzky-Golay"):
            with st.spinner("Running Savitzky-Golay... this may take a moment ⏳"):
                status_text = st.empty()
                status_text.text("Processing...")
                df_sg = detect_popins_savgol(
                    df, window_length=window_len, polyorder=poly
                )
                st.plotly_chart(
                    px.scatter(
                        df_sg,
                        x="Depth (nm)",
                        y="Load (µN)",
                        color=df_sg["popin_savgol"].astype(str),
                    )
                )

st.sidebar.title("About Merrypopins")
st.sidebar.markdown(
    """
Merrypopins is an open-source nanoindentation analysis toolkit for robust pop-in detection in material science datasets.

🔗 [GitHub](https://github.com/CAcarSci/merrypopins)
"""
)
