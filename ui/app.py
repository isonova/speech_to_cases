import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# ========== CONFIG (EDIT IF NEEDED) ====================

# Root of your project (speech_to_cases/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default HuggingFace cache folder on Windows
# You can override this from the UI textbox
DEFAULT_HF_CACHE_DIR = r"C:\Users\isode\hf_cache"

# Name of the Docker image you built
DOCKER_IMAGE = "whisper-pipeline"

# Relative path to the pipeline script inside the project root
# If your file is speech_to_cases/Pipeline/pipeline.py, this is correct:
PIPELINE_SCRIPT = "Pipeline/pipeline.py"

# Subfolder where we store uploaded audio files
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Path where pipeline.py writes its final JSON
PIPELINE_OUTPUT_PATH = PROJECT_ROOT / "pipeline_output.json"

# =======================================================


def ensure_uploads_dir() -> None:
    """Create uploads directory if it doesn't exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def run_docker_pipeline(relative_audio_path: str, cache_dir: str) -> Path:
    """
    Run the Docker pipeline for a given audio file.

    relative_audio_path is the path to the audio file
    relative to PROJECT_ROOT (e.g. 'uploads/my_call.wav').
    cache_dir is the HF cache folder on the host (Windows).
    """
    # Remove old output if present
    if PIPELINE_OUTPUT_PATH.exists():
        PIPELINE_OUTPUT_PATH.unlink()

    # Volume mounts for Docker
    project_vol = f"{PROJECT_ROOT}:/app"
    cache_vol = f"{cache_dir}:/root/.cache/huggingface"

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        project_vol,
        "-v",
        cache_vol,
        DOCKER_IMAGE,
        "python",
        PIPELINE_SCRIPT,        # <--- run Pipeline/pipeline.py
        relative_audio_path,
    ]

    # Run the command from the project root so paths line up
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Docker pipeline failed:\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    if not PIPELINE_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"pipeline_output.json not found at {PIPELINE_OUTPUT_PATH}"
        )

    return PIPELINE_OUTPUT_PATH


def load_pipeline_output(path: Path) -> pd.DataFrame:
    """Load pipeline_output.json into a DataFrame."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("pipeline_output.json is not a list of objects.")

    df = pd.DataFrame(data)

    # Ensure expected columns exist
    for col in ["case_index", "text", "summary"]:
        if col not in df.columns:
            df[col] = ""

    # Sort nicely
    if "case_index" in df.columns:
        df = df.sort_values("case_index").reset_index(drop=True)

    return df


def main():
    st.set_page_config(
        page_title="Speech-to-Cases Dashboard",
        layout="wide",
    )

    st.title("📊 Speech-to-Cases Dashboard")
    st.caption(
        "Upload a call recording, run the ML pipeline in Docker, and inspect segments + summaries."
    )

    # -------- SIDEBAR: upload + controls --------
    with st.sidebar:
        st.header("Settings")

        st.markdown("**1. Upload an audio file**")
        uploaded_file = st.file_uploader(
            "Audio file (.wav, .mp3, .m4a, .ogg)",
            type=["wav", "mp3", "m4a", "ogg"],
        )

        st.markdown("**2. HuggingFace cache folder (host)**")
        hf_cache_input = st.text_input(
            "HF cache path (on your Windows machine)",
            value=DEFAULT_HF_CACHE_DIR,
        )

        run_button = st.button("▶ Run pipeline", type="primary")

    # -------- MAIN LOGIC --------
    ensure_uploads_dir()

    if run_button:
        if uploaded_file is None:
            st.error("Please upload an audio file first.")
            return

        cache_dir = hf_cache_input.strip()
        if not cache_dir:
            st.error("Please provide a valid HF cache path.")
            return

        # Save uploaded file to uploads/ inside project root
        audio_dest = UPLOADS_DIR / uploaded_file.name
        with audio_dest.open("wb") as f:
            f.write(uploaded_file.read())

        # Path relative to project root, Docker sees it under /app
        relative_audio_path = audio_dest.relative_to(PROJECT_ROOT).as_posix()

        st.info(f"Saved uploaded audio to {audio_dest}")

        # Run the Docker pipeline
        with st.spinner("Running Docker pipeline: ASR → segmentation → summarization..."):
            try:
                output_path = run_docker_pipeline(relative_audio_path, cache_dir)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                return

        st.success(f"Pipeline completed. Output: {output_path}")

        # Load and display results
        try:
            df = load_pipeline_output(output_path)
        except Exception as e:
            st.error(f"Failed to load pipeline_output.json: {e}")
            return

        show_results(df)
    else:
        st.info("Upload an audio file and click **Run pipeline** to start.")


def show_results(df: pd.DataFrame) -> None:
    """Render the segmentation + summarization results."""
    st.subheader("Segments overview")

    # Search
    search_query = st.text_input(
        "Search in text or summary",
        value="",
        placeholder="e.g. withdrawal, verification, payment...",
    )

    filtered_df = df.copy()
    if search_query:
        mask = (
            df["text"].str.contains(search_query, case=False, na=False)
            | df["summary"].str.contains(search_query, case=False, na=False)
        )
        filtered_df = df[mask]

    st.write(f"Showing {len(filtered_df)} of {len(df)} segments")

    # Compact table view
    st.dataframe(
        filtered_df[["case_index", "summary"]],
        use_container_width=True,
        height=300,
    )

    st.subheader("Segment details")

    if filtered_df.empty:
        st.info("No segments match the current search.")
        return

    # Choose a segment
    case_indices = filtered_df["case_index"].tolist()
    selected_case_index = st.selectbox(
        "Select segment (case_index)", options=case_indices
    )

    selected_row = df[df["case_index"] == selected_case_index].iloc[0]

    st.markdown(f"### Segment #{selected_case_index}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Summary**")
        st.write(selected_row.get("summary", ""))

    with col2:
        st.markdown("**Full Text (segmented transcript)**")
        st.write(selected_row.get("text", ""))

    # Show raw JSON for debugging / export
    with st.expander("🔍 Raw JSON row"):
        st.json(selected_row.to_dict())


if __name__ == "__main__":
    main()
