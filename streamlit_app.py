"""Streamlit UI for patient huddle analysis."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.config import DEFAULT_HUDDLE_MODEL, resolve_repo_path
from src.services.huddle_analyzer import HuddleAnalyzer

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Patient Huddle Analysis",
    page_icon="🏥",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────

# st.markdown(
#     """
#     <style>
#     .stApp { max-width: 760px; margin: auto; }
#     .block-container { padding-top: 2rem; padding-bottom: 2rem; }
#     h1 { color: #1F4E79; }
#     h3 { color: #2E75B6; margin-top: 1.4rem; }

#     /* ── Green checkboxes ── */
#     .stCheckbox label { font-size: 0.97rem; }

#     /* native checkbox accent (fallback for browsers that expose the input) */
#     [data-testid="stCheckbox"] input[type="checkbox"] {
#         accent-color: #28a745;
#     }

#     /* Streamlit's custom checkbox — the visual indicator is the FIRST span inside label */
#     [data-testid="stCheckbox"] label > span:first-of-type {
#         border-color: #28a745 !important;
#         outline-color: #28a745 !important;
#     }
#     /* filled state — only the indicator span, NOT the text span */
#     [data-testid="stCheckbox"] label > input:checked ~ span:first-of-type,
#     [data-testid="stCheckbox"] label > span[aria-checked="true"],
#     [data-testid="stCheckbox"] label > span[data-checked="true"] {
#         background-color: #28a745 !important;
#         border-color: #28a745 !important;
#     }
#     /* the SVG tick mark inside the indicator */
#     [data-testid="stCheckbox"] label > span:first-of-type svg {
#         fill: white !important;
#         color: white !important;
#     }

#     /* ── Buttons ── */
#     .stButton > button {
#         background-color: #1F4E79;
#         color: white;
#         border-radius: 6px;
#         padding: 0.5rem 2rem;
#         font-size: 1rem;
#         font-weight: 600;
#         margin-top: 0.5rem;
#     }
#     .stButton > button:hover { background-color: #2E75B6; }
#     .stDownloadButton > button {
#         background-color: #28a745;
#         color: white;
#         border-radius: 6px;
#         padding: 0.5rem 2rem;
#         font-size: 1rem;
#         font-weight: 600;
#     }
#     .stDownloadButton > button:hover { background-color: #218838; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏥 Patient Huddle Analysis")
st.markdown("Run AI-powered pre-huddle analysis for a patient and download the summary PDF.")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────

st.markdown("### Patient")
patient_id = st.text_input(
    "Patient ID",
    placeholder="e.g. 123455",
    help="Enter the patient ID to look up in patients.json",
).strip()

with st.expander("⚙️  Analyses to run", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        run_medication = st.checkbox("Medication Gap Analysis", value=True)
        run_lab        = st.checkbox("Lab Report Analysis", value=True)
    with col2:
        run_summary = st.checkbox("Doctor Pre-Huddle Summary", value=True)

st.divider()

# ── Run ───────────────────────────────────────────────────────────────────────

run_clicked = st.button("Run Analysis", use_container_width=True)

if run_clicked:
    if not patient_id:
        st.error("Please enter a Patient ID before running.")
        st.stop()

    if not any([run_medication, run_lab, run_summary]):
        st.error("Please select at least one analysis to run.")
        st.stop()

    patients_json = str(resolve_repo_path("patients.json"))

    status = st.status("Running analysis…", expanded=True)

    with status:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                analyzer = HuddleAnalyzer()

                steps = []
                if run_medication: steps.append("Medication Gap Analysis")
                if run_lab:        steps.append("Lab Report Analysis (per-report + combined)")
                if run_summary:   steps.append("Doctor Pre-Huddle Summary")

                st.write(f"**Patient:** `{patient_id}`")
                st.write("**Running:** " + " · ".join(steps))
                st.divider()

                if run_lab:
                    st.write("⏳ [Tool Call] Fetching clinical thresholds…")

                # Live progress: each message is written into the status box
                def on_progress(msg: str) -> None:
                    status.write(msg)

                analyzer.analyze_patient_huddle(
                    patient_id=patient_id,
                    patients_json_path=patients_json,
                    output_dir=tmp_dir,
                    model=DEFAULT_HUDDLE_MODEL,
                    use_web_search=run_lab,
                    enable_medication_analysis=run_medication,
                    enable_per_report_lab_analysis=run_lab,
                    enable_combined_lab_analysis=run_lab,
                    enable_doctor_summary=run_summary,
                    progress_callback=on_progress,
                )

                pdf_path = Path(tmp_dir) / f"{patient_id}.pdf"

                if not pdf_path.exists():
                    status.update(label="Analysis complete — PDF not generated.", state="error")
                    st.error("The analysis ran but no PDF was produced. Check the patient ID.")
                    st.stop()

                pdf_bytes = pdf_path.read_bytes()
                st.divider()
                st.write("🎉 All steps complete!")
                status.update(label="Analysis complete!", state="complete", expanded=False)

            except ValueError as exc:
                status.update(label="Error", state="error", expanded=True)
                st.error(f"**Patient not found:** {exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                status.update(label="Error", state="error", expanded=True)
                st.error(f"**Unexpected error:** {exc}")
                st.stop()

    # Download button rendered outside the status/tmp block after bytes are captured
    st.success("Your huddle summary PDF is ready.")
    st.download_button(
        label="⬇️  Download PDF",
        data=pdf_bytes,
        file_name=f"huddle_{patient_id}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
