# Speech-to-Cases Pipeline 🗣️ ➜ 📂 ➜ 📝

A simple but production-ready pipeline that turns **call audio** into:

- A full **transcript**
- Several **cases** (segments of the conversation)
- A short **summary** for each case

This is useful for call centers, finance, support, etc.

---

##  What this project does

1. **ASR (Speech-to-Text)**  
   Takes an audio file (e.g. `sample_call.wav`) and creates `transcript.txt`.

2. **Segmentation**  
   Splits the transcript into meaningful chunks (cases) using ML.  
   Output: `cases.json`

3. **Summarization**  
   Summarizes each case into a short description.  
   Output: `summaries.json` and `pipeline_output.json`.

---

## 📂 Project Structure

```text
speech_to_cases/
├── transcribe_call.py        # Audio (.wav) -> transcript.txt
├── segment_cases_ml.py       # transcript.txt -> cases.json
├── summarize_cases.py        # cases.json -> summaries.json
├── pipeline.py               # Runs all 3 steps end-to-end
│
├── Dockerfile                # For running everything in Docker
├── Makefile                  # Shortcuts (make build, make pipeline, etc.)
├── README.md                 # This file
│
├── sample_call.wav           # Example audio (input)
├── transcript.txt            # Example transcript (generated)
├── cases.json                # Example segments (generated)
├── summaries.json            # Example summaries (generated)
├── pipeline_output.json      # Final combined output (generated)
└── ui/
    └── app.py                # Simple Streamlit UI (we'll add this soon)

🧱 How to build and run (Docker)

1️⃣ Build the Docker image

From the project folder:
docker build -t whisper-pipeline .

2️⃣ Run full pipeline (audio -> JSON cases + summaries)
On Windows PowerShell:

docker run --rm `
  -v "${PWD}:/app" `
  -v C:\Users\isode\hf_cache:/root/.cache/huggingface `
  whisper-pipeline python pipeline.py sample_call.wav


After it finishes, you should see a file:
pipeline_output.json

This file contains a list of objects like:
[
  {
    "case_index": 0,
    "text": "Full text of the first case...",
    "summary": "Short summary of what this part of the call is about."
  }
]


🧪 Development notes

You can run each step separately too:
# 1) Audio -> transcript
python transcribe_call.py sample_call.wav

# 2) Transcript -> cases
python segment_cases_ml.py transcript.txt

# 3) Cases -> summaries
python summarize_cases.py cases.json