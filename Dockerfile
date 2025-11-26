FROM python:3.11-slim

# System deps (ffmpeg for Whisper)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all Python dependencies in ONE layer (clean + cached)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        openai-whisper \
        datasets \
        huggingface_hub \
        sentence-transformers \
        transformers \
        sentencepiece \
        accelerate

# Copy project files into image
COPY . .

# Default command (can be overridden)
CMD ["python", "transcribe_call.py", "sample_call.wav"]

RUN pip install pandas openpyxl
