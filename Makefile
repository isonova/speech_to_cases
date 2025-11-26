.RECIPEPREFIX = >

DOCKER_RUN = docker run --rm -v "$(CURDIR):/app" -v C:/Users/isode/hf_cache:/root/.cache/huggingface whisper-pipeline

build:
>docker build -t whisper-pipeline .

asr:
>$(DOCKER_RUN) python transcribe_call.py sample_call.wav

segment:
>$(DOCKER_RUN) python segment_cases_ml.py transcript.txt

summarize:
>$(DOCKER_RUN) python summarize_cases.py cases.json

pipeline:
>$(DOCKER_RUN) python pipeline.py sample_call.wav

clean:
>del transcript.txt cases.json summaries.json pipeline_output.json 2>nul || true