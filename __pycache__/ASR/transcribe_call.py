import sys
import whisper


def transcribe_audio(audio_path: str) -> str:
    print("Loading Whisper model (small)...")
    model = whisper.load_model("small")

    print(f"Transcribing audio: {audio_path}")
    result = model.transcribe(audio_path)

    transcript = result["text"]
    return transcript


def main():
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "sample_call.wav"

    transcript = transcribe_audio(audio_file)

    print("\n=== FINAL TRANSCRIPT ===\n")
    print(transcript)
    print("\n========================\n")


if __name__ == "__main__":
    main()
