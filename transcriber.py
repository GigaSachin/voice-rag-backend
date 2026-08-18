import whisper


def transcribe_audio(audio_file):

    print("Loading Whisper model...")

    model = whisper.load_model("base")

    print("Transcribing audio...")

    result = model.transcribe(audio_file)

    return result["text"]


if __name__ == "__main__":

    audio_file = "voice_input.wav"

    text = transcribe_audio(audio_file)

    print("\n--- TRANSCRIBED TEXT ---")
    print(text)