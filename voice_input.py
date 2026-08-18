import sounddevice as sd
from scipy.io.wavfile import write


def record_audio(filename="voice_input.wav", duration=5, sample_rate=16000):
    print("🎤 Recording started...")
    print("Speak now!")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write(filename, sample_rate, audio)

    print(f"✅ Recording saved as {filename}")

    return filename


if __name__ == "__main__":
    record_audio()