from voice_input import record_audio


def main():
    # Step 1: Record voice
    audio_file = record_audio()

    # Step 2: Show recorded file
    print("\nAudio file ready:")
    print(audio_file)


if __name__ == "__main__":
    main()