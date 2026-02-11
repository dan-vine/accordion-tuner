"""
Command-line interface for the tuner
"""

import time

import numpy as np
import sounddevice as sd

from .constants import A4_REFERENCE, BUFFER_SIZE, SAMPLE_RATE
from .multi_pitch_detector import MultiPitchDetector, MultiPitchResult


def main():
    """Simple command-line tuner"""
    detector = MultiPitchDetector(
        sample_rate=SAMPLE_RATE,
        reference=A4_REFERENCE,
    )

    last_result: MultiPitchResult = MultiPitchResult()

    def audio_callback(indata, frames, time_info, status):
        nonlocal last_result
        samples = indata[:, 0]  # Mono
        last_result = detector.process(samples)

    print("Instrument Tuner")
    print("================")
    print(f"Reference: {A4_REFERENCE} Hz")
    print("Press Ctrl+C to exit\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
        ):
            while True:
                if last_result.valid and last_result.maxima:
                    primary = last_result.maxima[0]
                    cents = primary.cents

                    # Build cents bar: |-------|O|-------|
                    bar_width = 30
                    center = bar_width // 2
                    pos = int(center + (cents / 50) * center)
                    pos = max(0, min(bar_width - 1, pos))

                    bar = ["-"] * bar_width
                    bar[center] = "|"
                    bar[pos] = "O"

                    cents_str = f"{cents:+5.1f}"
                    print(
                        f"\r{primary.note_name:2}{primary.octave} "
                        f"{primary.frequency:7.2f} Hz "
                        f"[{''.join(bar)}] "
                        f"{cents_str} cents  ",
                        end="",
                        flush=True,
                    )
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nStopping...")


if __name__ == "__main__":
    main()
