"""Terminal UX: a thinking spinner and a per-turn token/context summary."""
import itertools
import sys
import threading
import time


class Spinner:
    """Stdout spinner shown while a blocking call runs. No-op when output is piped."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = "thinking"):
        self.message = message
        self._stop = threading.Event()
        self._thread = None
        self._active = sys.stdout.isatty()

    def __enter__(self):
        if self._active:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r  {frame} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.08)

    def __exit__(self, *exc):
        if self._active:
            self._stop.set()
            self._thread.join()
            sys.stdout.write("\r\033[K")   # clear the spinner line


def usage_line(prompt_tokens: int, turn_in: int, turn_out: int, window: int) -> str:
    """One-line context/token summary for the turn just finished."""
    pct = (100 * prompt_tokens / window) if window else 0
    return f"  [ctx {prompt_tokens}/{window} ({pct:.0f}%) | this turn: {turn_in} in + {turn_out} out]"
