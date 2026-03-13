"""Terminal progress bar with tqdm fallback."""

from __future__ import annotations

import shutil
import sys
import time

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ProgressBar:
    """
    A rich terminal progress bar with ETA, speed, and percentage.
    Falls back to tqdm if available, otherwise uses a custom
    implementation.
    """

    def __init__(
        self, total: int, desc: str = "", unit: str = "it"
    ) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        self._term_width = shutil.get_terminal_size((80, 20)).columns
        self._use_tqdm = HAS_TQDM and total > 0
        self._tqdm_bar = None

        if self._use_tqdm:
            self._tqdm_bar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                bar_format="{l_bar}{bar:30}{r_bar}",
                ncols=min(self._term_width, 120),
            )
        elif total > 0:
            self._print_bar()

    def update(self, n: int = 1, suffix: str = "") -> None:
        """Advance the progress bar by *n* steps."""
        self.current = min(self.current + n, self.total)
        if self._use_tqdm:
            self._tqdm_bar.update(n)
            if suffix:
                self._tqdm_bar.set_postfix_str(suffix)
        else:
            self._print_bar(suffix)

    def _print_bar(self, suffix: str = "") -> None:
        if self.total == 0:
            return
        elapsed = time.time() - self.start_time
        pct = self.current / self.total
        speed = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / speed if speed > 0 else 0

        elapsed_str = self._fmt_time(elapsed)
        eta_str = self._fmt_time(eta)

        bar_width = max(20, min(40, self._term_width - 60))
        filled = int(bar_width * pct)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        desc_part = f"{self.desc}: " if self.desc else ""
        line = (
            f"\r{desc_part}{pct:>6.1%} |{bar}| "
            f"{self.current}/{self.total} "
            f"[{elapsed_str}|{eta_str}, {speed:.1f} {self.unit}/s]"
        )
        if suffix:
            line += f" {suffix}"

        line = line[: self._term_width].ljust(self._term_width)
        sys.stderr.write(line)
        sys.stderr.flush()

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m{seconds % 60:02.0f}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m:02d}m"

    def close(self) -> None:
        if self._use_tqdm and self._tqdm_bar:
            self._tqdm_bar.close()
        elif self.total > 0:
            self._print_bar()
            sys.stderr.write("\n")
            sys.stderr.flush()

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
