from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO


class TeeStream:
    def __init__(self, stream: TextIO, log_file: TextIO, log_carriage_returns: bool = False) -> None:
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")
        self.log_carriage_returns = log_carriage_returns
        self._cplightsit_tee = True
        self._cplightsit_log_path = str(Path(log_file.name))

    def write(self, data: str) -> int:
        written = self.stream.write(data)
        if not self.log_file.closed:
            if self.log_carriage_returns or "\r" not in data:
                self.log_file.write(data)
            elif "\n" in data:
                self.log_file.write("\n")
        return written

    def flush(self) -> None:
        self.stream.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()

    def __getattr__(self, name: str) -> object:
        return getattr(self.stream, name)


def unwrap_terminal_stream(stream: TextIO) -> TextIO:
    current = stream
    while getattr(current, "_cplightsit_tee", False):
        current = getattr(current, "stream")  # type: ignore[assignment]
    return current


def _iter_stream_handlers() -> list[logging.StreamHandler]:
    handlers: list[logging.StreamHandler] = []
    logger_dict = logging.root.manager.loggerDict
    for logger in [logging.getLogger(), *[item for item in logger_dict.values() if isinstance(item, logging.Logger)]]:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handlers.append(handler)
    return handlers


def _redirect_logging_handlers(old_stdout: TextIO, old_stderr: TextIO) -> None:
    for handler in _iter_stream_handlers():
        if handler.stream is old_stdout:
            handler.setStream(sys.stdout)
        elif handler.stream is old_stderr:
            handler.setStream(sys.stderr)


def setup_output_log(
    result_dir: str | Path,
    rank: int = 0,
    enabled: bool = True,
    rank_zero_only: bool = True,
) -> Path | None:
    if not enabled:
        return None
    if rank_zero_only and rank != 0:
        return None

    log_path = Path(result_dir) / ("log.txt" if rank == 0 else f"log_rank{rank}.txt")
    current_log_path = getattr(sys.stdout, "_cplightsit_log_path", None)
    if current_log_path == str(log_path):
        return log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = TeeStream(old_stdout, log_file)  # type: ignore[assignment]
    sys.stderr = TeeStream(old_stderr, log_file)  # type: ignore[assignment]
    _redirect_logging_handlers(old_stdout, old_stderr)
    print(f"Writing process output to {log_path}")
    return log_path
