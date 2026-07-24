from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import time

from easyllama.helpers import _download_file

CONTENT = b"easyllama" * 1024 * 128


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(CONTENT)))
        self.end_headers()
        for offset in range(0, len(CONTENT), 16 * 1024):
            self.wfile.write(CONTENT[offset : offset + 16 * 1024])
            self.wfile.flush()
            time.sleep(0.08)

    def log_message(self, format: str, *args: object) -> None:
        pass


class Messages(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
Thread(target=server.serve_forever, daemon=True).start()
messages = Messages()
logger = logging.getLogger("easyllama.helpers")
logger.addHandler(messages)
logger.setLevel(logging.INFO)
try:
    with TemporaryDirectory() as temp_dir:
        destination = Path(temp_dir) / "download.bin"
        _download_file(f"http://127.0.0.1:{server.server_port}/file", destination, None)
        assert destination.read_bytes() == CONTENT
finally:
    logger.removeHandler(messages)
    server.shutdown()

assert any("%" in message and "/s, ETA" in message for message in messages.messages)
assert any("Download complete" in message and "/s)" in message for message in messages.messages)
