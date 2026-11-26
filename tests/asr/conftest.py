from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from websockets.sync.server import ServerConnection, serve


@contextmanager
def ws_server(handler: Callable[[ServerConnection], None]) -> Iterator[Any]:
    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=3)
            assert not thread.is_alive()


@pytest.fixture
def server_factory() -> Callable[[Callable[[ServerConnection], None]], Any]:
    return ws_server
