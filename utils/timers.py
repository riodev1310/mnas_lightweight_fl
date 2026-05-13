from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def elapsed_timer() -> Iterator[callable]:
    start = time.time()
    yield lambda: time.time() - start
