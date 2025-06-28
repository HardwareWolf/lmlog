"""
High-performance logging backends with modern Python 3.11+ features.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Union, TextIO, Protocol
from threading import Lock
from collections.abc import Callable

from .serializers import FastJSONEncoder, MsgSpecEncoder


class LogBackend(Protocol):
    """Protocol for log backends."""

    def write(self, event: Dict[str, Any]) -> None:
        """Write event to backend."""
        ...

    def flush(self) -> None:
        """Flush any buffered data."""
        ...

    def close(self) -> None:
        """Close the backend."""
        ...


class FileBackend:
    """
    High-performance file backend with async writes.

    Uses ThreadPoolExecutor for non-blocking I/O operations.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        encoder: str = "orjson",
        async_writes: bool = True,
        buffer_size: int = 8192,
    ):
        """
        Initialize the file backend.

        Args:
            file_path (str or Path): Path to the log file.
            encoder (str): JSON encoder to use ('orjson' or 'msgspec').
            async_writes (bool): Whether to use async writes with ThreadPoolExecutor.
            buffer_size (int): Buffer size for file writes.
        Raises:
            ValueError: If an unknown encoder is specified.
        """
        self.file_path = Path(file_path)
        self.async_writes = async_writes
        self.buffer_size = buffer_size
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1) if async_writes else None

        if encoder == "orjson":
            self._encoder: Union[FastJSONEncoder, MsgSpecEncoder] = FastJSONEncoder()
        elif encoder == "msgspec":
            self._encoder = MsgSpecEncoder()
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

    def write(self, event: Dict[str, Any]) -> None:
        """Write event to file.

        Args:
            event (Dict[str, Any]): The event data to write.
        Raises:
            RuntimeError: If async writes are enabled but no executor is set.
        """
        if self.async_writes and self._executor:
            self._executor.submit(self._write_sync, event)
        else:
            self._write_sync(event)

    def _write_sync(self, event: Dict[str, Any]) -> None:
        """Synchronous write implementation."""
        with self._lock:
            data = self._encoder.encode(event) + b"\n"
            with open(self.file_path, "ab", buffering=self.buffer_size) as f:
                f.write(data)

    async def awrite(self, event: Dict[str, Any]) -> None:
        """Write event to file asynchronously."""
        if self.async_writes and self._executor:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._write_sync, event)
        else:
            self._write_sync(event)

    def flush(self) -> None:
        """Flush any pending writes."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = ThreadPoolExecutor(max_workers=1)

    def close(self) -> None:
        """Close the backend."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


class StreamBackend:
    """
    High-performance stream backend.

    Optimized for stdout/stderr with minimal overhead.
    """

    def __init__(
        self, stream: TextIO, encoder: str = "orjson", auto_flush: bool = True
    ):
        self.stream = stream
        self.auto_flush = auto_flush
        self._lock = Lock()

        if encoder == "orjson":
            self._encoder: Union[FastJSONEncoder, MsgSpecEncoder] = FastJSONEncoder()
        elif encoder == "msgspec":
            self._encoder = MsgSpecEncoder()
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

    def write(self, event: Dict[str, Any]) -> None:
        """Write event to stream."""
        with self._lock:
            data = self._encoder.encode(event).decode("utf-8")
            self.stream.write(data + "\n")
            if self.auto_flush:
                self.stream.flush()

    async def awrite(self, event: Dict[str, Any]) -> None:
        """Write event to stream asynchronously."""
        self.write(event)

    def flush(self) -> None:
        """Flush stream."""
        self.stream.flush()

    def close(self) -> None:
        """Close stream if not stdout/stderr."""
        if (
            hasattr(self.stream, "close")
            and hasattr(self.stream, "name")
            and self.stream.name
            not in (
                "<stdout>",
                "<stderr>",
            )
        ):
            self.stream.close()


class AsyncFileBackend:
    """
    Async file backend using asyncio for maximum performance.

    Leverages Python 3.11+ asyncio improvements for better concurrency.
    """

    def __init__(self, file_path: Union[str, Path], encoder: str = "orjson"):
        self.file_path = Path(file_path)
        self._write_queue: asyncio.Queue[Union[Dict[str, Any], None]] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None

        if encoder == "orjson":
            self._encoder: Union[FastJSONEncoder, MsgSpecEncoder] = FastJSONEncoder()
        elif encoder == "msgspec":
            self._encoder = MsgSpecEncoder()
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

    async def start(self) -> None:
        """Start the async writer task."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._writer_loop())

    async def stop(self) -> None:
        """Stop the async writer task."""
        self._running = False
        if self._task:
            await self._write_queue.put(None)
            await self._task

    async def write(self, event: Dict[str, Any]) -> None:
        """Queue event for async writing."""
        await self._write_queue.put(event)

    async def _writer_loop(self) -> None:
        """Main writer loop."""
        file_writer = self._open_file()
        while self._running:
            event = await self._write_queue.get()
            if event is None:
                break

            data = self._encoder.encode(event).decode("utf-8")
            file_writer(data + "\n")

    def _open_file(self) -> Callable[[str], None]:
        """Open file for async writing."""

        def write_data(data: str) -> None:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(data)

        return write_data

    def flush(self) -> None:
        """Flush is handled automatically in async mode."""
        pass

    def close(self) -> None:
        """Close backend."""
        if self._task and not self._task.done():
            asyncio.create_task(self.stop())


class BatchingBackend:
    """
    Batching backend for high-throughput scenarios.

    Accumulates events and writes them in batches for optimal performance.
    """

    def __init__(
        self, backend: LogBackend, batch_size: int = 100, flush_interval: float = 1.0
    ):
        self.backend = backend
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._batch: list[Dict[str, Any]] = []
        self._lock = Lock()
        self._last_flush = time.time()

    def write(self, event: Dict[str, Any]) -> None:
        """Add event to batch."""
        with self._lock:
            self._batch.append(event)

            current_time = time.time()
            should_flush = (
                len(self._batch) >= self.batch_size
                or (current_time - self._last_flush) >= self.flush_interval
            )

            if should_flush:
                self._flush_batch()

    def _flush_batch(self) -> None:
        """Flush current batch."""
        if not self._batch:
            return

        for event in self._batch:
            self.backend.write(event)

        self._batch.clear()
        self._last_flush = time.time()
        self.backend.flush()

    def flush(self) -> None:
        """Force flush current batch."""
        with self._lock:
            self._flush_batch()

    def close(self) -> None:
        """Close backend after flushing."""
        self.flush()
        self.backend.close()
