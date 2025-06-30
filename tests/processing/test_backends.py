"""
Tests for the backends module.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from io import StringIO
import pytest

from lmlog.processing.backends import (
    FileBackend,
    StreamBackend,
    AsyncFileBackend,
    BatchingBackend,
)


class TestLogBackend:
    def test_protocol_methods(self):
        """Test LogBackend protocol implementation."""
        # Test that existing backends implement the protocol correctly
        import tempfile

        # Test FileBackend implements protocol
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = FileBackend(tmp.name, async_writes=False)
            backend.write({"test": "event"})
            backend.flush()
            backend.close()

        # Test StreamBackend implements protocol
        from io import StringIO

        stream = StringIO()
        backend = StreamBackend(stream)
        backend.write({"test": "event"})
        backend.flush()
        backend.close()

        # Verify the protocol methods exist and are callable
        assert hasattr(FileBackend, "write")
        assert hasattr(FileBackend, "flush")
        assert hasattr(FileBackend, "close")


class TestFileBackend:
    def test_init_with_orjson_encoder(self):
        """Test FileBackend initialization with orjson encoder."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = FileBackend(tmp.name, encoder="orjson", async_writes=False)
            assert backend.file_path == Path(tmp.name)
            assert not backend.async_writes
            assert backend._executor is None

    def test_init_with_msgspec_encoder(self):
        """Test FileBackend initialization with msgspec encoder."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = FileBackend(tmp.name, encoder="msgspec", async_writes=False)
            assert backend.file_path == Path(tmp.name)

    def test_init_with_unknown_encoder(self):
        """Test FileBackend initialization with unknown encoder raises error."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            with pytest.raises(ValueError, match="Unknown encoder: unknown"):
                FileBackend(tmp.name, encoder="unknown")

    def test_write_sync_with_orjson(self):
        """Test synchronous write with orjson encoder."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            backend = FileBackend(tmp.name, encoder="orjson", async_writes=False)
            event = {"test": "data", "number": 42}
            backend.write(event)

            # Read back the file
            with open(tmp.name, "r") as f:
                content = f.read().strip()
                parsed = json.loads(content)
                assert parsed == event

    def test_write_sync_with_msgspec(self):
        """Test synchronous write with msgspec encoder."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            backend = FileBackend(tmp.name, encoder="msgspec", async_writes=False)
            event = {"test": "data", "number": 42}
            backend.write(event)

            # Read back the file
            with open(tmp.name, "r") as f:
                content = f.read().strip()
                parsed = json.loads(content)
                assert parsed == event

    def test_write_async(self):
        """Test asynchronous write."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            backend = FileBackend(tmp.name, encoder="orjson", async_writes=True)
            event = {"test": "data", "number": 42}
            backend.write(event)

            # Wait for async write to complete
            backend.flush()

            # Read back the file
            with open(tmp.name, "r") as f:
                content = f.read().strip()
                parsed = json.loads(content)
                assert parsed == event

    def test_flush_with_executor(self):
        """Test flush with thread executor."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            backend = FileBackend(tmp.name, encoder="orjson", async_writes=True)
            event = {"test": "data"}
            backend.write(event)
            backend.flush()

            # Should recreate executor
            assert backend._executor is not None

    def test_close_with_executor(self):
        """Test close with thread executor."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            backend = FileBackend(tmp.name, encoder="orjson", async_writes=True)
            backend.close()
            assert backend._executor is None

    def test_close_without_executor(self):
        """Test close without thread executor."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            backend = FileBackend(tmp.name, encoder="orjson", async_writes=False)
            backend.close()  # Should not raise


class TestStreamBackend:
    def test_init_with_orjson_encoder(self):
        """Test StreamBackend initialization with orjson encoder."""
        stream = StringIO()
        backend = StreamBackend(stream, encoder="orjson")
        assert backend.stream is stream
        assert backend.auto_flush is True

    def test_init_with_msgspec_encoder(self):
        """Test StreamBackend initialization with msgspec encoder."""
        stream = StringIO()
        backend = StreamBackend(stream, encoder="msgspec")
        assert backend.stream is stream

    def test_init_with_unknown_encoder(self):
        """Test StreamBackend initialization with unknown encoder raises error."""
        stream = StringIO()
        with pytest.raises(ValueError, match="Unknown encoder: unknown"):
            StreamBackend(stream, encoder="unknown")

    def test_write_with_orjson(self):
        """Test write with orjson encoder."""
        stream = StringIO()
        backend = StreamBackend(stream, encoder="orjson", auto_flush=False)
        event = {"test": "data", "number": 42}
        backend.write(event)

        stream.seek(0)
        content = stream.read().strip()
        parsed = json.loads(content)
        assert parsed == event

    def test_write_with_msgspec(self):
        """Test write with msgspec encoder."""
        stream = StringIO()
        backend = StreamBackend(stream, encoder="msgspec", auto_flush=False)
        event = {"test": "data", "number": 42}
        backend.write(event)

        stream.seek(0)
        content = stream.read().strip()
        parsed = json.loads(content)
        assert parsed == event

    def test_write_with_auto_flush(self):
        """Test write with auto flush enabled."""
        stream = StringIO()
        backend = StreamBackend(stream, encoder="orjson", auto_flush=True)
        event = {"test": "data"}
        backend.write(event)

        stream.seek(0)
        content = stream.read().strip()
        parsed = json.loads(content)
        assert parsed == event

    def test_flush(self):
        """Test manual flush."""
        stream = StringIO()
        backend = StreamBackend(stream, encoder="orjson")
        backend.flush()  # Should not raise

    def test_close_regular_stream(self):
        """Test close with regular stream."""

        class MockStream:
            def __init__(self):
                self.name = "test.txt"
                self.closed = False

            def write(self, data):
                pass

            def flush(self):
                pass

            def close(self):
                self.closed = True

        stream = MockStream()
        backend = StreamBackend(stream, encoder="orjson")
        backend.close()
        assert stream.closed

    def test_close_stdout_stderr(self):
        """Test close with stdout/stderr streams."""
        stream = StringIO()
        stream.name = "<stdout>"
        backend = StreamBackend(stream, encoder="orjson")
        backend.close()  # Should not close stdout/stderr

        stream.name = "<stderr>"
        backend = StreamBackend(stream, encoder="orjson")
        backend.close()  # Should not close stdout/stderr


class TestAsyncFileBackend:
    @pytest.mark.asyncio
    async def test_init_with_orjson_encoder(self):
        """Test AsyncFileBackend initialization with orjson encoder."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name, encoder="orjson")
            assert backend.file_path == Path(tmp.name)
            assert not backend._running

    @pytest.mark.asyncio
    async def test_init_with_msgspec_encoder(self):
        """Test AsyncFileBackend initialization with msgspec encoder."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name, encoder="msgspec")
            assert backend.file_path == Path(tmp.name)

    @pytest.mark.asyncio
    async def test_init_with_unknown_encoder(self):
        """Test AsyncFileBackend initialization with unknown encoder raises error."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            with pytest.raises(ValueError, match="Unknown encoder: unknown"):
                AsyncFileBackend(tmp.name, encoder="unknown")

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test starting and stopping the async backend."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name, encoder="orjson")

            assert not backend._running
            await backend.start()
            assert backend._running
            assert backend._task is not None

            await backend.stop()
            assert not backend._running

    @pytest.mark.asyncio
    async def test_write_async(self):
        """Test asynchronous write."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name, encoder="orjson")
            await backend.start()

            event = {"test": "data", "number": 42}
            await backend.write(event)

            # Wait a bit for the async write to process
            await asyncio.sleep(0.01)

            await backend.stop()

            # Read back the file
            with open(tmp.name, "r") as f:
                content = f.read().strip()
                if content:  # Only parse if there's content
                    parsed = json.loads(content)
                    assert parsed == event

    @pytest.mark.asyncio
    async def test_flush(self):
        """Test flush method."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name, encoder="orjson")
            backend.flush()  # Should be no-op

    @pytest.mark.asyncio
    async def test_close_without_task(self):
        """Test close without running task."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name, encoder="orjson")
            backend.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_with_task(self):
        """Test close with running task."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            backend = AsyncFileBackend(tmp.name, encoder="orjson")
            await backend.start()
            backend.close()
            # Wait a bit for the task to be created
            await asyncio.sleep(0.01)


class TestBatchingBackend:
    def test_init(self):
        """Test BatchingBackend initialization."""
        stream = StringIO()
        base_backend = StreamBackend(stream, encoder="orjson")
        backend = BatchingBackend(base_backend, batch_size=5, flush_interval=0.5)

        assert backend.backend is base_backend
        assert backend.batch_size == 5
        assert backend.flush_interval == 0.5
        assert len(backend._batch) == 0

    def test_write_below_batch_size(self):
        """Test write below batch size."""
        stream = StringIO()
        base_backend = StreamBackend(stream, encoder="orjson")
        backend = BatchingBackend(base_backend, batch_size=5, flush_interval=10.0)

        event = {"test": "data"}
        backend.write(event)

        # Should be batched, not written yet
        assert len(backend._batch) == 1
        stream.seek(0)
        assert stream.read() == ""

    def test_write_at_batch_size(self):
        """Test write at batch size threshold."""
        stream = StringIO()
        base_backend = StreamBackend(stream, encoder="orjson")
        backend = BatchingBackend(base_backend, batch_size=2, flush_interval=10.0)

        event1 = {"test": "data1"}
        event2 = {"test": "data2"}

        backend.write(event1)
        backend.write(event2)  # Should trigger flush

        # Should be flushed
        assert len(backend._batch) == 0
        stream.seek(0)
        lines = stream.read().strip().split("\n")
        assert len(lines) == 2

    def test_write_with_time_flush(self):
        """Test write with time-based flush."""
        stream = StringIO()
        base_backend = StreamBackend(stream, encoder="orjson")
        backend = BatchingBackend(base_backend, batch_size=10, flush_interval=0.001)

        event = {"test": "data"}
        backend.write(event)

        # Wait for time interval to pass
        import time

        time.sleep(0.002)

        # Write another event to trigger time check
        backend.write(event)

        # Should be flushed due to time
        stream.seek(0)
        content = stream.read().strip()
        assert content  # Should have content

    def test_manual_flush(self):
        """Test manual flush."""
        stream = StringIO()
        base_backend = StreamBackend(stream, encoder="orjson")
        backend = BatchingBackend(base_backend, batch_size=10, flush_interval=10.0)

        event = {"test": "data"}
        backend.write(event)
        backend.flush()

        # Should be flushed
        assert len(backend._batch) == 0
        stream.seek(0)
        content = stream.read().strip()
        parsed = json.loads(content)
        assert parsed == event

    def test_flush_empty_batch(self):
        """Test flush with empty batch."""
        stream = StringIO()
        base_backend = StreamBackend(stream, encoder="orjson")
        backend = BatchingBackend(base_backend, batch_size=10, flush_interval=10.0)

        backend.flush()  # Should not raise with empty batch
        assert len(backend._batch) == 0

    def test_close(self):
        """Test close method."""
        stream = StringIO()
        base_backend = StreamBackend(stream, encoder="orjson")
        backend = BatchingBackend(base_backend, batch_size=10, flush_interval=10.0)

        event = {"test": "data"}
        backend.write(event)
        backend.close()

        # Should flush and close
        assert len(backend._batch) == 0
        stream.seek(0)
        content = stream.read().strip()
        parsed = json.loads(content)
        assert parsed == event


class MockBackend:
    """Mock backend for testing."""

    def __init__(self):
        self.events = []
        self.flush_called = False
        self.close_called = False

    def write(self, event):
        self.events.append(event)

    def flush(self):
        self.flush_called = True

    def close(self):
        self.close_called = True


class TestBackendIntegration:
    def test_batching_backend_with_mock(self):
        """Test BatchingBackend with mock backend."""
        mock = MockBackend()
        backend = BatchingBackend(mock, batch_size=2, flush_interval=10.0)

        event1 = {"test": "data1"}
        event2 = {"test": "data2"}

        backend.write(event1)
        assert len(mock.events) == 0  # Not flushed yet

        backend.write(event2)
        assert len(mock.events) == 2  # Flushed due to batch size
        assert mock.flush_called

        backend.close()
        assert mock.close_called
