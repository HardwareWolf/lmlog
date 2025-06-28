"""
Object pooling for high-performance memory management.
"""

import threading
from collections import deque
from typing import Any, Dict, Optional, TypeVar, Generic, Callable

T = TypeVar("T")


class ObjectPool(Generic[T]):
    """
    Generic object pool for high-performance object reuse.

    Uses a thread-safe deque for O(1) operations and provides
    automatic cleanup of unused objects.
    """

    __slots__ = (
        "_pool",
        "_factory",
        "_reset_func",
        "_max_size",
        "_lock",
        "_created_count",
    )

    def __init__(
        self,
        factory: Callable[[], T],
        reset_func: Optional[Callable[[T], None]] = None,
        max_size: int = 1000,
    ):
        """
        Initialize object pool.

        Args:
            factory: Function to create new objects
            reset_func: Function to reset objects before reuse
            max_size: Maximum number of objects in pool
        """
        self._pool: deque[T] = deque(maxlen=max_size)
        self._factory = factory
        self._reset_func = reset_func
        self._max_size = max_size
        self._lock = threading.Lock()
        self._created_count = 0

    def acquire(self) -> T:
        """
        Acquire an object from the pool.

        Returns:
            Object from pool or newly created object
        """
        with self._lock:
            if self._pool:
                return self._pool.popleft()

        self._created_count += 1
        return self._factory()

    def release(self, obj: T) -> None:
        """
        Release an object back to the pool.

        Args:
            obj: Object to return to pool
        """
        if self._reset_func:
            self._reset_func(obj)

        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(obj)

    def size(self) -> int:
        """Get current pool size."""
        return len(self._pool)

    def created_count(self) -> int:
        """Get total number of objects created."""
        return self._created_count


class EventPool:
    """
    Specialized pool for log event dictionaries.
    """

    __slots__ = ("_pool", "_lock", "_max_size", "_created_count")

    def __init__(self, max_size: int = 1000):
        """
        Initialize event pool.

        Args:
            max_size: Maximum number of events in pool
        """
        self._pool: deque[Dict[str, Any]] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._max_size = max_size
        self._created_count = 0

    def acquire(self) -> Dict[str, Any]:
        """
        Acquire an event dictionary from the pool.

        Returns:
            Event dictionary from pool or newly created
        """
        with self._lock:
            if self._pool:
                return self._pool.popleft()

        self._created_count += 1
        return {}

    def release(self, event: Dict[str, Any]) -> None:
        """
        Release an event dictionary back to the pool.

        Args:
            event: Event dictionary to return to pool
        """
        event.clear()

        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(event)

    def size(self) -> int:
        """Get current pool size."""
        return len(self._pool)

    def created_count(self) -> int:
        """Get total number of events created."""
        return self._created_count


class StringPool:
    """
    Pool for frequently used strings to reduce allocations.
    """

    __slots__ = ("_cache", "_lock", "_max_size")

    def __init__(self, max_size: int = 10000):
        """
        Initialize string pool.

        Args:
            max_size: Maximum number of strings to cache
        """
        self._cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._max_size = max_size

    def intern(self, string: str) -> str:
        """
        Intern a string for reuse.

        Args:
            string: String to intern

        Returns:
            Interned string instance
        """
        with self._lock:
            if string in self._cache:
                return self._cache[string]

            if len(self._cache) < self._max_size:
                self._cache[string] = string
                return string

        return string

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear the string cache."""
        with self._lock:
            self._cache.clear()


class BufferPool:
    """
    Pool for reusable byte buffers.
    """

    __slots__ = ("_pools", "_lock", "_default_size")

    def __init__(self, default_size: int = 8192):
        """
        Initialize buffer pool.

        Args:
            default_size: Default buffer size
        """
        self._pools: Dict[int, deque[bytearray]] = {}
        self._lock = threading.Lock()
        self._default_size = default_size

    def acquire(self, size: Optional[int] = None) -> bytearray:
        """
        Acquire a buffer of specified size.

        Args:
            size: Buffer size (uses default if None)

        Returns:
            Buffer of requested size
        """
        if size is None:
            size = self._default_size

        size = self._round_up_to_power_of_2(size)

        with self._lock:
            if size not in self._pools:
                self._pools[size] = deque(maxlen=100)

            pool = self._pools[size]
            if pool:
                buffer = pool.popleft()
                if len(buffer) > 0:
                    for i in range(len(buffer)):
                        buffer[i] = 0
                return buffer

        return bytearray(size)

    def release(self, buffer: bytearray) -> None:
        """
        Release a buffer back to the pool.

        Args:
            buffer: Buffer to return to pool
        """
        size = len(buffer)

        with self._lock:
            if size not in self._pools:
                self._pools[size] = deque(maxlen=100)

            pool = self._pools[size]
            if len(pool) < 100:
                pool.append(buffer)

    def _round_up_to_power_of_2(self, n: int) -> int:
        """Round up to nearest power of 2."""
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()


_global_event_pool = EventPool()
_global_string_pool = StringPool()
_global_buffer_pool = BufferPool()


def get_event_pool() -> EventPool:
    """Get the global event pool."""
    return _global_event_pool


def get_string_pool() -> StringPool:
    """Get the global string pool."""
    return _global_string_pool


def get_buffer_pool() -> BufferPool:
    """Get the global buffer pool."""
    return _global_buffer_pool
