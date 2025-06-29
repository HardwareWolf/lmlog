"""
Tests for object pooling functionality.
"""

from lmlog.pools import ObjectPool, EventPool, StringPool, BufferPool


class TestObjectPool:
    """Test suite for ObjectPool."""

    def test_basic_operations(self):
        """Test basic pool operations."""
        pool = ObjectPool(dict, max_size=10)

        obj1 = pool.acquire()
        assert isinstance(obj1, dict)
        assert pool.created_count() == 1

        pool.release(obj1)
        assert pool.size() == 1

        obj2 = pool.acquire()
        assert obj2 is obj1  # Same object reused
        assert pool.created_count() == 1

    def test_reset_function(self):
        """Test pool with reset function."""

        def reset_dict(d):
            d.clear()
            d["reset"] = True

        pool = ObjectPool(dict, reset_func=reset_dict, max_size=10)

        obj = pool.acquire()
        obj["test"] = "value"
        pool.release(obj)

        obj2 = pool.acquire()
        assert obj2 is obj
        assert obj2["reset"] is True
        assert "test" not in obj2

    def test_max_size_limit(self):
        """Test pool size limits."""
        pool = ObjectPool(dict, max_size=2)

        obj1 = pool.acquire()
        obj2 = pool.acquire()
        obj3 = pool.acquire()

        pool.release(obj1)
        pool.release(obj2)
        pool.release(obj3)  # Should be discarded due to size limit

        assert pool.size() == 2


class TestEventPool:
    """Test suite for EventPool."""

    def test_event_acquisition(self):
        """Test event dictionary acquisition."""
        pool = EventPool(max_size=10)

        event1 = pool.acquire()
        assert isinstance(event1, dict)
        assert len(event1) == 0

        event1["test"] = "value"
        pool.release(event1)

        event2 = pool.acquire()
        assert event2 is event1
        assert len(event2) == 0  # Should be cleared

    def test_pool_statistics(self):
        """Test pool statistics."""
        pool = EventPool(max_size=5)

        assert pool.size() == 0
        assert pool.created_count() == 0

        event = pool.acquire()
        assert pool.created_count() == 1

        pool.release(event)
        assert pool.size() == 1


class TestStringPool:
    """Test suite for StringPool."""

    def test_string_interning(self):
        """Test string interning."""
        pool = StringPool(max_size=100)

        str1 = pool.intern("test_string")
        str2 = pool.intern("test_string")

        assert str1 is str2
        assert pool.size() == 1

    def test_max_size_behavior(self):
        """Test behavior when max size is reached."""
        pool = StringPool(max_size=2)

        str1 = pool.intern("string1")
        str2 = pool.intern("string2")
        pool.intern("string3")  # Should not be cached

        assert pool.size() == 2
        assert pool.intern("string1") is str1
        assert pool.intern("string2") is str2

    def test_cache_clearing(self):
        """Test cache clearing."""
        pool = StringPool()

        pool.intern("test1")
        pool.intern("test2")
        assert pool.size() == 2

        pool.clear()
        assert pool.size() == 0


class TestBufferPool:
    """Test suite for BufferPool."""

    def test_buffer_acquisition(self):
        """Test buffer acquisition."""
        pool = BufferPool(default_size=1024)

        buffer1 = pool.acquire()
        assert isinstance(buffer1, bytearray)
        assert len(buffer1) == 1024

        buffer2 = pool.acquire(2048)
        assert len(buffer2) == 2048

    def test_buffer_reuse(self):
        """Test buffer reuse."""
        pool = BufferPool()

        buffer1 = pool.acquire(1024)
        buffer1[:4] = b"test"
        pool.release(buffer1)

        buffer2 = pool.acquire(1024)
        assert buffer2 is buffer1
        assert buffer2[:4] == b"\x00\x00\x00\x00"  # Should be cleared

    def test_power_of_2_rounding(self):
        """Test power of 2 rounding."""
        pool = BufferPool()

        buffer = pool.acquire(1000)
        assert len(buffer) == 1024  # Rounded up to next power of 2

    def test_different_sizes(self):
        """Test handling different buffer sizes."""
        pool = BufferPool()

        buffer1 = pool.acquire(512)
        buffer2 = pool.acquire(1024)

        pool.release(buffer1)
        pool.release(buffer2)

        # Should get correct size back
        buffer3 = pool.acquire(512)
        buffer4 = pool.acquire(1024)

        assert buffer3 is buffer1
        assert buffer4 is buffer2


class TestGlobalPools:
    """Test global pool functions."""

    def test_global_pools_exist(self):
        """Test that global pools are accessible."""
        from lmlog.pools import get_event_pool, get_string_pool, get_buffer_pool

        event_pool = get_event_pool()
        string_pool = get_string_pool()
        buffer_pool = get_buffer_pool()

        assert isinstance(event_pool, EventPool)
        assert isinstance(string_pool, StringPool)
        assert isinstance(buffer_pool, BufferPool)

    def test_global_pools_singleton(self):
        """Test that global pools are singletons."""
        from lmlog.pools import get_event_pool, get_string_pool, get_buffer_pool

        assert get_event_pool() is get_event_pool()
        assert get_string_pool() is get_string_pool()
        assert get_buffer_pool() is get_buffer_pool()
