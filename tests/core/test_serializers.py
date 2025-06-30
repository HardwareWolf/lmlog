"""
Tests for the serializers module.
"""

import json
from datetime import datetime

from lmlog.core.serializers import (
    FastJSONEncoder,
    MsgSpecEncoder,
    EventSerializer,
    create_event_struct,
    default_serializer,
)


class TestFastJSONEncoder:
    def test_encode_bytes(self):
        """Test encoding to bytes."""
        encoder = FastJSONEncoder()
        data = {"test": "data", "number": 42}
        result = encoder.encode(data)
        assert isinstance(result, bytes)

        # Decode and verify
        decoded = json.loads(result.decode("utf-8"))
        assert decoded == data

    def test_encode_str(self):
        """Test encoding to string."""
        encoder = FastJSONEncoder()
        data = {"test": "data", "number": 42}
        result = encoder.encode_str(data)
        assert isinstance(result, str)

        # Decode and verify
        decoded = json.loads(result)
        assert decoded == data

    def test_encode_with_datetime(self):
        """Test encoding with datetime objects."""
        encoder = FastJSONEncoder()
        dt = datetime(2023, 1, 1, 12, 0, 0)
        data = {"timestamp": dt, "value": 42}
        result = encoder.encode_str(data)

        # Should be able to decode
        decoded = json.loads(result)
        assert "timestamp" in decoded
        assert decoded["value"] == 42


class TestMsgSpecEncoder:
    def test_init(self):
        """Test MsgSpecEncoder initialization."""
        encoder = MsgSpecEncoder()
        assert encoder._encoder is not None
        assert encoder._decoder is not None

    def test_encode_bytes(self):
        """Test encoding structured event to bytes."""
        encoder = MsgSpecEncoder()
        event = EventSerializer(
            event_type="test",
            timestamp="2023-01-01T12:00:00Z",
            source={"file": "test.py", "line": 1},
        )
        result = encoder.encode(event)
        assert isinstance(result, bytes)

    def test_encode_str(self):
        """Test encoding structured event to string."""
        encoder = MsgSpecEncoder()
        event = EventSerializer(
            event_type="test",
            timestamp="2023-01-01T12:00:00Z",
            source={"file": "test.py", "line": 1},
        )
        result = encoder.encode_str(event)
        assert isinstance(result, str)

        # Should be valid JSON
        json.loads(result)

    def test_decode(self):
        """Test decoding JSON to event structure."""
        encoder = MsgSpecEncoder()
        event = EventSerializer(
            event_type="test",
            timestamp="2023-01-01T12:00:00Z",
            source={"file": "test.py", "line": 1},
            context={"key": "value"},
        )

        # Encode then decode
        encoded = encoder.encode(event)
        decoded = encoder.decode(encoded)

        assert isinstance(decoded, EventSerializer)
        assert decoded.event_type == "test"
        assert decoded.timestamp == "2023-01-01T12:00:00Z"
        assert decoded.context == {"key": "value"}

    def test_decode_from_string(self):
        """Test decoding from string."""
        encoder = MsgSpecEncoder()
        json_str = '{"event_type": "test", "timestamp": "2023-01-01T12:00:00Z", "source": {"file": "test.py"}}'

        decoded = encoder.decode(json_str)
        assert isinstance(decoded, EventSerializer)
        assert decoded.event_type == "test"


class TestEventSerializer:
    def test_create_minimal_event(self):
        """Test creating minimal event."""
        event = EventSerializer(
            event_type="test",
            timestamp="2023-01-01T12:00:00Z",
            source={"file": "test.py", "line": 1},
        )
        assert event.event_type == "test"
        assert event.timestamp == "2023-01-01T12:00:00Z"
        assert event.source == {"file": "test.py", "line": 1}
        assert event.entity is None
        assert event.operation is None

    def test_create_full_event(self):
        """Test creating event with all fields."""
        event = EventSerializer(
            event_type="test",
            timestamp="2023-01-01T12:00:00Z",
            source={"file": "test.py", "line": 1},
            entity={"type": "user", "id": "123"},
            operation="test_op",
            context={"key": "value"},
            performance={"duration_ms": 100},
            error_info={"type": "ValueError", "message": "test error"},
            state_change={"field": "status", "before": "old", "after": "new"},
            violation={"rule": "test_rule"},
            integration={"service": "test_service"},
            authentication={"type": "oauth"},
            behavior={"type": "anomaly"},
        )
        assert event.event_type == "test"
        assert event.entity == {"type": "user", "id": "123"}
        assert event.operation == "test_op"
        assert event.context == {"key": "value"}
        assert event.performance == {"duration_ms": 100}
        assert event.error_info == {"type": "ValueError", "message": "test error"}
        assert event.state_change == {
            "field": "status",
            "before": "old",
            "after": "new",
        }
        assert event.violation == {"rule": "test_rule"}
        assert event.integration == {"service": "test_service"}
        assert event.authentication == {"type": "oauth"}
        assert event.behavior == {"type": "anomaly"}

    def test_event_is_frozen(self):
        """Test that EventSerializer is immutable."""
        event = EventSerializer(
            event_type="test",
            timestamp="2023-01-01T12:00:00Z",
            source={"file": "test.py", "line": 1},
        )

        # Should not be able to modify
        try:
            event.event_type = "modified"
            assert False, "Should not be able to modify frozen struct"
        except AttributeError:
            pass  # Expected


class TestCreateEventStruct:
    def test_create_minimal_struct(self):
        """Test creating minimal event struct."""
        event = create_event_struct(
            event_type="test",
            timestamp="2023-01-01T12:00:00Z",
            source={"file": "test.py", "line": 1},
        )
        assert isinstance(event, EventSerializer)
        assert event.event_type == "test"
        assert event.timestamp == "2023-01-01T12:00:00Z"
        assert event.source == {"file": "test.py", "line": 1}

    def test_create_struct_with_kwargs(self):
        """Test creating event struct with additional kwargs."""
        event = create_event_struct(
            event_type="test",
            timestamp="2023-01-01T12:00:00Z",
            source={"file": "test.py", "line": 1},
            operation="test_op",
            context={"key": "value"},
            entity={"type": "user", "id": "123"},
        )
        assert event.operation == "test_op"
        assert event.context == {"key": "value"}
        assert event.entity == {"type": "user", "id": "123"}

    def test_create_struct_filters_none_values(self):
        """Test that None values are filtered out."""
        event = create_event_struct(
            event_type="test",
            timestamp="2023-01-01T12:00:00Z",
            source={"file": "test.py", "line": 1},
            operation=None,
            context={"key": "value"},
            entity=None,
        )
        assert event.operation is None
        assert event.context == {"key": "value"}
        assert event.entity is None


class TestDefaultSerializer:
    def test_serialize_datetime(self):
        """Test serializing datetime objects."""
        dt = datetime(2023, 1, 1, 12, 0, 0)
        result = default_serializer(dt)
        assert result == "2023-01-01T12:00:00Z"

    def test_serialize_other_objects(self):
        """Test serializing other objects."""

        class TestClass:
            def __str__(self):
                return "test_object"

        obj = TestClass()
        result = default_serializer(obj)
        assert result == "test_object"

    def test_serialize_string(self):
        """Test serializing strings."""
        result = default_serializer("test_string")
        assert result == "test_string"

    def test_serialize_number(self):
        """Test serializing numbers."""
        result = default_serializer(42)
        assert result == "42"


class TestLLMLoggerEncoders:
    """Test encoder selection in LLMLogger."""

    def test_json_encoder_selection(self):
        """Test FastJSONEncoder selection."""
        from io import StringIO
        from lmlog import LLMLogger
        from lmlog.core.serializers import FastJSONEncoder
        
        output = StringIO()
        logger = LLMLogger(
            output=output,
            encoder="json",  # Select FastJSONEncoder
            async_processing=False,
        )

        # Verify encoder type
        assert isinstance(logger._encoder, FastJSONEncoder)
