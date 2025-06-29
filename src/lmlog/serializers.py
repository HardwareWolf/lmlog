"""
High-performance serialization using modern Python 3.11+ libraries.
"""

import orjson
import msgspec
from typing import Any, Dict, Union
from datetime import datetime


class EventSerializer(msgspec.Struct, frozen=True):
    """
    High-performance event structure using msgspec.

    Provides 10-80x faster serialization than standard JSON
    while maintaining type safety and validation.
    """

    event_type: str
    timestamp: str
    source: Dict[str, Any]
    entity: Dict[str, Any] | None = None
    operation: str | None = None
    context: Dict[str, Any] | None = None
    performance: Dict[str, Any] | None = None
    error_info: Dict[str, Any] | None = None
    state_change: Dict[str, Any] | None = None
    violation: Dict[str, Any] | None = None
    integration: Dict[str, Any] | None = None
    authentication: Dict[str, Any] | None = None
    behavior: Dict[str, Any] | None = None


class FastJSONEncoder:
    """
    Ultra-fast JSON encoder using orjson.

    Provides significant performance improvements over standard json module:
    - 2-3x faster encoding
    - Built-in datetime serialization
    - Better memory efficiency
    """

    def encode(self, obj: Any) -> bytes:
        """Encode object to JSON bytes.

        Args:
            obj: The object to encode, can be any JSON-serializable type.
        Returns:
            bytes: The JSON-encoded bytes representation of the object.
        Raises:
            TypeError: If the object cannot be serialized to JSON.
        """
        return orjson.dumps(obj, option=orjson.OPT_UTC_Z | orjson.OPT_SERIALIZE_NUMPY)

    def encode_str(self, obj: Any) -> str:
        """Encode object to JSON string."""
        return orjson.dumps(
            obj, option=orjson.OPT_UTC_Z | orjson.OPT_SERIALIZE_NUMPY
        ).decode("utf-8")


class MsgSpecEncoder:
    """
    Ultra-high-performance structured encoder using msgspec.

    Provides schema validation with zero runtime overhead
    and 10-80x faster performance than alternatives.
    """

    def __init__(self):
        self._encoder = msgspec.json.Encoder()
        self._decoder = msgspec.json.Decoder(EventSerializer)

    def encode(self, event: Union[EventSerializer, Dict[str, Any]]) -> bytes:
        """Encode structured event to JSON bytes."""
        return self._encoder.encode(event)

    def encode_str(self, event: Union[EventSerializer, Dict[str, Any]]) -> str:
        """Encode structured event to JSON string."""
        return self.encode(event).decode("utf-8")

    def decode(self, data: Union[bytes, str]) -> EventSerializer:
        """Decode JSON to validated event structure."""
        return self._decoder.decode(data)


def create_event_struct(
    event_type: str, timestamp: str, source: Dict[str, Any], **kwargs
) -> EventSerializer:
    """
    Create a validated event structure.

    Uses msgspec for zero-cost validation and optimal performance.
    """
    return EventSerializer(
        event_type=event_type,
        timestamp=timestamp,
        source=source,
        **{k: v for k, v in kwargs.items() if v is not None},
    )


def default_serializer(obj: Any) -> Any:
    """Default serializer for complex types."""
    if isinstance(obj, datetime):
        return obj.isoformat() + "Z"
    return str(obj)
