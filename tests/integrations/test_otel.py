"""
Tests for OpenTelemetry integration.
"""

import pytest
from unittest.mock import Mock, patch

from lmlog.integrations.otel import (
    TraceContextExtractor,
    CorrelationContext,
    MetricGenerator,
    ResourceDetector,
    extract_trace_context,
    is_otel_available,
)

try:
    import opentelemetry  # noqa: F401

    OTEL_INSTALLED = True
except ImportError:
    OTEL_INSTALLED = False


class TestTraceContextExtractor:
    """Test trace context extractor."""

    def test_without_otel(self):
        """Test behavior without OpenTelemetry."""
        with patch("lmlog.integrations.otel.OTEL_AVAILABLE", False):
            extractor = TraceContextExtractor()
            assert not extractor.is_available()

            context = extractor.extract_context()
            assert context == {}

    @pytest.mark.skipif(not OTEL_INSTALLED, reason="OpenTelemetry not installed")
    @patch("lmlog.integrations.otel.OTEL_AVAILABLE", True)
    @patch("lmlog.integrations.otel.trace")
    @patch("lmlog.integrations.otel.baggage")
    def test_with_otel_no_span(self, mock_baggage, mock_trace):
        """Test with OpenTelemetry but no active span."""
        mock_trace.get_tracer.return_value = Mock()
        mock_trace.get_current_span.return_value = None
        mock_baggage.get_all.return_value = {}

        extractor = TraceContextExtractor()
        context = extractor.extract_context()

        assert context == {}

    @pytest.mark.skipif(not OTEL_INSTALLED, reason="OpenTelemetry not installed")
    @patch("lmlog.integrations.otel.OTEL_AVAILABLE", True)
    @patch("lmlog.integrations.otel.trace")
    @patch("lmlog.integrations.otel.baggage")
    def test_with_active_span(self, mock_baggage, mock_trace):
        """Test with active span."""
        mock_span_context = Mock()
        mock_span_context.trace_id = 0x123456789ABCDEF0123456789ABCDEF0
        mock_span_context.span_id = 0x123456789ABCDEF0
        mock_span_context.trace_flags = 1
        mock_span_context.trace_state = {}

        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_span.get_span_context.return_value = mock_span_context
        mock_span.name = "test_span"
        mock_span.kind.name = "INTERNAL"

        mock_trace.get_tracer.return_value = Mock()
        mock_trace.get_current_span.return_value = mock_span
        mock_baggage.get_all.return_value = {"key": "value"}

        extractor = TraceContextExtractor()
        context = extractor.extract_context()

        assert "trace_id" in context
        assert "span_id" in context
        assert context["baggage"] == {"key": "value"}

    @pytest.mark.skipif(not OTEL_INSTALLED, reason="OpenTelemetry not installed")
    @patch("lmlog.integrations.otel.OTEL_AVAILABLE", True)
    def test_start_span(self):
        """Test span creation."""
        mock_tracer = Mock()
        mock_span = Mock()

        # Create a proper context manager mock
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_span)
        context_manager.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span.return_value = context_manager

        extractor = TraceContextExtractor()
        extractor._tracer = mock_tracer

        with extractor.start_span("test_span", test_attr="value") as span:
            assert span == mock_span

        # Verify that set_attributes was called
        mock_span.set_attributes.assert_called_once_with({"test_attr": "value"})


class TestCorrelationContext:
    """Test correlation context."""

    def test_context_management(self):
        """Test context setting and getting."""
        context = CorrelationContext()

        context.set_context(user_id="123", session="abc")
        retrieved = context.get_context()

        assert retrieved["user_id"] == "123"
        assert retrieved["session"] == "abc"

    def test_context_clearing(self):
        """Test context clearing."""
        context = CorrelationContext()

        context.set_context(user_id="123")
        context.clear_context()

        retrieved = context.get_context()
        assert "user_id" not in retrieved

    def test_with_context(self):
        """Test with_context manager."""
        context = CorrelationContext()

        context.set_context(original="value")

        with context.with_context(temp="temporary"):
            retrieved = context.get_context()
            assert retrieved["temp"] == "temporary"
            assert retrieved["original"] == "value"

        retrieved = context.get_context()
        assert "temp" not in retrieved
        assert retrieved["original"] == "value"

    def test_extractor_integration(self):
        """Test extractor integration."""
        mock_extractor = Mock()
        mock_extractor.extract_context.return_value = {"trace_id": "123"}

        context = CorrelationContext()
        context.add_extractor(mock_extractor)

        retrieved = context.get_context()
        assert retrieved["trace_id"] == "123"


class TestMetricGenerator:
    """Test metric generator."""

    def test_without_otel(self):
        """Test behavior without OpenTelemetry."""
        with patch("lmlog.integrations.otel.OTEL_AVAILABLE", False):
            generator = MetricGenerator()

            generator.increment_counter("test_counter")
            generator.record_histogram("test_histogram", 1.0)
            generator.generate_from_event({"event_type": "test"})

    @patch("lmlog.integrations.otel.OTEL_AVAILABLE", True)
    def test_with_otel_no_metrics(self):
        """Test with OpenTelemetry but no metrics module."""
        with patch("lmlog.integrations.otel.MetricGenerator.__init__") as mock_init:
            mock_init.return_value = None

            generator = MetricGenerator.__new__(MetricGenerator)
            generator._enabled = False
            generator._meter = None
            generator._counters = {}
            generator._histograms = {}

            generator.increment_counter("test_counter")
            generator.record_histogram("test_histogram", 1.0)
            generator.generate_from_event({"event_type": "test"})

    @patch("lmlog.integrations.otel.OTEL_AVAILABLE", True)
    def test_counter_creation(self):
        """Test counter creation and use."""
        mock_meter = Mock()
        mock_counter = Mock()
        mock_meter.create_counter.return_value = mock_counter

        with patch("lmlog.integrations.otel.MetricGenerator.__init__") as mock_init:
            mock_init.return_value = None

            generator = MetricGenerator.__new__(MetricGenerator)
            generator._enabled = True
            generator._meter = mock_meter
            generator._counters = {}
            generator._histograms = {}

            generator.increment_counter("test_counter", 5, {"attr": "value"})

            mock_meter.create_counter.assert_called_once()
            mock_counter.add.assert_called_once_with(5, {"attr": "value"})

    @patch("lmlog.integrations.otel.OTEL_AVAILABLE", True)
    def test_event_metrics_generation(self):
        """Test event metrics generation."""
        mock_meter = Mock()
        mock_counter = Mock()
        mock_histogram = Mock()
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_histogram.return_value = mock_histogram

        with patch("lmlog.integrations.otel.MetricGenerator.__init__") as mock_init:
            mock_init.return_value = None

            generator = MetricGenerator.__new__(MetricGenerator)
            generator._enabled = True
            generator._meter = mock_meter
            generator._counters = {}
            generator._histograms = {}

            event = {
                "event_type": "performance_issue",
                "level": "warning",
                "context": {"duration_ms": 500, "operation": "database_query"},
            }

            generator.generate_from_event(event)

            assert mock_meter.create_counter.call_count >= 1
            assert mock_meter.create_histogram.call_count >= 1


class TestResourceDetector:
    """Test resource detector."""

    def test_basic_detection(self):
        """Test basic resource detection."""
        detector = ResourceDetector()
        resource = detector.get_resource_info()

        assert "service.name" in resource
        assert "telemetry.sdk.name" in resource
        assert "host.name" in resource
        assert "os.name" in resource
        assert "process.pid" in resource

    @patch.dict(
        "os.environ",
        {
            "OTEL_SERVICE_NAME": "test-service",
            "OTEL_SERVICE_VERSION": "1.0.0",
            "KUBERNETES_SERVICE_HOST": "k8s-api",
            "K8S_NAMESPACE": "test-ns",
            "K8S_CLUSTER_NAME": "test-cluster",
        },
    )
    def test_kubernetes_detection(self):
        """Test Kubernetes resource detection."""
        detector = ResourceDetector()
        resource = detector.get_resource_info()

        assert resource["service.name"] == "test-service"
        assert resource["service.version"] == "1.0.0"
        assert "k8s.namespace.name" in resource
        assert "k8s.cluster.name" in resource


class TestGlobalFunctions:
    """Test global functions."""

    def test_extract_trace_context(self):
        """Test extract_trace_context function."""
        context = extract_trace_context()
        assert isinstance(context, dict)

    def test_is_otel_available(self):
        """Test is_otel_available function."""
        result = is_otel_available()
        assert isinstance(result, bool)
