"""
LMLogger module.
"""

from typing import Any, Dict, Optional, Union, TextIO, List
from pathlib import Path

from .base_logger import LLMLogger
from ..intelligence.classification import IntelligentEventClassifier
from ..intelligence.legacy_ml_cost_aware import CostAwareLogger, CostBudget, CostMetrics
from ..intelligence.sampling import Sampler


class LegacyLMLogger(LLMLogger):
    """
    Enhanced LMLogger with intelligent features from Phase 2.

    Additional features:
    - ML-based event classification
    - Pattern detection and aggregation
    - Cost-aware logging with budgets
    - Intelligent sampling based on classification
    """

    __slots__ = (
        "_classifier",
        "_aggregator",
        "_cost_manager",
        "_enable_classification",
        "_enable_aggregation",
        "_enable_cost_awareness",
    )

    def __init__(
        self,
        output: Union[str, Path, TextIO] = "llm_log.jsonl",
        enabled: bool = True,
        global_context: Optional[Dict[str, Any]] = None,
        sampler: Optional[Sampler] = None,
        async_processing: bool = True,
        encoder: str = "msgspec",
        max_events_per_second: int = 1000,
        buffer_size: int = 0,
        auto_flush: bool = True,
        enable_classification: bool = True,
        enable_aggregation: bool = True,
        enable_cost_awareness: bool = True,
        cost_budget: Optional[CostBudget] = None,
        aggregation_window: int = 60,
        aggregation_threshold: float = 0.8,
    ):
        """
        Initialize enhanced LMLogger.

        Args:
            output: Output destination
            enabled: Whether logging is enabled
            global_context: Global context for all events
            sampler: Sampling strategy
            async_processing: Enable async processing
            encoder: Encoder type ("msgspec" or "json")
            max_events_per_second: Target events per second
            buffer_size: Maximum number of events to buffer
            auto_flush: Whether to auto-flush the buffer
            enable_classification: Enable ML-based event classification
            enable_aggregation: Enable pattern detection and aggregation
            enable_cost_awareness: Enable cost-aware features
            cost_budget: Cost budget configuration
            aggregation_window: Window size for aggregation (seconds)
            aggregation_threshold: Similarity threshold for aggregation
        """
        super().__init__(
            output=output,
            enabled=enabled,
            global_context=global_context,
            sampler=sampler,
            async_processing=async_processing,
            encoder=encoder,
            max_events_per_second=max_events_per_second,
            buffer_size=buffer_size,
            auto_flush=auto_flush,
        )

        self._enable_classification = enable_classification
        self._enable_aggregation = enable_aggregation
        self._enable_cost_awareness = enable_cost_awareness

        self._classifier: Optional[IntelligentEventClassifier] = None
        self._aggregator = None  # Optional[SmartAggregator] = None
        self._cost_manager: Optional[CostAwareLogger] = None

        if enable_classification:
            self._classifier = IntelligentEventClassifier()

        # Aggregation disabled in legacy logger
        # if enable_aggregation:
        #     self._aggregator = SmartAggregator(
        #         window_seconds=aggregation_window,
        #         similarity_threshold=aggregation_threshold,
        #     )

        if enable_cost_awareness:
            if cost_budget is None:
                cost_budget = CostBudget(
                    max_daily_bytes=1024 * 1024 * 1024,
                    max_events_per_second=10000,
                )
            self._cost_manager = CostAwareLogger(cost_budget)

    def log_event(
        self,
        event_type: str,
        level: str = "info",
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Log a structured event with intelligent processing.

        Args:
            event_type: Type of event
            level: Log level
            entity_type: Type of entity involved
            entity_id: ID of entity involved
            context: Additional context
            **kwargs: Additional event fields
        """
        if not self._enabled:
            return

        context = context or {}

        event_dict = self._create_base_event()
        event_dict.update(
            {
                "event_type": event_type,
                "level": level,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "context": context,
                **kwargs,
            }
        )

        if self._classifier and self._enable_classification:
            classification = self._classifier.classify_event(event_dict)

            event_dict["classification"] = {
                "type": classification.event_type.value,
                "priority": classification.priority.value,
                "confidence": classification.confidence,
                "anomaly_score": classification.anomaly_score,
            }

            priority = classification.priority.value / 5.0
        else:
            priority = 0.5

        if self._cost_manager and self._enable_cost_awareness:
            if not self._cost_manager.should_log(event_dict, priority):
                with self._lock:
                    self._stats["events_sampled_out"] += 1
                return

        # Aggregation disabled in legacy logger
        # if self._aggregator and self._enable_aggregation:
        #     aggregated = self._aggregator.process_event(event_dict)
        #     if aggregated:
        #         self._write_aggregated_event(aggregated)
        #         return

        if not self._should_sample_event(event_type, level, context):
            return

        if self._cost_manager and self._enable_cost_awareness:
            self._cost_manager.process_event(event_dict)

        self._write_event(event_dict)
        with self._lock:
            self._stats["events_logged"] += 1

    def _write_aggregated_event(self, aggregated) -> None:
        """Write an aggregated event."""
        event = self._create_base_event()

        sample_events = []
        for sample in aggregated.sample_events[:3]:
            if isinstance(sample, dict):
                sanitized = {}
                for key, value in sample.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        sanitized[key] = value
                    elif isinstance(value, dict) and key != "aggregation":
                        sanitized[key] = {
                            k: v
                            for k, v in value.items()
                            if isinstance(v, (str, int, float, bool, type(None)))
                        }
                sample_events.append(sanitized)

        event.update(
            {
                "event_type": "aggregated_event",
                "level": "info",
                "aggregation": {
                    "pattern_id": aggregated.pattern_id,
                    "original_type": aggregated.event_type,
                    "pattern": aggregated.pattern,
                    "count": aggregated.count,
                    "time_window": aggregated.time_window,
                    "statistics": aggregated.statistics,
                    "variables": aggregated.variables,
                    "sample_events": sample_events,
                },
            }
        )

        if self._cost_manager and self._enable_cost_awareness:
            self._cost_manager.process_event(event)

        self._write_event(event)
        with self._lock:
            self._stats["events_logged"] += 1

    def get_classification_stats(self) -> Optional[Dict[str, Any]]:
        """Get classification statistics."""
        if self._classifier:
            return self._classifier.get_statistics()
        return None

    def get_aggregation_stats(self) -> Optional[Dict[str, Any]]:
        """Get aggregation statistics."""
        if self._aggregator:
            return self._aggregator.get_statistics()
        return None

    def get_cost_metrics(self) -> Optional[CostMetrics]:
        """Get cost metrics."""
        if self._cost_manager:
            return self._cost_manager.get_cost_metrics()
        return None

    def get_cost_forecast(self) -> Optional[Dict[str, Any]]:
        """Get cost forecast."""
        if self._cost_manager:
            return self._cost_manager.get_cost_forecast()
        return None

    def get_aggregated_events(self):
        """Get current aggregated events."""
        # Aggregation disabled in legacy logger
        return []

    def enable_feature(self, feature: str) -> None:
        """
        Enable a specific feature.

        Args:
            feature: Feature name ("classification", "aggregation", "cost_awareness")
        """
        if feature == "classification":
            self._enable_classification = True
            if not self._classifier:
                self._classifier = IntelligentEventClassifier()
        elif feature == "aggregation":
            # Aggregation disabled in legacy logger
            pass
        elif feature == "cost_awareness":
            self._enable_cost_awareness = True
            if not self._cost_manager:
                self._cost_manager = CostAwareLogger(
                    CostBudget(
                        max_daily_bytes=1024 * 1024 * 1024,
                        max_events_per_second=10000,
                    )
                )

    def disable_feature(self, feature: str) -> None:
        """
        Disable a specific feature.

        Args:
            feature: Feature name ("classification", "aggregation", "cost_awareness")
        """
        if feature == "classification":
            self._enable_classification = False
        elif feature == "aggregation":
            self._enable_aggregation = False
        elif feature == "cost_awareness":
            self._enable_cost_awareness = False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive logger statistics."""
        stats = super().get_stats()

        if self._classifier:
            stats["classification"] = self.get_classification_stats()

        if self._aggregator:
            stats["aggregation"] = self.get_aggregation_stats()

        if self._cost_manager:
            stats["cost_metrics"] = self.get_cost_metrics()
            stats["cost_forecast"] = self.get_cost_forecast()

        stats["features"] = {
            "classification": self._enable_classification,
            "aggregation": self._enable_aggregation,
            "cost_awareness": self._enable_cost_awareness,
        }

        return stats
