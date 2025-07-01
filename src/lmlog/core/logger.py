"""
Enhanced logger with rule-based intelligence features.
"""

from typing import Any, Dict, Optional, Union, TextIO
from pathlib import Path

from .base_logger import LLMLogger
from ..intelligence.rules import RuleBasedClassifier, EventClassification
from ..intelligence.aggregation import PatternBasedAggregator
from ..intelligence.cost_aware import CostController, CostBudget
from ..intelligence.sampling import Sampler


class LMLogger(LLMLogger):
    """
    Enhanced LMLogger with fast, rule-based intelligence features.

    Features:
    - Rule-based event classification (sub-millisecond performance)
    - Pattern-based log aggregation for noise reduction
    - Cost-aware controls with budget management
    - Predictable behavior with no ML dependencies
    """

    __slots__ = (
        "_classifier",
        "_aggregator", 
        "_cost_controller",
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
        async_processing: bool = False,
        encoder: str = "msgspec",
        max_events_per_second: int = 1000,
        buffer_size: int = 0,
        auto_flush: bool = True,
        enable_classification: bool = True,
        enable_aggregation: bool = True,
        enable_cost_awareness: bool = False,
        cost_budget: Optional[CostBudget] = None,
        aggregation_threshold: int = 5,
        max_patterns: int = 1000,
    ):
        """
        Initialize enhanced LMLogger.

        Args:
            output: Output destination
            enabled: Whether logging is enabled
            global_context: Global context for all events
            sampler: Sampling strategy
            async_processing: Enable async processing (default False)
            encoder: Encoder type ("msgspec" or "json")
            max_events_per_second: Target events per second
            buffer_size: Maximum number of events to buffer
            auto_flush: Whether to auto-flush the buffer
            enable_classification: Enable rule-based event classification
            enable_aggregation: Enable pattern-based aggregation
            enable_cost_awareness: Enable cost controls
            cost_budget: Cost budget configuration
            aggregation_threshold: Minimum events to trigger aggregation
            max_patterns: Maximum aggregation patterns to track
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

        # Initialize components
        self._classifier: Optional[RuleBasedClassifier] = None
        self._aggregator: Optional[PatternBasedAggregator] = None
        self._cost_controller: Optional[CostController] = None

        if enable_classification:
            self._classifier = RuleBasedClassifier()

        if enable_aggregation:
            self._aggregator = PatternBasedAggregator(
                aggregation_threshold=aggregation_threshold,
                max_patterns=max_patterns,
            )

        if enable_cost_awareness and cost_budget:
            self._cost_controller = CostController(cost_budget)

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
        Log an event with intelligence processing.

        Args:
            event_type: Type of event
            level: Log level
            entity_type: Type of entity involved
            entity_id: ID of entity involved
            context: Additional context
            **kwargs: Additional event data
        """
        if not self._enabled:
            return

        # Build event dictionary
        event = {
            "event_type": event_type,
            "level": level.upper(),
            **kwargs,
        }

        if entity_type:
            event["entity_type"] = entity_type
        if entity_id:
            event["entity_id"] = entity_id
        if context:
            event.update(context)

        # Apply intelligence processing
        classification = None
        priority_level = 3  # Default medium priority

        if self._enable_classification and self._classifier:
            classification = self._classifier.classify_event(event)
            priority_level = classification.priority.value
            
            # Enrich event with classification data
            event["classified_type"] = classification.event_type.value
            event["priority"] = classification.priority.value
            event["confidence"] = classification.confidence
            event["suggested_sampling_rate"] = classification.suggested_sampling_rate

        # Check cost controls
        if self._enable_cost_awareness and self._cost_controller:
            if not self._cost_controller.should_log_event(event, priority_level):
                return  # Event throttled due to budget constraints

        # Check aggregation
        should_aggregate = False
        if self._enable_aggregation and self._aggregator:
            should_aggregate = self._aggregator.should_aggregate(event)
            
            if should_aggregate:
                # Log aggregated event instead
                aggregated_data = self._aggregator.get_aggregated_event(event)
                if aggregated_data:
                    # Remove conflicting keys
                    clean_data = {k: v for k, v in aggregated_data.items() 
                                 if k not in ['event_type', 'level']}
                    super().log_event(
                        event_type="aggregated_events",
                        level="info",
                        **clean_data
                    )
                return

        # Log the original event
        super().log_event(
            event_type=event_type,
            level=level,
            entity_type=entity_type,
            entity_id=entity_id,
            context=context,
            **kwargs,
        )

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

    def get_cost_metrics(self) -> Optional[Dict[str, Any]]:
        """Get cost controller metrics."""
        if self._cost_controller:
            return self._cost_controller.get_statistics()
        return None

    def enable_feature(self, feature: str) -> None:
        """
        Enable an intelligence feature.

        Args:
            feature: Feature to enable ("classification", "aggregation", "cost_awareness")
        """
        if feature == "classification" and not self._classifier:
            self._enable_classification = True
            self._classifier = RuleBasedClassifier()
        elif feature == "aggregation" and not self._aggregator:
            self._enable_aggregation = True
            self._aggregator = PatternBasedAggregator()
        elif feature == "cost_awareness" and not self._cost_controller:
            self._enable_cost_awareness = True
            # Need budget to enable cost awareness
            default_budget = CostBudget(
                max_events_per_second=1000,
                max_daily_events=100000,
                alert_threshold=0.8
            )
            self._cost_controller = CostController(default_budget)

    def disable_feature(self, feature: str) -> None:
        """
        Disable an intelligence feature.

        Args:
            feature: Feature to disable ("classification", "aggregation", "cost_awareness")
        """
        if feature == "classification":
            self._enable_classification = False
            self._classifier = None
        elif feature == "aggregation":
            self._enable_aggregation = False
            self._aggregator = None
        elif feature == "cost_awareness":
            self._enable_cost_awareness = False
            self._cost_controller = None

    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of all intelligence features and their performance."""
        summary = {
            "classification_enabled": self._enable_classification,
            "aggregation_enabled": self._enable_aggregation,
            "cost_awareness_enabled": self._enable_cost_awareness,
        }

        if self._classifier:
            summary["classification_stats"] = self._classifier.get_statistics()

        if self._aggregator:
            summary["aggregation_stats"] = self._aggregator.get_statistics()

        if self._cost_controller:
            summary["cost_stats"] = self._cost_controller.get_statistics()

        return summary