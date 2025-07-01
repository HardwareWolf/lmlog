"""Intelligence functionality for fast, predictable log processing."""

# Rule-based classification (replaces ML-based classification)
from .rules import RuleBasedClassifier, EventType, EventPriority, EventClassification

# Pattern-based aggregation
from .aggregation import PatternBasedAggregator, AggregatedEvent

# Cost-aware controls
from .cost_aware import (
    CostController,
    CostBudget,
    CostMetrics,
    CostTier,
    Throttler,
    CostSampler,
)

# Keep sampling for backward compatibility
from .sampling import Sampler, ProbabilisticSampler, AlwaysSampler, NeverSampler

__all__ = [
    # Rule-based classification
    "RuleBasedClassifier",
    "EventType",
    "EventPriority", 
    "EventClassification",
    
    # Pattern-based aggregation
    "PatternBasedAggregator",
    "AggregatedEvent",
    
    # Cost controls
    "CostController",
    "CostBudget",
    "CostMetrics",
    "CostTier",
    "Throttler",
    "CostSampler",
    
    # Sampling (backward compatibility)
    "Sampler",
    "ProbabilisticSampler",
    "AlwaysSampler",
    "NeverSampler",
]