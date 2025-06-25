import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class LLMLoggerConfig:
    """Configuration for LLMLogger."""

    output: str = "llm_log.jsonl"
    enabled: bool = True
    buffer_size: int = 0
    auto_flush: bool = True
    global_context: Optional[Dict[str, Any]] = None

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "LLMLoggerConfig":
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMLoggerConfig":
        """Create configuration from dictionary."""
        return cls(**data)

    def to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
