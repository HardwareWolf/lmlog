import json
import tempfile
from pathlib import Path


from lmlog import LLMLoggerConfig


class TestLLMLoggerConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = LLMLoggerConfig()

        assert config.output == "llm_log.jsonl"
        assert config.enabled is True
        assert config.buffer_size == 0
        assert config.auto_flush is True
        assert config.global_context is None

    def test_config_with_parameters(self):
        """Test configuration with custom parameters."""
        config = LLMLoggerConfig(
            output="custom.log",
            enabled=False,
            buffer_size=10,
            auto_flush=False,
            global_context={"app": "test"},
        )

        assert config.output == "custom.log"
        assert config.enabled is False
        assert config.buffer_size == 10
        assert config.auto_flush is False
        assert config.global_context == {"app": "test"}

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        data = {
            "output": "dict.log",
            "enabled": True,
            "buffer_size": 5,
            "global_context": {"env": "test"},
        }

        config = LLMLoggerConfig.from_dict(data)

        assert config.output == "dict.log"
        assert config.buffer_size == 5
        assert config.global_context == {"env": "test"}

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = LLMLoggerConfig(
            output="test.log", buffer_size=3, global_context={"version": "1.0"}
        )

        data = config.to_dict()

        assert data["output"] == "test.log"
        assert data["buffer_size"] == 3
        assert data["global_context"] == {"version": "1.0"}
        assert data["enabled"] is True
        assert data["auto_flush"] is True

    def test_from_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "output": "file.log",
            "enabled": False,
            "buffer_size": 7,
            "auto_flush": False,
            "global_context": {"source": "file"},
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(config_data, f)
            temp_path = f.name

        config = LLMLoggerConfig.from_file(temp_path)

        assert config.output == "file.log"
        assert config.enabled is False
        assert config.buffer_size == 7
        assert config.auto_flush is False
        assert config.global_context == {"source": "file"}

        Path(temp_path).unlink()

    def test_to_file(self):
        """Test saving configuration to JSON file."""
        config = LLMLoggerConfig(
            output="save.log", buffer_size=12, global_context={"saved": True}
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        config.to_file(temp_path)

        with open(temp_path, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["output"] == "save.log"
        assert loaded_data["buffer_size"] == 12
        assert loaded_data["global_context"] == {"saved": True}

        Path(temp_path).unlink()

    def test_from_file_with_path_object(self):
        """Test loading configuration using Path object."""
        config_data = {"output": "path.log", "buffer_size": 2}

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(config_data, f)
            temp_path = Path(f.name)

        config = LLMLoggerConfig.from_file(temp_path)

        assert config.output == "path.log"
        assert config.buffer_size == 2

        temp_path.unlink()

    def test_roundtrip_file_operations(self):
        """Test saving and loading configuration maintains data integrity."""
        original_config = LLMLoggerConfig(
            output="roundtrip.log",
            enabled=False,
            buffer_size=15,
            auto_flush=False,
            global_context={"test": "roundtrip", "number": 42},
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        original_config.to_file(temp_path)
        loaded_config = LLMLoggerConfig.from_file(temp_path)

        assert original_config.output == loaded_config.output
        assert original_config.enabled == loaded_config.enabled
        assert original_config.buffer_size == loaded_config.buffer_size
        assert original_config.auto_flush == loaded_config.auto_flush
        assert original_config.global_context == loaded_config.global_context

        Path(temp_path).unlink()
