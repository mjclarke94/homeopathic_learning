"""Tests for stage infrastructure."""

from pathlib import Path
from typing import Any
import tempfile

import pytest

from subliminal_distillation.experiment.stage import (
    Stage,
    StageOutput,
    get_registered_stages,
    get_stage_class,
    register_stage,
)


class TestStageOutput:
    """Tests for StageOutput base class."""

    def test_create_output(self):
        """Create a custom output class."""

        class MyOutput(StageOutput):
            path: str
            count: int

        output = MyOutput(path="/tmp/test", count=42)
        assert output.path == "/tmp/test"
        assert output.count == 42

    def test_output_serialization(self):
        """Output serializes to JSON."""

        class MyOutput(StageOutput):
            path: str
            count: int

        output = MyOutput(path="/tmp/test", count=42)
        json_str = output.model_dump_json()
        # Check that the JSON contains the expected values
        assert "/tmp/test" in json_str
        assert "42" in json_str

    def test_output_with_path(self):
        """Output handles Path objects."""

        class MyOutput(StageOutput):
            output_path: Path
            name: str

        output = MyOutput(output_path=Path("/tmp/test"), name="test")
        json_str = output.model_dump_json()
        # Path should be serialized as string
        assert "/tmp/test" in json_str


class TestStageProtocol:
    """Tests for Stage base class."""

    def test_create_stage(self):
        """Create a minimal stage implementation."""

        class TestOutput(StageOutput):
            result: str

        class TestStage(Stage):
            name = "test_stage"
            depends_on = []
            Output = TestOutput

            def get_relevant_config(self, cfg):
                return {"key": "value"}

            def run(self, cfg, output_path, deps):
                pass

        stage = TestStage()
        assert stage.name == "test_stage"
        assert stage.depends_on == []

    def test_stage_with_dependencies(self):
        """Stage with dependencies."""

        class TestOutput(StageOutput):
            result: str

        class ChildStage(Stage):
            name = "child_stage"
            depends_on = ["parent_stage"]
            Output = TestOutput

            def get_relevant_config(self, cfg):
                return {}

            def run(self, cfg, output_path, deps):
                pass

        stage = ChildStage()
        assert "parent_stage" in stage.depends_on

    def test_write_and_read_manifest(self):
        """Write manifest and read output."""

        class TestOutput(StageOutput):
            result: str
            count: int

        class TestStage(Stage):
            name = "test_manifest"
            depends_on = []
            Output = TestOutput

            def get_relevant_config(self, cfg):
                return {}

            def run(self, cfg, output_path, deps):
                self._write_manifest(self.Output(result="success", count=42))

        with tempfile.TemporaryDirectory() as tmpdir:
            stage = TestStage()
            stage._output_path = Path(tmpdir)

            # Write manifest
            stage.run(None, Path(tmpdir), {})

            # Read output
            output = stage.output
            assert output.result == "success"
            assert output.count == 42

    def test_is_complete(self):
        """Check stage completion status."""

        class TestOutput(StageOutput):
            done: bool

        class TestStage(Stage):
            name = "test_complete"
            Output = TestOutput

            def get_relevant_config(self, cfg):
                return {}

            def run(self, cfg, output_path, deps):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            stage = TestStage()
            stage._output_path = Path(tmpdir)

            # Not complete before manifest exists
            assert not stage.is_complete()

            # Create manifest
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text('{"done": true}')

            # Now complete
            assert stage.is_complete()


class TestStageRegistry:
    """Tests for stage registration."""

    def test_register_stage(self):
        """Register a stage class."""

        class UniqueOutput(StageOutput):
            value: str

        @register_stage
        class UniqueTestStage(Stage):
            name = "unique_test_stage_for_registry"
            Output = UniqueOutput

            def get_relevant_config(self, cfg):
                return {}

            def run(self, cfg, output_path, deps):
                pass

        # Should be in registry
        registered = get_registered_stages()
        assert "unique_test_stage_for_registry" in registered
        assert registered["unique_test_stage_for_registry"] == UniqueTestStage

    def test_get_stage_class(self):
        """Get stage class by name."""

        class AnotherOutput(StageOutput):
            value: str

        @register_stage
        class AnotherUniqueStage(Stage):
            name = "another_unique_stage"
            Output = AnotherOutput

            def get_relevant_config(self, cfg):
                return {}

            def run(self, cfg, output_path, deps):
                pass

        cls = get_stage_class("another_unique_stage")
        assert cls == AnotherUniqueStage

    def test_get_nonexistent_stage(self):
        """Get nonexistent stage raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            get_stage_class("nonexistent_stage_xyz")

    def test_register_without_name_raises(self):
        """Registering stage without name raises ValueError."""
        with pytest.raises(ValueError, match="must define 'name'"):

            @register_stage
            class BadStage:
                pass
