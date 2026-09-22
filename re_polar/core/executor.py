"""Program execution on top of the layer-execution engine (re_polar.core.layer_engine).

Adapter seam: ALL program execution goes through here, so an engine API
change touches only this file.
"""

from contextlib import contextmanager

from .layer_engine import LayerEngine
from .ir import Program
from .grammar import validate_program


class ProgramExecutor:
    """Applies Program layer paths to a LayerEngine's model, restoring after use."""

    def __init__(self, engine: LayerEngine):
        self.engine = engine

    @classmethod
    def from_model_id(cls, model_id: str) -> "ProgramExecutor":
        return cls(LayerEngine(model_id))

    @contextmanager
    def apply(self, program: Program):
        """Context manager: reroute layers per program, yield the model, restore."""
        validate_program(program)
        if program.num_layers != self.engine.num_layers:
            raise ValueError(
                f"Program is for {program.num_layers} layers, model has {self.engine.num_layers}."
            )
        self.engine.apply_layer_rerouting(program.to_layer_path())
        try:
            yield self.engine.model
        finally:
            self.engine.restore_original()
