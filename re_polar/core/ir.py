"""Program IR, per-input layer-execution programs.

Formalization from "Skip a Layer or Loop It? Learning Program-of-Layers in
LLMs" (arXiv:2606.06574): a program partitions the layer stack [0, D) into
contiguous segments of at most MAX_SEGMENT_LEN layers, each executed with one
op (keep / skip / repeat). Reimplemented from the paper, no PoLar code is
copied or imported (see NOTICE.md).

A Program expands to a flat layer-index execution path (`to_layer_path()`)
that re_polar.core.layer_engine.LayerEngine.apply_layer_rerouting() executes directly.
`params` carries op arguments (e.g. repeat count) so the schema can grow.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

MAX_SEGMENT_LEN = 4  # PoLar's validated constraint


class Op(str, Enum):
    KEEP = "keep"
    SKIP = "skip"
    REPEAT = "repeat"


@dataclass(frozen=True)
class Segment:
    start: int  # first layer index, inclusive
    end: int    # past-last layer index, exclusive
    op: Op
    params: Dict = field(default_factory=dict)  # REPEAT: {"times": k>=2}

    def __len__(self) -> int:
        return self.end - self.start

    @property
    def times(self) -> int:
        return self.params.get("times", 2 if self.op is Op.REPEAT else 1)

    def to_dict(self) -> Dict:
        return {"start": self.start, "end": self.end, "op": self.op.value, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: Dict) -> "Segment":
        return cls(start=d["start"], end=d["end"], op=Op(d["op"]), params=dict(d.get("params", {})))


@dataclass
class Program:
    num_layers: int
    segments: List[Segment]

    @classmethod
    def identity(cls, num_layers: int) -> "Program":
        segments = [
            Segment(start=i, end=min(i + MAX_SEGMENT_LEN, num_layers), op=Op.KEEP)
            for i in range(0, num_layers, MAX_SEGMENT_LEN)
        ]
        return cls(num_layers=num_layers, segments=segments)

    def to_layer_path(self) -> List[int]:
        """Flat layer-index execution path for LayerEngine.apply_layer_rerouting()."""
        path: List[int] = []
        for seg in self.segments:
            indices = list(range(seg.start, seg.end))
            if seg.op is Op.KEEP:
                path.extend(indices)
            elif seg.op is Op.SKIP:
                pass
            elif seg.op is Op.REPEAT:
                path.extend(indices * seg.times)
            else:
                raise ValueError(f"Unsupported op: {seg.op}")
        if not path:
            raise ValueError("Program skips all layers, empty execution path.")
        return path

    def is_identity(self) -> bool:
        return self.to_layer_path() == list(range(self.num_layers))

    def to_dict(self) -> Dict:
        return {"num_layers": self.num_layers, "segments": [s.to_dict() for s in self.segments]}

    @classmethod
    def from_dict(cls, d: Dict) -> "Program":
        return cls(num_layers=d["num_layers"], segments=[Segment.from_dict(s) for s in d["segments"]])
