from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional, Any

@dataclass
class Sample:
    dataset: str
    image_id: str
    split: str
    t1_path: str
    t2_path: str
    references: list[str]
    change_flag: Optional[bool] = None
    change_mask: Optional[str] = None
    semantic_t1: Optional[str] = None
    semantic_t2: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    def to_dict(self): return asdict(self)

@dataclass
class Event:
    object: Optional[str] = None
    action: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None
    location: Optional[str] = None
    quantity: Optional[str] = None
    exact_count: Optional[int] = None
    change_state: Optional[str] = None
    negated: bool = False
    confidence: float = 1.0
    text: str = ""
    def to_dict(self): return asdict(self)
