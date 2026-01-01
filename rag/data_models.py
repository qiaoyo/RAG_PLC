from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PLCExample:
    instruction: str
    input: str
    output: str
    description: str
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    library_dependency: Optional[str] = ""
    iec_standard: Optional[str] = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any]
