"""The root of the stages a codec is made of: quantizers (rescomp.quant) and lossless backends (rescomp.lossless).

A stage has a kind, a name, a header record (`to_dict`) and a way to say which names it builds (`matches`).
Subclasses register themselves; `stage(kind, name, **options)` builds one. A new quantizer or compressor is one
class plus `@register`; nothing else in the library changes.
"""
from abc import ABC, abstractmethod

REGISTRY = {}                          # kind -> [stage classes], in registration order


def register(cls):
    REGISTRY.setdefault(cls.kind, []).append(cls)
    return cls


class Stage(ABC):
    kind: str                          # "quant" or "lossless"

    def __init__(self, name):
        self.name = name

    @classmethod
    @abstractmethod
    def matches(cls, name):
        """Whether this class builds the stage called `name`."""

    @abstractmethod
    def to_dict(self):
        """What the file header records about this stage."""

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


def stage(kind, name, **options):
    """Build the registered stage of `kind` that matches `name`."""
    for cls in REGISTRY.get(kind, []):
        if cls.matches(name):
            return cls(name, **options)
    raise ValueError(f"unknown {kind} stage {name!r}; have {names(kind)}")


def names(kind):
    """The registered classes' names or patterns, for error messages."""
    return [getattr(c, "name", None) or c.pattern.pattern for c in REGISTRY.get(kind, [])]
