"""Transforms: one layer's activations [B, T, d] -> a few rows per sequence [B, k, d], for probes that need a
vector per sequence rather than one per token.

    tf = transform("mean+last")          # rows ["mean", "last"]; also "mean", "last"
    rows = tf(x, mask)                   # x [B, T, d] any float dtype, mask [B, T] (1 = real token) or None
                                         # -> [B, 2, d] fp32, on x's device

`Mean` is the attention-mask-weighted mean over the real tokens (a row with no real token gives zeros); `Last` is
the last real token (the last position where the mask is 1, so left and right padding both work); `Stacked`
puts several transforms' rows side by side. Each is a registered Stage of kind "transform", so
`stage("transform", "mean")` builds one too; the row names go into the file header.
"""
import torch

from .stage import Stage, register, stage


class Transform(Stage):
    kind = "transform"
    rows: list                              # the names of the k rows, in order

    @classmethod
    def matches(cls, name):
        return name == cls.name

    def to_dict(self):
        return {"name": self.name, "rows": list(self.rows)}

    @property
    def k(self):
        return len(self.rows)

    def __call__(self, x, mask=None):
        """x [B, T, d], mask [B, T] or None (all tokens real) -> [B, k, d] fp32 on x's device."""
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, d], got {tuple(x.shape)}")
        if mask is not None:
            mask = torch.as_tensor(mask).to(x.device)
            if tuple(mask.shape) != tuple(x.shape[:2]):
                raise ValueError(f"mask {tuple(mask.shape)} does not match x {tuple(x.shape)}")
        return self.apply(x.float(), mask)

    def apply(self, x, mask):
        raise NotImplementedError


@register
class Mean(Transform):
    name = "mean"
    rows = ["mean"]

    def __init__(self, name="mean"):
        super().__init__(name)

    def apply(self, x, mask):
        if mask is None:
            return x.mean(1, keepdim=True)
        m = mask.to(x.dtype)
        return (x * m[..., None]).sum(1, keepdim=True) / m.sum(1).clamp(min=1)[:, None, None]


@register
class Last(Transform):
    name = "last"
    rows = ["last"]

    def __init__(self, name="last"):
        super().__init__(name)

    def apply(self, x, mask):
        B, T, d = x.shape
        if mask is None:
            return x[:, T - 1 : T]
        last = T - 1 - mask.flip(1).to(torch.int64).argmax(1)       # the last position where the mask is 1
        return x[torch.arange(B, device=x.device), last][:, None]


class Stacked(Transform):
    """Several transforms' rows side by side; its name joins theirs with '+'."""

    def __init__(self, parts):
        super().__init__("+".join(p.name for p in parts))
        self.parts = list(parts)
        self.rows = [r for p in parts for r in p.rows]
        if len(set(self.rows)) != len(self.rows):
            raise ValueError(f"duplicate row names in {self.name!r}")

    @classmethod
    def matches(cls, name):
        return False

    def apply(self, x, mask):
        return torch.cat([p.apply(x, mask) for p in self.parts], 1)


def transform(name):
    """A Transform from its name: "mean", "last", or several joined with '+' ("mean+last")."""
    if isinstance(name, Transform):
        return name
    parts = [stage("transform", p.strip()) for p in str(name).split("+")]
    return parts[0] if len(parts) == 1 else Stacked(parts)
