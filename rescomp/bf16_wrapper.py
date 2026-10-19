"""What the quantizer does not touch: outlier channels and sink rows, kept exactly in bf16.

Why. The residual stream of a transformer has a handful of "massive" channels whose values are hundreds of times
larger than everything else (up to 3,000 on GPT-2, 15,000 on Qwen3-8B, in a few fixed channel positions). Any
quantizer that picks its scale from the largest value in a token or a group is dominated by them: the scale
becomes huge and every ordinary value collapses onto a few codes. The fix is simple and was the single largest
fidelity lever in the study (error divided by 2-4): take those channels out before quantizing, store them exactly
as bf16, and put zeros in their place so the quantizer sees only ordinary values. The same treatment applies to
the first few tokens of a sequence (`+sink<s>`): the attention-sink rows (the BOS row on Gemma, for instance)
carry an outsized norm and are stored exactly.

How the channels are chosen ("calibration" in the earlier study). Per stored layer, the n channels with the most
energy in what the quantizer is about to see: the activation itself at a keyframe, the delta against the previous
layer at a delta layer. Massive channels are a fixed property of a model, the same on every input, so one look
is enough: the Sink measures each layer in its own hook on the first forward call (`choose_layer_channels`),
`transcode` and the tests measure a sample stack (`choose_channels`). The chosen ids go into the file header and
the decoder reads them from there. Nothing else is learned or fitted.
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class OutlierChannels:
    """Per stored layer, the channel ids kept in bf16: ids is [L, n] int64, or None when the spec keeps none."""
    ids: Optional[torch.Tensor] = None
    source: str = "sample"

    def to_dict(self):
        return {"source": self.source, "ids": None if self.ids is None else self.ids.tolist()}

    @staticmethod
    def from_dict(d):
        ids = d.get("ids")
        return OutlierChannels(None if ids is None else torch.tensor(ids, dtype=torch.int64), d.get("source", "sample"))

    @staticmethod
    def placeholder(n_layers, n):
        """Rows to be filled layer by layer during the first call (unique ids, so validation passes meanwhile)."""
        return OutlierChannels(torch.arange(n).repeat(n_layers, 1), source="first_call")


def check_channels(channels, spec, n_layers, d=None):
    """The record matches the spec (n per layer) and, when d is known, holds valid distinct channel ids."""
    if spec.outliers == 0:
        return
    if channels is None or channels.ids is None:
        raise ValueError(f"spec {spec} keeps outlier channels in bf16 but none were chosen (rescomp.choose_channels)")
    if tuple(channels.ids.shape) != (n_layers, spec.outliers):
        raise ValueError(f"outlier channels are {tuple(channels.ids.shape)}, need ({n_layers}, {spec.outliers})")
    if d is not None:
        o = channels.ids
        if o.numel() and (o.min() < 0 or o.max() >= d):
            raise ValueError(f"outlier channel ids must be in [0, {d})")
        if any(len(set(row)) != len(row) for row in o.tolist()):
            raise ValueError("outlier channel ids repeat within a layer")


@torch.no_grad()
def choose_layer_channels(v, n):
    """The n highest-energy channels of one layer's values v [B, T, d] (what is about to be quantized), sorted."""
    energy = v.float().pow(2).sum(tuple(range(v.dim() - 1)))
    return energy.topk(n).indices.sort().values


@torch.no_grad()
def choose_channels(X, spec, source="sample"):
    """Offline: X [L, N, T, d], the L stored layers of a sample, in storage order -> OutlierChannels.
    Keyframes measure x, delta layers x - x_prev (open loop, close enough to pick channels); sink rows are
    stored exactly anyway and do not count."""
    from .spec import Spec
    spec = Spec.parse(spec)
    if spec.outliers == 0:
        return OutlierChannels(None, source)
    L, d = X.shape[0], X.shape[-1]
    if spec.outliers > d:
        raise ValueError(f"out{spec.outliers} > d={d}")
    if spec.sink >= X.shape[2]:
        raise ValueError(f"sink{spec.sink} leaves no tokens to choose channels on (T={X.shape[2]})")
    body = X[:, :, spec.sink:]
    rows = []
    for i in range(L):
        v = body[i] if spec.is_key(i) else body[i].float() - body[i - 1].float()
        rows.append(choose_layer_channels(v, spec.outliers).cpu())
    return OutlierChannels(torch.stack(rows), source)


def split(v, O, sink):
    """Take the sink rows and outlier channels out of v (in place, zeros left behind) -> (sink_rows, outliers) in bf16."""
    sink_rows = outl = None
    if sink:
        sink_rows = v[:, :sink, :].to(torch.bfloat16)
        v[:, :sink, :] = 0
    if O is not None:
        outl = v[..., O].to(torch.bfloat16)
        v[..., O] = 0
    return sink_rows, outl


def restore(vhat, O, sink, sink_rows, outl):
    """Put the exact rows and channels back into a reconstruction (in place)."""
    if O is not None:
        vhat[..., O] = outl.float()
    if sink:
        vhat[:, :sink, :] = sink_rows.float()
    return vhat
