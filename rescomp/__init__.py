"""rescomp: compress residual-stream activations and store them with the metadata interp work needs.

    import rescomp as rc
    spec = rc.Spec.parse("int4_g384+layerK4+out12")
    cal = rc.calibrate(x_sample, spec)                      # x_sample [L, N, T, d]
    with rc.Writer("acts.rsc", model=info, layers=[8, 9, 10, 11], spec=spec, calib=cal) as w:
        w.write(acts, tokens)                               # acts [L, B, T, d], tokens [B, T]; one call = one chunk
    r = rc.Reader("acts.rsc")
    r.read(layer=9, seqs=slice(0, 16))                      # [16, T, d] fp32, the encoder's own reconstruction
"""
from ._version import __version__
from .spec import Spec
from .codec import Calib, calibrate, Encoder, Decoder, encode, decode
from .container import Writer, Reader, transcode, FORMAT, VERSION
from .model import ModelAdapter, FunctionAdapter, TLAdapter
from .store import Store, Reader2, is_v2
from .sink import Sink, Layers
from .regimes import REGIMES, resolve as resolve_spec
from .model import tl_layers


def hf_layers(model, layers, tokenizer=None, model_id=None, revision=None):
    """Modules to hook for a Sink on a Hugging Face causal LM (imports transformers lazily)."""
    from .hf import hf_layers as _hf
    return _hf(model, layers, tokenizer, model_id, revision)


def open(path, verify=True):
    """Open a stored capture: a format-2 stem (<stem>.meta.parquet + blob files) or a single RSC1 file."""
    import os
    path = str(path)
    if is_v2(path):
        return Reader2(path, verify=verify)
    if os.path.isfile(path):
        return Reader(path, verify=verify)
    raise ValueError(f"{path}: neither a format-2 stem nor an RSC1 file")

__all__ = ["Spec", "Calib", "calibrate", "Encoder", "Decoder", "encode", "decode",
           "Writer", "Reader", "transcode", "FORMAT", "VERSION", "__version__",
           "ModelAdapter", "FunctionAdapter", "TLAdapter", "Store", "Reader2", "is_v2", "open",
           "Sink", "Layers", "hf_layers", "tl_layers", "REGIMES", "resolve_spec"]
