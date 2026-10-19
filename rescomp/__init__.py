"""rescomp: compress the residual-stream activations of a running model and keep them, with their metadata.

    import rescomp as rc
    sink = rc.Sink("runs/x", layers=rc.hf_layers(model, [8, 9, 10, 11], tok), spec="balanced")
    with sink:
        model(**batch)                        # the model runs untouched; one forward call = one chunk
    r = rc.open("runs/x")
    r.read(9, seqs=[0, 1])                    # each sequence whole, fp32: the encoder's own reconstruction
"""
from ._version import __version__
from .spec import Spec
from .bf16_wrapper import OutlierChannels, choose_channels
from .codec import Encoder, Decoder, encode, decode
from .store import Store, Reader, transcode, is_v2
from .sink import Sink, Layers
from .regimes import REGIMES, resolve as resolve_spec
from .tl import tl_layers


def open(stem, verify=True, bucket=None, prefix=None, s3=None):
    """Open a stored capture by its stem: a local path, an s3://bucket/prefix/name URI, or a name with bucket=
    and prefix= (<stem>.meta.parquet + <stem>.NNNNN.rsc)."""
    return Reader(stem, verify=verify, bucket=bucket, prefix=prefix, s3=s3)


def hf_layers(model, layers, tokenizer=None, model_id=None, revision=None):
    """Modules to hook for a Sink on a Hugging Face causal LM (imports transformers lazily)."""
    from .hf import hf_layers as _hf
    return _hf(model, layers, tokenizer, model_id, revision)


__all__ = ["Spec", "OutlierChannels", "choose_channels", "Encoder", "Decoder", "encode", "decode", "Store", "Reader", "transcode",
           "is_v2", "open", "Sink", "Layers", "hf_layers", "tl_layers", "REGIMES", "resolve_spec", "__version__"]
