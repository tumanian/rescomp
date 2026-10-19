"""rescomp: compress the residual-stream activations of a running model and keep them, with their metadata.

    import rescomp as rc
    sink = rc.Sink("runs/x", layers=rc.bind(model, [8, 9, 10, 11], tokenizer=tok), spec="balanced")
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
from .sink import Sink
from .hooks import Layers
from .bindings import ModelBinding, ModuleBinding, bind
from .regimes import REGIMES, resolve as resolve_spec


def open(stem, verify=True, bucket=None, prefix=None, s3=None):
    """Open a stored capture by its stem: a local path, an s3://bucket/prefix/name URI, or a name with bucket=
    and prefix= (<stem>.meta.parquet + <stem>.NNNNN.rsc)."""
    return Reader(stem, verify=verify, bucket=bucket, prefix=prefix, s3=s3)


__all__ = ["Spec", "OutlierChannels", "choose_channels", "Encoder", "Decoder", "encode", "decode", "Store", "Reader", "transcode",
           "is_v2", "open", "Sink", "Layers", "ModelBinding", "ModuleBinding", "bind", "REGIMES", "resolve_spec", "__version__"]
