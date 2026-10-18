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

__all__ = ["Spec", "Calib", "calibrate", "Encoder", "Decoder", "encode", "decode",
           "Writer", "Reader", "transcode", "FORMAT", "VERSION", "__version__"]
