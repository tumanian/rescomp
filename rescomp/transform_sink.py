"""The TransformSink: a Sink that stores a few rows per sequence (the masked mean, the last token) instead of
every token, for probes. Same hooks, tokens, annotations, store, flush and S3 as the Sink.

    sink = TransformSink("runs/probe", layers=bind(model, [20, 28]), transform="mean+last")
    with sink:
        for batch in loader:
            model(**batch)                         # the prompt only; generation steps are skipped
    r = rc.open("runs/probe"); r.pooled(28, "mean")    # [N, d] fp32, in sequence order

Per hooked layer the transform turns the activation [B, T, d] into [B, k, d] fp32 on the device (k = 2 for
"mean+last"); the rows accumulate per layer until `chunk_seqs` sequences are in, then go through the codec's
encoder and into the store as one chunk of shape [N, k, d] per layer. The default codec is "exact" (an fp16
cast): the last token of a bf16 model is stored bit for bit, a mean is rounded to fp16; codec="fp32" keeps the
means exactly too. The site becomes "<site>.<transform name>" and the header's extra records the row names
(`extra["pooled"] = {"name": ..., "rows": [...]}`), which `Reader.pooled` uses.

A KV-cache continuation call (a generation step) is skipped: the rows describe the prompt the model was given,
and the sink keeps no state across calls. Without the model's cache signal the Sink's rule applies: a one-token
call after a call of the same batch size counts as a continuation.
"""
import torch

from .bf16_wrapper import check_channels, choose_channels
from .sink import Sink
from .transforms import transform as as_transform


class TransformSink(Sink):
    def __init__(self, stem, layers, codec="exact", *, transform="mean+last", chunk_seqs=4096, site="resid_post",
                 extra=None, **kw):
        self.transform = as_transform(transform)
        self.chunk_seqs = int(chunk_seqs)
        ex = dict(extra or {})
        ex["pooled"] = self.transform.to_dict()
        super().__init__(stem, layers, codec, site=f"{site}.{self.transform.name}", extra=ex, **kw)
        self._acc = {l: [] for l in self.order}      # per layer: pooled rows [B, k, d] on the device, call by call
        self._acc_ids, self._acc_n = [], 0
        self.n_skipped = 0

    # -- extension point 1: pool the layer instead of encoding it
    def _layer_step(self, i, l, x, call):
        if call["cont"]:
            return                                     # the model says this is a cached step: nothing to pool
        if i == 0:
            m = call["mask"]
            if m is not None and m.shape[-1] != x.shape[1]:
                m = m[..., -x.shape[1]:]                 # a generation step without the cache signal: the mask covers the whole sequence so far
            call["mask_on"] = None if m is None else m.to(x.device)
        call.setdefault("pooled", {})[l] = self.transform(x, call["mask_on"])

    # -- extension point 2: accumulate; a chunk is emitted when enough sequences are in
    def _call_step(self, call, cont):
        if cont or call["cont"]:
            self.n_skipped += 1
            return
        seq_ids, _ = self._register(call, False)
        for l in self.order:
            self._acc[l].append(call["pooled"][l])
        self._acc_ids += seq_ids
        self._acc_n += len(seq_ids)
        if self._acc_n >= self.chunk_seqs:
            self._emit()

    def _emit(self):
        """Encode the accumulated rows (one [N, k, d] tensor per layer) and buffer them as one chunk."""
        if self._acc_n == 0:
            return
        acts = [torch.cat(self._acc[l], 0) for l in self.order]
        N, k, d = acts[0].shape
        if self.enc is None:
            self._codec_for(acts, d)
        self.enc.reset()
        streams, err = [], []
        for i, x in enumerate(acts):
            st, xhat = self.enc.encode_layer(i, x, reconstruct=self.fidelity)
            if self.fidelity:
                e = x - xhat
                err.append(torch.stack([e.pow(2).sum(), x.pow(2).sum(), e.abs().max()]))
            streams.append(st)
        self.store.note_raw(N * k * d * 2 * len(self.order), d)
        self._buffer.append((list(self._acc_ids), [0] * N, streams, torch.stack(err) if self.fidelity else None))
        self._buffered += sum(t.numel() * t.element_size() for st in streams for t in st.values())
        self._acc = {l: [] for l in self.order}
        self._acc_ids, self._acc_n = [], 0
        if self._buffered >= self.buffer_bytes:
            self.flush()

    def _codec_for(self, acts, d):
        """First chunk: resolve the spec and, for a lossy codec that keeps outlier channels, choose them on
        this chunk's rows (there is no per-layer hook to choose them in)."""
        c = self.codec
        spec = c.resolve(d)
        if spec.outliers:
            if c.channels is None:
                c.channels = choose_channels(torch.stack(acts), spec, source="first_chunk")
            check_channels(c.channels, spec, len(self.order), d)
        self.enc = c.encoder(len(self.order))

    def _write_buffer(self):
        self._emit()                                   # flush and close also write what is still accumulating
        super()._write_buffer()

    def stats(self):
        s = super().stats()
        s.update(skipped_calls=self.n_skipped, pending_sequences=self._acc_n, rows=list(self.transform.rows))
        return s
