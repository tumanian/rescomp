"""TransformerLens: the binding (hook points are modules; the site is blocks.<l>.hook_resid_post)."""
from .bindings import ModelBinding


class TransformerLensBinding(ModelBinding):
    loader = "transformer_lens"

    def blocks(self):
        return {l: blk.hook_resid_post for l, blk in enumerate(self.model.blocks)}

    def hook_name(self, layer):
        return {"transformer_lens": f"blocks.{layer}.hook_resid_post"}

    def info(self, tokenizer=None, model_id=None, revision=None):
        cfg = self.model.cfg
        return {"id": model_id or getattr(cfg, "model_name", None), "loader": self.loader, "n_layers": cfg.n_layers,
                "d_model": cfg.d_model, "dtype": str(cfg.dtype).removeprefix("torch."),
                "config": {"n_heads": cfg.n_heads, "d_head": cfg.d_head, "d_mlp": cfg.d_mlp, "d_vocab": cfg.d_vocab,
                           "n_ctx": cfg.n_ctx, "act_fn": cfg.act_fn}}
