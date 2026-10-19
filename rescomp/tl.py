"""TransformerLens: which modules a Sink hooks (hook points are modules) and what the header should say."""
from .hooks import Layers, extract_tokens_generic


def tl_layers(model, layers, model_id=None):
    cfg = model.cfg
    layers = sorted(int(l) for l in layers)
    if any(not 0 <= l < cfg.n_layers for l in layers):
        raise ValueError(f"layers {layers} outside [0, {cfg.n_layers})")
    info = {"id": model_id or getattr(cfg, "model_name", None), "loader": "transformer_lens", "n_layers": cfg.n_layers,
            "d_model": cfg.d_model, "dtype": str(cfg.dtype).removeprefix("torch."),
            "config": {"n_heads": cfg.n_heads, "d_head": cfg.d_head, "d_mlp": cfg.d_mlp, "d_vocab": cfg.d_vocab,
                       "n_ctx": cfg.n_ctx, "act_fn": cfg.act_fn}}
    return Layers(modules={l: model.blocks[l].hook_resid_post for l in layers}, root=model,
                  hook_names={l: {"transformer_lens": f"blocks.{l}.hook_resid_post"} for l in layers},
                  model_info=info, extract=extract_tokens_generic, n_layers=cfg.n_layers)
