"""Four named regimes, from exact to compact. Each is one spec template whose group and outlier counts scale
with the model width d; a full spec string works anywhere a regime name does.

| Regime     | Spec                                        | Measured on eight models (RESULTS.md)            |
|------------|---------------------------------------------|--------------------------------------------------|
| exact      | fp16                                        | 1.5x, bit-exact for bf16 values in fp16's range  |
| faithful   | int8_tok+out8                               | 2.5x, <= 0.7% perplexity, per-layer KL ~ 0.001   |
| balanced   | int4_g{d/8}+layerK4+out{d/128}+key=bf16     | 2.4-3.0x, every 4th layer exact, +1-2% perplexity|
| compact    | int4_g{d/8}+layerK4+out{d/128}              | 5.0-5.2x, +4-7% perplexity (more on small models)|
"""
from .spec import Spec

REGIMES = {
    "exact": ("fp16", "1.5x; bit-exact for bf16 values within fp16's range (the Sink checks every call)"),
    "faithful": ("int8_tok+out8", "2.5x; <= 0.7% perplexity; per-layer KL about 0.001"),
    "balanced": ("int4_g{G8}+layerK4+out{O2}+key=bf16", "2.4-3.0x; every 4th layer exact; +1-2% perplexity"),
    "compact": ("int4_g{G8}+layerK4+out{O2}", "5.0-5.2x; +4-7% perplexity, more on small models"),
}


def is_regime(name):
    return isinstance(name, str) and name in REGIMES


def needs_width(spec):
    """True if `spec` is a regime name or a template with {G8}/{O2}/{G}/{O} placeholders."""
    return is_regime(spec) or (isinstance(spec, str) and "{" in spec)


def resolve(spec, d=None):
    """A regime name, a template with {G8}/{O2}/{G}/{O}, or a spec string -> Spec. Width-dependent sizes need d:
    {G8} = d/8, {O2} = max(8, d/128) (the balanced/compact proportions), {G} = d/2, {O} = max(8, d/64)."""
    if isinstance(spec, Spec):
        return spec
    template = REGIMES[spec][0] if is_regime(spec) else spec
    if "{" in template:
        if d is None:
            raise ValueError(f"{spec!r} needs the model width d to resolve its group and outlier counts")
        g8, g2 = max(1, d // 8), max(1, d // 2)
        template = (template.replace("{G8}", str(g8)).replace("{G}", str(g2))
                            .replace("{O2}", str(max(8, d // 128))).replace("{O}", str(max(8, d // 64))))
    return Spec.parse(template)
