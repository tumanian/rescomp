import pytest

from rescomp import Spec


@pytest.mark.parametrize("name", ["fp16", "bf16", "int4_g384+layerK4+out12", "int4_g512+layerK4+out32+key=bf16",
                                  "uabs0.1+out8", "mxfp4+layerK2+sink1", "nvfp4"])
def test_parse_name_roundtrip(name):
    s = Spec.parse(name)
    assert s.name == name
    assert Spec.parse(s.name) == s
    assert Spec.parse(s) is s


def test_fields():
    s = Spec.parse("int4_g64+layerK4+out8+sink2+key=bf16")
    assert s.quant == "int4_g64" and s.K == 4 and s.outliers == 8 and s.sink == 2 and s.key_quant == "bf16"
    assert [s.is_key(i) for i in range(6)] == [True, False, False, False, True, False]
    assert s.quant_for(0) == "bf16" and s.quant_for(1) == "int4_g64" and s.quant_for(4) == "bf16"


def test_no_delta_means_every_layer_is_key():
    s = Spec.parse("int8_tok")
    assert all(s.is_key(i) for i in range(10))


def test_redundant_key_dropped_from_name():
    assert Spec(quant="bf16", K=2, key_quant="bf16").name == "bf16+layerK2"


@pytest.mark.parametrize("bad", ["int3_g64", "int4", "fp64", "int4_g64+foo", "int4_g64+key=bf16", "uabs0", "uabs-1",
                                 "int4_g16+out2+out4", "bf16+layerK1", "int4_g16+layerK2+layerK4"])
def test_bad_specs(bad):
    with pytest.raises(ValueError):
        Spec.parse(bad)


def test_negative_fields():
    with pytest.raises(ValueError):
        Spec(quant="bf16", K=-1)
