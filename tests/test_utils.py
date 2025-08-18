from gen.utils import set_seed, stable_hash

def test_seed_repeatability():
    a = set_seed(1234)
    b = set_seed(1234)
    assert a == b

def test_hash():
    assert stable_hash("abc") == stable_hash("abc")
    assert stable_hash("abc") != stable_hash("abd")
