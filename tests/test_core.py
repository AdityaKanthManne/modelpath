from modelpath.core import ModelSpec, classify, select_models, route, ModelUnavailable
import pytest

REGISTRY = [ModelSpec("extract", ("extraction",), priority=0), ModelSpec("code", ("coding",), priority=1), ModelSpec("reason", ("reasoning",), priority=2)]

def test_classification():
    assert classify("Extract fields as JSON") == "extraction"
    assert classify("Write Python code") == "coding"
    assert classify("Explain the result") == "reasoning"

def test_ordering():
    assert [m.name for m in select_models("coding", REGISTRY)] == ["code", "extract", "reason"]

def test_fallback():
    def fake(model, prompt):
        if model == "extract":
            raise ModelUnavailable("offline")
        return {"response": "valid", "eval_count": 2}
    result = route("Extract JSON", REGISTRY, generate=fake)
    assert result["model"] == "code"
    assert result["attempts"] == 2
    assert len(result["errors"]) == 1

def test_total_failure():
    with pytest.raises(ModelUnavailable):
        route("test", REGISTRY, generate=lambda *_: (_ for _ in ()).throw(ModelUnavailable("offline")))
