import pytest
from modelpath.optimizer import ModelBenchmark, choose_model


def test_quality_constrained_routing():
    rows = [ModelBenchmark("fast", "json", .75, 80, 500),
            ModelBenchmark("good", "json", .96, 120, 1000),
            ModelBenchmark("slow", "json", .99, 600, 900)]
    assert choose_model(rows, task="json", min_accuracy=.95).name == "good"
    assert choose_model(rows, task="json", min_accuracy=.95, memory_budget_mb=950).name == "slow"


def test_no_silent_quality_fallback():
    with pytest.raises(ValueError):
        choose_model([ModelBenchmark("a", "code", .70, 30, 100)], task="code", min_accuracy=.9)


def test_benchmark_validation():
    with pytest.raises(ValueError):
        ModelBenchmark("x", "json", 1.2, 20, 100)
