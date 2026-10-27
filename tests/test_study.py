import json
from pathlib import Path

import pytest

from scripts.run_full_experiments import commands
from traffic_signs.evaluation.analysis import aggregate
from traffic_signs.evaluation.metrics import wilson_interval


def test_wilson_interval():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=0.0001)
    assert high == pytest.approx(0.5962, abs=0.0001)
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_study_commands_are_unique_and_staged(tmp_path):
    training = list(commands("train", tmp_path))
    assert len(training) == 12
    assert len({path for path, _ in training}) == 12
    assert len(list(commands("test", tmp_path))) == 24
    assert all("--split" not in command for _, command in training)


def test_aggregate_rejects_incompatible_runs(tmp_path: Path):
    paths = []
    for seed in (2026, 2027, 2028):
        run = tmp_path / str(seed)
        (run / "test").mkdir(parents=True)
        (run / "experiment.json").write_text(json.dumps({
            "config": {"seed": seed, "model": "cnn", "output_dir": str(run)},
            "split_path_sha256": "same"}))
        path = run / "test/metrics.json"
        path.write_text(json.dumps({"metrics": {"accuracy": seed - 2025,
                                                "macro_f1": 0.5, "ece": 0.1}}))
        paths.append(path)
    result = aggregate(paths)
    assert result["accuracy"]["mean"] == 2
    assert result["accuracy"]["std"] == 1
    metadata = json.loads((tmp_path / "2028/experiment.json").read_text())
    metadata["config"]["model"] = "resnet18"
    (tmp_path / "2028/experiment.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Incompatible"):
        aggregate(paths)
