import json
import subprocess
import sys
from pathlib import Path

from traffic_signs.robustness.corruptions import NAMES, SEVERITIES


def test_summary_requires_complete_conditions(tmp_path):
    root = tmp_path / "runs"
    configs = {"cnn": ("cnn", False, False, False),
               "frozen_resnet18": ("resnet18", True, True, False),
               "finetuned_resnet18": ("resnet18", True, False, False),
               "robust_resnet18": ("resnet18", True, False, True)}
    for model, (architecture, pretrained, frozen, robust) in configs.items():
        for seed in (2026, 2027, 2028):
            run = root / model / f"seed_{seed}"
            (run / "test").mkdir(parents=True)
            (run / "robustness_test").mkdir()
            (run / "experiment.json").write_text(json.dumps({
                "config": {"model": architecture, "pretrained": pretrained, "frozen": frozen,
                           "robust_augmentation": robust, "seed": seed, "output_dir": str(run)},
                "split_path_sha256": "same"}))
            (run / "test/metrics.json").write_text(json.dumps({
                "metrics": {"accuracy": 0.9, "macro_f1": 0.8, "ece": 0.1}}))
            records = [{"corruption": name, "severity": severity, "accuracy": 0.8,
                        "seed": 2026, "split": "test"}
                       for name in NAMES for severity in SEVERITIES]
            records.append({"corruption": "clean", "severity": 0, "accuracy": 0.9,
                            "seed": 2026, "split": "test"})
            (run / "robustness_test/results.json").write_text(json.dumps(records))
    output = tmp_path / "report"
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_study.py"
    command = [sys.executable, str(script), "--root", str(root), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    result = json.loads((output / "final_summary.json").read_text())
    assert result["cnn"]["robustness"]["blur:1"]["change_from_clean"]["mean"] < 0
    broken = root / "cnn/seed_2026/robustness_test/results.json"
    broken.write_text("[]")
    assert subprocess.run(command, check=False, capture_output=True).returncode != 0
