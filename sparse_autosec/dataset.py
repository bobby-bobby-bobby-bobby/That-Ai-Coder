from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class TrainingSample:
    code: str
    label: int
    signature: str
    source: str


class OpenSourceDatasetLoader:
    """Loads training samples from open-source vulnerability datasets.

    Primary source: https://github.com/secureIT-project/CVEfixes
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def clone_cvefixes_repo(self) -> Path:
        dst = self.workspace / "CVEfixes"
        if not dst.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/secureIT-project/CVEfixes.git", str(dst)],
                check=True,
                capture_output=True,
                text=True,
            )
        return dst

    def load_from_csv(self, csv_file: Path, code_field: str, label_field: str, signature_field: str) -> List[TrainingSample]:
        items: List[TrainingSample] = []
        with csv_file.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                code = row.get(code_field, "")
                if not code:
                    continue
                label = int(row.get(label_field, "0"))
                signature = row.get(signature_field, "unknown")
                items.append(TrainingSample(code=code, label=label, signature=signature, source=str(csv_file)))
        return items

    def load_jsonl(self, path: Path) -> List[TrainingSample]:
        out: List[TrainingSample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            out.append(
                TrainingSample(
                    code=obj.get("code", ""),
                    label=int(obj.get("label", 0)),
                    signature=obj.get("signature", "unknown"),
                    source=str(path),
                )
            )
        return [item for item in out if item.code.strip()]

    def build_bootstrap_samples(self) -> List[TrainingSample]:
        return [
            TrainingSample(code="def run(user):\n    return eval(user)", label=1, signature="py_eval_user_input", source="bootstrap"),
            TrainingSample(code="def run(user):\n    return int(user) + 1", label=0, signature="safe_input_parse", source="bootstrap"),
            TrainingSample(code="import pickle\ndef run(buf):\n    return pickle.loads(buf)", label=1, signature="unsafe_deserialization", source="bootstrap"),
            TrainingSample(code="import json\ndef run(buf):\n    return json.loads(buf)", label=0, signature="safe_deserialization", source="bootstrap"),
        ]

    @staticmethod
    def split(samples: List[TrainingSample], train_ratio: float = 0.8) -> Dict[str, List[TrainingSample]]:
        cut = max(1, int(len(samples) * train_ratio))
        return {"train": samples[:cut], "valid": samples[cut:]}

    @staticmethod
    def as_task_text(sample: TrainingSample) -> str:
        return f"scan signature={sample.signature} code={sample.code[:200]}"

    @staticmethod
    def class_balance(samples: Iterable[TrainingSample]) -> Dict[int, int]:
        out = {0: 0, 1: 0}
        for sample in samples:
            out[sample.label] = out.get(sample.label, 0) + 1
        return out
