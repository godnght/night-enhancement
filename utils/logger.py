from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class JsonlLogger:
    def __init__(self, log_file: str) -> None:
        self.path = Path(log_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, values: Dict) -> None:
        payload = {"time": datetime.now().isoformat(), **values}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
