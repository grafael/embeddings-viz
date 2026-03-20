"""Project paths and config.yaml loading."""

from pathlib import Path

import yaml

PROJECT_DIR = Path.cwd()
CONFIG_PATH = PROJECT_DIR / "config.yaml"
VOCAB_FILE = PROJECT_DIR / "vocab.txt"
VOCAB_CACHE_DIR = PROJECT_DIR / "vocab"
VOCAB_CACHE_DIR.mkdir(exist_ok=True)

with open(CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

_raw_models = _config.get("models", ["all-MiniLM-L6-v2"])
AVAILABLE_MODELS = []
for _m in _raw_models:
    if isinstance(_m, str):
        AVAILABLE_MODELS.append({"name": _m, "type": "embedding"})
    elif isinstance(_m, dict):
        AVAILABLE_MODELS.append({"name": _m["name"], "type": _m.get("type", "embedding")})

MODEL_TYPE_MAP = {m["name"]: m["type"] for m in AVAILABLE_MODELS}
MODEL_NAMES = [m["name"] for m in AVAILABLE_MODELS]
DEFAULT_MODEL = _config.get("default_model", MODEL_NAMES[0])
