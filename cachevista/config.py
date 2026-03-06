from pathlib import Path
import yaml

_root = Path(__file__).parent.parent


def load():
    with open(_root / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)
