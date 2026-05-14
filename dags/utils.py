from pathlib import Path
import yaml

def load_config(path) -> dict:
    p = Path(path)
    
    if not p.exists():
        raise

    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)