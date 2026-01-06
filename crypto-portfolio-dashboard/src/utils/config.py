from __future__ import annotations
from pathlib import Path
import os, re, yaml
from typing import Any, Dict

_ENV_RE = re.compile(r"\$\{([^:}]+)(?::([^}]*))?\}")  # ${VAR} or ${VAR:default}

def _env_interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m):
            var, default = m.group(1), m.group(2)
            return os.getenv(var, default if default is not None else "")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _env_interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_env_interpolate(v) for v in value]
    return value

def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_settings() -> Dict[str, Any]:
    base = Path("config/settings.yaml")
    local = Path("config/settings.local.yaml")
    cfg = yaml.safe_load(base.read_text()) if base.exists() else {}
    if local.exists():
        cfg = _deep_update(cfg, yaml.safe_load(local.read_text()) or {})
    return _env_interpolate(cfg)


