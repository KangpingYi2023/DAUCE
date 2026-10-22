"""
Single-file experiment context that replaces the six ad-hoc communication files
(model_path.txt, dataset_path.txt, random_seed.txt, split_indices.txt,
is_drift.txt, parameters.json) used to pass state between the orchestrator
(end2end/experiment.py) and the model subprocesses (Naru / FACE).

Why this exists
---------------
The orchestrator still launches each workload as a subprocess, so cross-process
state (drift verdict, split indices, evolving seed, per-run params) must be
persisted somewhere. Previously each piece lived in its own file at a fixed
path, which (a) let the text-format drift flag collide with the stats pickle
format, and (b) forced all experiments to share global files, preventing
parallel runs. This module consolidates everything into one structured JSON
document whose location is resolved from the DAUCE_CONTEXT environment variable.

Resolution rules
----------------
- If DAUCE_CONTEXT is set (the orchestrator sets a unique path per experiment
  run), every subprocess in the chain inherits it via os.environ and therefore
  reads/writes the same file. Unique paths let independent experiments run
  without clobbering each other's state.
- If DAUCE_CONTEXT is unset (a model script run standalone, e.g. from
  experiments.sh), it falls back to the fixed default path, preserving the old
  single-file behaviour.

Only one subprocess in a given experiment runs at a time, so the naive
read-modify-write below needs no locking.
"""

import json
import os
from pathlib import Path
from typing import Any

from update_utils import path_util

ENV_CONTEXT_PATH = "DAUCE_CONTEXT"
DEFAULT_CONTEXT_PATH = "./end2end/communicate/context.json"


def get_context_path() -> Path:
    """Resolve the context file path from the environment (or the default)."""
    relative = os.environ.get(ENV_CONTEXT_PATH, DEFAULT_CONTEXT_PATH)
    return path_util.get_absolute_path(relative)


class ExperimentContext:
    """
    Thin JSON-backed key/value store for one experiment run.

    Every accessor does a fresh load/save so that state written by one
    subprocess is visible to the next one. In the later single-process refactor
    this same object can be promoted to an in-memory instance without changing
    the communicator API layered on top of it.
    """

    def __init__(self, path: Path = None):
        self.path = Path(path) if path is not None else get_context_path()

    # ---- raw document I/O ------------------------------------------------
    def _load(self) -> dict:
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    # ---- top-level fields ------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self._load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        data = self._load()
        data[key] = value
        self._save(data)

    # ---- dotted params subtree (backs JsonCommunicator) ------------------
    def get_param(self, dotted_key: str) -> Any:
        node = self._load().get("params", {})
        for key in dotted_key.split("."):
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    def set_param(self, dotted_key: str, value: Any) -> None:
        data = self._load()
        params = data.setdefault("params", {})
        keys = dotted_key.split(".")
        node = params
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value
        self._save(data)

    def erase_params(self) -> None:
        data = self._load()
        data["params"] = {}
        self._save(data)

    def params_as_str(self) -> str:
        return str(self._load().get("params", {}))
