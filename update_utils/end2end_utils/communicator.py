"""
Cross-process communicators for the end2end orchestrator.

These classes keep the exact public API that the model subprocesses
(Naru / FACE) and the orchestrator already call, but their backing store is now
a single JSON document (see context.ExperimentContext) instead of six separate
files. Call sites are therefore unchanged.
"""

from pathlib import Path
from typing import List

from update_utils import path_util
from update_utils.end2end_utils.context import ExperimentContext

# Top-level keys inside the shared context document.
KEY_MODEL_PATH = "model_path"
KEY_DATASET_PATH = "dataset_path"
KEY_RANDOM_SEED = "random_seed"
KEY_SPLIT_INDICES = "split_indices"
KEY_DRIFT = "drift"


class RandomSeedCommunicator:
    """The evolving per-workload seed (starts at 0, incremented via update())."""

    def __init__(self):
        self.ctx = ExperimentContext()

    def get(self) -> int:
        value = self.ctx.get(KEY_RANDOM_SEED, 0)
        print(f"RandomSeedCommunicator.get: {int(value)}")
        return int(value)

    def update(self):
        self.ctx.set(KEY_RANDOM_SEED, self.get() + 1)

    def set(self, new_seed: int):
        self.ctx.set(KEY_RANDOM_SEED, int(new_seed))


class _PathCommunicator:
    """Shared logic for path-valued fields (stored relative, returned absolute)."""

    def __init__(self, key: str, prompt: str):
        self.ctx = ExperimentContext()
        self.key = key
        self.prompt = prompt

    def get(self) -> Path:
        content = self.ctx.get(self.key)
        print(f"GET-{self.prompt}-PATH={content}")
        return path_util.get_absolute_path(content)

    def set(self, new_path: str):
        self.ctx.set(self.key, new_path)
        print(f"SET-{self.prompt}-PATH={new_path}")


class ModelPathCommunicator(_PathCommunicator):
    def __init__(self):
        super().__init__(key=KEY_MODEL_PATH, prompt="MODEL")


class DatasetPathCommunicator(_PathCommunicator):
    def __init__(self):
        super().__init__(key=KEY_DATASET_PATH, prompt="DATASET")


class DriftCommunicator:
    """
    Drift verdict, stored natively as a JSON bool (single table) or dict (stats,
    per-table). Storing it natively removes the old text-vs-pickle format hazard;
    get()/get_dict() still fail loudly on the wrong shape or on a cleared value.
    """

    def __init__(self):
        self.ctx = ExperimentContext()

    def get(self) -> bool:
        value = self.ctx.get(KEY_DRIFT)
        if isinstance(value, bool):
            return value
        raise ValueError(
            f"DriftCommunicator.get expected a bool but read {value!r}. A dict "
            f"means the stats path should call get_dict(); None means the file "
            f"was cleared and the child never wrote a fresh verdict."
        )

    def get_dict(self) -> dict:
        value = self.ctx.get(KEY_DRIFT)
        if isinstance(value, dict):
            return value
        raise ValueError(
            f"DriftCommunicator.get_dict expected a dict but read {value!r}."
        )

    def set(self, is_drift):
        if isinstance(is_drift, (bool, dict)):
            self.ctx.set(KEY_DRIFT, is_drift)
        else:
            raise TypeError(
                f"DriftCommunicator.set expects bool or dict, got {type(is_drift)}"
            )

    def clear(self):
        """
        Reset the drift verdict before a data-update workload runs. If the child
        crashes and never writes a fresh result, get()/get_dict() then fail loudly
        instead of returning a previous run's verdict.
        """
        self.ctx.set(KEY_DRIFT, None)


class SplitIndicesCommunicator:
    """Update-batch landmarks written by the data-update child, read by the
    model-update child."""

    def __init__(self):
        self.ctx = ExperimentContext()

    def get(self) -> List[int]:
        value = self.ctx.get(KEY_SPLIT_INDICES, [])
        return [int(item) for item in value]

    def set(self, array: list):
        self.ctx.set(KEY_SPLIT_INDICES, [int(item) for item in array])


if __name__ == "__main__":
    model_path_communicator = ModelPathCommunicator()
    dataset_path_communicator = DatasetPathCommunicator()

    model_path_communicator.set("./models/new-model-path.pt")
    print(model_path_communicator.get())

    dataset_path_communicator.set("./datasets/new-dataset-path.pt")
    print(dataset_path_communicator.get())
