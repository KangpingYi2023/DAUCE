"""
JsonCommunicator: dotted-key parameter store (epochs, learning rate, factor,
skew ratio, the fixed sample random_seed, ...).

Backed by the "params" subtree of the shared ExperimentContext document, so it
lives in the same file as the other communicators. Public API is unchanged.
"""

from update_utils.end2end_utils.context import ExperimentContext


class JsonCommunicator:
    def __init__(self):
        self.ctx = ExperimentContext()

    def get_all_in_str(self):
        return self.ctx.params_as_str()

    def erase_all(self):
        self.ctx.erase_params()

    def get(self, key_path):
        return self.ctx.get_param(key_path)

    def get_with_default(self, key_path, default):
        """Return the parameter, or ``default`` when it was never set.

        Lets a component declare its own defaults and still be overridable from a
        config, without every caller having to null-check.
        """
        value = self.ctx.get_param(key_path)
        return default if value is None else value

    def set(self, key_path, value):
        self.ctx.set_param(key_path, value)

    def update(self, key_path, value):
        # Only update when the key already exists, mirroring the original.
        if self.ctx.get_param(key_path) is not None:
            self.ctx.set_param(key_path, value)


if __name__ == "__main__":
    print(JsonCommunicator().get_all_in_str())
    JsonCommunicator().erase_all()
    print(JsonCommunicator().get_all_in_str())
