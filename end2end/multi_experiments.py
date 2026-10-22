import os
import sys
import time
import yaml
from itertools import product
from pathlib import Path

sys.path.append("./")
from update_utils import path_util
from update_utils.end2end_utils.script_runner import PythonScriptRunner
from update_utils.end2end_utils.json_communicator import JsonCommunicator
from update_utils.end2end_utils.context import ENV_CONTEXT_PATH

ALLOWED_MODELS = ["naru", "face", "transformer"]

def filter_params_dict(cur_model: str, params_dict: dict) -> dict:
    unwanted_models = [model for model in ALLOWED_MODELS if model != cur_model]
    params_dict = params_dict.copy()  # To prevent modifying the original parameters

    # Delete parameters unrelated to the current model from params_dict
    for key in list(params_dict.keys()):
        # If the key starts with the name of other models, delete it
        if any(key.startswith(model) for model in unwanted_models):
            params_dict.pop(key)

    return params_dict


def run_multi_experiments(
        script_path: Path,
        args_dict: dict,
        params_dict: dict,
        check_experiments_order: bool = False
):
    # Iterate over each combination of arguments
    all_args_combinations = product(*args_dict.values())
    for combination in all_args_combinations:
        arg_dict = dict(zip(args_dict.keys(), combination))

        # Detector x updater pairing. When the config pins drift_test (E4's 2x2
        # cross needs every detector with every updater), keep it; otherwise
        # fall back to the historical coupling.
        if "drift_test" not in arg_dict or arg_dict["drift_test"] is None:
            if arg_dict["model_update"] == "update":
                arg_dict["drift_test"] = "ddup"
            elif arg_dict["model_update"] == "adapt":
                arg_dict["drift_test"] = "sauce"
            elif arg_dict["model_update"] == "none":
                arg_dict["drift_test"] = "none"

        # Per-dataset default batch size; an explicit update_size in the config
        # wins, so E4/E5/E6 can control the conformal block count.
        if "update_size" not in arg_dict or arg_dict["update_size"] is None:
            default_update_size = {
                "census": 8000,
                "forest": 100000,
                "bjaq": 80000,
                "power": 400000,
            }
            arg_dict["update_size"] = default_update_size.get(arg_dict["dataset"], 20000)

        # Iterate over each combination of params
        print("Plan to run experiment(s) with arguments: \t", arg_dict)
        # Filter on the *current* combination's model, not the grid's first
        # value: a multi-model grid would otherwise drop transformer.* params
        # from every transformer cell and crash define_workloads.
        filtered_params: dict = filter_params_dict(cur_model=arg_dict["model"], params_dict=params_dict)  # filter irrelevant params
        all_params_combinations = product(*[[(k, v) for v in values] for k, values in filtered_params.items()])
        for params in all_params_combinations:
            # Give this sub-experiment its own context file so its state never
            # collides with another run's. Every subprocess in the chain
            # (experiment.py and its Naru/FACE children) inherits DAUCE_CONTEXT
            # via os.environ, so they all read/write this same file. Unique paths
            # are what let independent experiments run in parallel later.
            context_rel = (
                f"./end2end/communicate/context_{os.getpid()}_{time.time_ns()}.json"
            )
            os.environ[ENV_CONTEXT_PATH] = context_rel

            # Set params
            for key, value in params:
                JsonCommunicator().set(key, value)
            # JsonCommunicator().print_all()

            # Run experiment
            print(f"> Run 1 sub-experiment with params: \t", params)
            if not check_experiments_order:
                runner = PythonScriptRunner(script_path=script_path, args=arg_dict)
                runner.run_script()

            # The child has finished (run_script blocks), so drop this run's
            # context file rather than leaving per-run files to accumulate.
            context_abs = path_util.get_absolute_path(context_rel)
            if os.path.isfile(context_abs):
                os.remove(context_abs)


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument(
        "--config", default="./end2end/configs/args.yaml",
        help="experiment grid YAML (args_combination + params_combination)",
    )
    _cli_args = _parser.parse_args()

    # self-defined parameters
    args_yaml_path = path_util.get_absolute_path(_cli_args.config)

    # e2e parameters for each dataset
    # args_yaml_path = path_util.get_absolute_path("./end2end/configs/e2e-bjaq+naru.yaml")
    # args_yaml_path = path_util.get_absolute_path("./end2end/configs/e2e-bjaq+transformer.yaml")
    # args_yaml_path = path_util.get_absolute_path("./end2end/configs/e2e-census+naru.yaml")
    # args_yaml_path = path_util.get_absolute_path("./end2end/configs/e2e-census+transformer.yaml")
    # args_yaml_path = path_util.get_absolute_path("./end2end/configs/e2e-forest+naru.yaml")
    # args_yaml_path = path_util.get_absolute_path("./end2end/configs/e2e-forest+transformer.yaml")

    # Read experiment parameters
    with open(args_yaml_path, 'r', encoding='utf-8') as file:
        args_from_yaml = yaml.safe_load(file)  # Use safe_load to read the YAML file
    args_combination = args_from_yaml["args_combination"]
    params_combination = args_from_yaml["params_combination"]

    # Change params
    # args_combination["query_seed"]=[i for i in range(5, 20)]
    # params_combination["random_seed"]=[i for i in range(0, 5)]#+[1226]

    # Run multiple sets of experiments
    # check_order_only = True  # If True: Only prints the order of experiments, does not run them
    check_order_only = False  # If False: Runs the experiments

    run_multi_experiments(
        script_path=path_util.get_absolute_path("./end2end/experiment.py"),  # Path to experiment script
        args_dict=args_combination,
        params_dict=params_combination,
        check_experiments_order=check_order_only
    )
