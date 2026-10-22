import argparse
import glob
import os
import shutil
import sys
import time

import pytz

sys.path.append("./")
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from end2end.workload import (
    QueryWorkload,
    DataUpdateWorkload,
    WorkloadGenerator,
    BaseWorkload,
    PythonScriptWorkload,
)
from update_utils import path_util, log_util
from update_utils.arg_util import add_common_arguments, ArgType
from update_utils.end2end_utils import communicator, log_parser
from update_utils.end2end_utils.json_communicator import JsonCommunicator
from update_utils.db_utils import init_pg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end experiment parameter parsing")

    # Add common arguments
    common_args: List[ArgType] = [
        ArgType.DATA_UPDATE,
        ArgType.DATASET,
        ArgType.DEBUG,
        ArgType.DRIFT_TEST,
        ArgType.MODEL_UPDATE,
        ArgType.END2END,
    ]
    add_common_arguments(parser, arg_types=common_args)

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    parser.add_argument("--num_workload", type=int, default=5, help="Number of workloads")

    parsed_args = parser.parse_args()

    return parsed_args


def validate_argument(args):
    # The historical pairing (update->ddup, adapt->sauce) is the default but no
    # longer enforced: the E4 2x2 cross-ablation runs every detector with every
    # updater. Only pairings that cannot work are rejected.
    if args.model_update == "none" and args.drift_test not in (None, "none"):
        sys.exit("Argument error: model_update 'none' requires drift_test 'none'.")

    # random_seed must be >= 0
    if args.random_seed < 0:
        sys.exit("Argument error: random_seed must be >= 0.")

    # num_workload must be >= 1
    if args.num_workload < 1:
        sys.exit("Argument error: num_workload must be >= 1.")


def prepare_end2end_model(dataset: str, model: str):
    """
    Prepare the end2end model: copy the original model to the end2end folder, overwrite if it exists
    """

    # Get the model path pattern based on the method name
    if model == "naru":
        model_dir = "./models/naru_origin_models"
        reg_model_name = f"origin-{dataset}*.pt"
    elif model == "face":
        model_dir = "./models/face_origin_models"
        reg_model_name = f"{dataset}*.t"
    elif model == "transformer":
        model_dir = "./models/transformer_origin_models"
        reg_model_name = f"*{dataset}*.pt"
    else:
        raise ValueError(f"Invalid model name: {model}")
    reg_model_path = f"{model_dir}/{reg_model_name}"  # Model path pattern

    # Get the file name of the original model
    abs_model_reg: Path = path_util.get_absolute_path(reg_model_path)
    model_paths = glob.glob(str(abs_model_reg))  # Pattern match result
    assert model_paths, "No matching model paths found."

    if dataset in ["stats"]:
        end2end_model_dir = f"./models/end2end/{model}/{dataset}"
        if not os.path.exists(end2end_model_dir):
            os.mkdir(end2end_model_dir)
        abs_end2end_model_dir = path_util.get_absolute_path(end2end_model_dir)

        for model_path in model_paths:
            src_model_filename = os.path.basename(model_path)  # Original model file name
            # Get the path of the original model
            src_model_path = f"{model_dir}/{src_model_filename}"
            abs_src_model_path = path_util.get_absolute_path(src_model_path)

            print(f"Source Model path: {src_model_path}")

            # Copy the original model to the end2end folder, overwrite if it exists
            abs_end2end_model_path = os.path.join(abs_end2end_model_dir, src_model_filename)  # Path to save the end2end model
            # abs_end2end_model_path = path_util.get_absolute_path(end2end_model_path)
            shutil.copy2(src=abs_src_model_path, dst=abs_end2end_model_path)

        return end2end_model_dir, abs_end2end_model_dir
    else:
        model_path = model_paths[0]  # Take the first match
        src_model_filename = os.path.basename(model_path)  # Original model file name
        
        # Get the path of the original model
        src_model_path = f"{model_dir}/{src_model_filename}"
        abs_src_model_path = path_util.get_absolute_path(src_model_path)

        print(f"Source Model path: {src_model_path}")

        # Copy the original model to the end2end folder, overwrite if it exists
        end2end_model_path = f"./models/end2end/{model}/{src_model_filename}"  # Path to save the end2end model
        abs_end2end_model_path = path_util.get_absolute_path(end2end_model_path)
        shutil.copy2(src=abs_src_model_path, dst=abs_end2end_model_path)

        return end2end_model_path, abs_end2end_model_path


def prepare_end2end_dataset(dataset: str, model: str):
    """
    Prepare the end2end dataset: copy the original dataset to the end2end folder, overwrite if it exists
    """

    npy_tail = ".npy"

    if dataset in ["stats"]:
        dataset_name=dataset
        csv_dataset_dir=f"./data/{dataset_name}"
        npy_dataset_dir=f"./data/{dataset_name}/numpy"
        end2end_dataset_dir=f"./data/{dataset_name}/end2end"
        abs_csv_dataset_dir = path_util.get_absolute_path(csv_dataset_dir)
        abs_npy_dataset_dir = path_util.get_absolute_path(npy_dataset_dir)
        abs_end2end_dataset_dir = path_util.get_absolute_path(end2end_dataset_dir)
        
        table_list = []
        for table_path in os.listdir(abs_npy_dataset_dir):
            npy_table_filename = os.path.basename(table_path)
            table_name = os.path.splitext(npy_table_filename)[0]
            abs_csv_dataset_path = os.path.join(abs_csv_dataset_dir, f"{table_name}.csv")
            abs_npy_dataset_path = os.path.join(abs_npy_dataset_dir, npy_table_filename)

            abs_end2end_table_dir = os.path.join(abs_end2end_dataset_dir, table_name)
            if not os.path.exists(abs_end2end_table_dir):
                os.mkdir(abs_end2end_table_dir)

            # Copy the original dataset to the end2end folder, overwrite if it exists
            abs_end2end_npy_dataset_path = os.path.join(abs_end2end_table_dir, npy_table_filename)
            abs_end2end_csv_dataset_path = os.path.join(abs_end2end_table_dir, f"{table_name}.csv")
            shutil.copy2(src=abs_npy_dataset_path, dst=abs_end2end_npy_dataset_path)
            shutil.copy2(src=abs_csv_dataset_path, dst=abs_end2end_csv_dataset_path)

            # Clear the buffer pool
            table_list.append(table_name)
            abs_pool_path = os.path.join(abs_end2end_table_dir, f"{table_name}_pool.npy")
            if os.path.isfile(abs_pool_path):
                os.remove(abs_pool_path)
            abs_cache_path = os.path.join(abs_end2end_table_dir, f"{table_name}_dauce_cache.pkl")
            if os.path.isfile(abs_cache_path):
                os.remove(abs_cache_path)

        init_pg(dataset, csv_dataset_dir, table_list)

        return end2end_dataset_dir, abs_end2end_dataset_dir
    else:
        if "census" in dataset and model in ["naru", "transformer"]:
            # For the census dataset and naru model, use census_int.npy
            npy_tail = "_int.npy"

        # Get the dataset path
        if "_" in dataset:
            dataset_name=dataset.split("_")[0]
        else:
            dataset_name=dataset
        src_dataset_path = f"./data/{dataset_name}/{dataset}{npy_tail}"  # Original dataset path
        end2end_dataset_path = f"./data/{dataset_name}/end2end/{dataset}{npy_tail}"  # End2end dataset path
        abs_src_dataset_path = path_util.get_absolute_path(src_dataset_path)
        abs_end2end_dataset_path = path_util.get_absolute_path(end2end_dataset_path)

        # Copy the original dataset to the end2end folder, overwrite if it exists
        shutil.copy2(src=abs_src_dataset_path, dst=abs_end2end_dataset_path)

        # Clear the buffer pool
        pool_path = f"./data/{dataset_name}/end2end/{dataset}_pool.npy"
        if os.path.isfile(pool_path):
            os.remove(pool_path)
        cache_path = f"./data/{dataset_name}/end2end/{dataset}_dauce_cache.pkl"
        if os.path.isfile(cache_path):
            os.remove(cache_path)

        return end2end_dataset_path, abs_end2end_dataset_path


def get_workload_script_paths(model: str) -> Dict[str, Path]:
    """
    Get the python script paths for query, data-update, and model-update
    """

    # Naru
    if model in ["naru", "transformer"]:
        query = path_util.get_absolute_path("./Naru/eval_model.py")
        data_update = query  # Use the same script for query and data-update
        model_update = path_util.get_absolute_path("./Naru/incremental_train.py")

    # FACE
    elif model == "face":
        query = path_util.get_absolute_path("./FACE/evaluate/Evaluate-FACE-end2end.py")
        data_update = path_util.get_absolute_path(
            "./Naru/eval_model.py"
        )  # Use the same script for data-update as Naru
        model_update = path_util.get_absolute_path("./FACE/incremental_train.py")

    else:
        raise ValueError(f"Invalid model name: {model}")

    # Return a dictionary
    script_path_dict = {
        "query": query,
        "data_update": data_update,
        "model_update": model_update,
    }

    return script_path_dict


def define_workloads(
        args: argparse.Namespace,
        script_path_dict: Dict[str, Path],
        output_file_path: Path = None,
) -> Dict[str, BaseWorkload]:
    """
    Define the three types of workloads: query, data-update, and model-update

    Set the appropriate arguments based on args.model, then pass them to the constructors of QueryWorkload, DataUpdateWorkload, and PythonScriptWorkload
    """

    # >>> Define the query workload
    # Define arguments for QueryWorkload
    if args.model == "naru":
        query_workload_args = {
            "dataset": args.dataset,
            "model": args.model,
            "end2end": None,
            "query_seed": args.query_seed,
            "eval_type": "estimate",  # For distinguishing query and data-update workloads in Naru/eval_model.py
            "num_workload": args.num_workload,
            "data_update": args.data_update,
            "model_update": args.model_update,
            "random_seed": args.random_seed,
            "num_queries": args.num_queries,
        }
    elif args.model == "transformer":
        query_workload_args = {
            "dataset": args.dataset,
            "model": args.model,
            "end2end": None,
            "query_seed": args.query_seed,
            "eval_type": "estimate",  # For distinguishing query and data-update workloads in Naru/eval_model.py
            "num_workload": args.num_workload,
            "data_update": args.data_update,
            "model_update": args.model_update,
            "random_seed": args.random_seed,
            "num_queries": args.num_queries,
        }
    elif args.model == "face":
        query_workload_args = {
            "dataset": args.dataset,
            "query_seed": args.query_seed,
            "end2end": None,
            "num_workload": args.num_workload,
            "data_update": args.data_update,
            "model_update": args.model_update,
            "num_queries": args.num_queries,
        }
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    # Define QueryWorkload
    query_workload = QueryWorkload(
        args=query_workload_args,
        script_path=script_path_dict["query"],
        output_file_path=output_file_path,
    )

    # >>> Define the first query workload
    # Define arguments for QueryWorkload
    if args.model in ["naru", "transformer"]:
        first_query_workload_args = {
            "dataset": args.dataset,
            "model": args.model,
            "query_seed": args.query_seed,
            "num_queries": args.num_queries,
            "end2end": None,
            "eval_type": "first_estimate",  # For distinguishing query and data-update workloads in Naru/eval_model.py
            "num_workload": args.num_workload,
            "data_update": args.data_update,
            "model_update": args.model_update,
            "random_seed": args.random_seed,
        }
    elif args.model == "face":
        first_query_workload_args = {
            "dataset": args.dataset, 
            "end2end": None,
            "num_queries": args.num_queries,
            "query_seed": args.query_seed,
            "num_workload": args.num_workload,
            "data_update": args.data_update,
            "model_update": args.model_update,
        }
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    # Define QueryWorkload
    first_query_workload = QueryWorkload(
        args=first_query_workload_args,
        script_path=script_path_dict["query"],
        output_file_path=output_file_path,
    )

    # >>> Define the data-update workload
    # Define arguments for DataUpdateWorkload (Naru and FACE both use Naru/eval_model.py)
    data_update_workload_args = {
        "model": args.model,
        "data_update": args.data_update,
        "dataset": args.dataset,
        "drift_test": args.drift_test,
        "end2end": None,
        "update_size": args.update_size,
        "query_seed": args.query_seed,
        "eval_type": "drift",  # For distinguishing query and data-update workloads in Naru/eval_model.py
    }
    # Define DataUpdateWorkload
    data_update_workload = DataUpdateWorkload(
        args=data_update_workload_args,
        script_path=script_path_dict["data_update"],
        output_file_path=output_file_path,
    )

    # >>> Define the model-update workload
    # Run the model update script
    model_update_workload_args = {
        "dataset": args.dataset,
        "end2end": None,
        "model": args.model,
        "model_update": args.model_update,
        "update_size": args.update_size,
        # "epochs": 20 #if args.model=='naru' else 200,
    }
    if args.model == 'naru':
        epochs = int(JsonCommunicator().get(f"naru.epochs"))
    elif args.model == 'transformer':
        epochs = int(JsonCommunicator().get(f"transformer.epochs"))
    elif args.model == 'face':
        epochs = int(JsonCommunicator().get(f"face.epochs"))
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    model_update_workload_args['epochs'] = epochs
    model_update_workload = PythonScriptWorkload(
        args=model_update_workload_args,
        python_script_path=script_path_dict["model_update"],
        output_file_path=output_file_path,
    )

    # Return a dictionary
    workload_dict = {
        "first_query": first_query_workload,
        "query": query_workload,
        "data_update": data_update_workload,
        "model_update": model_update_workload,
    }

    return workload_dict


def generate_workloads(
        args: argparse.Namespace,
        dict_workload: Dict[str, BaseWorkload],
        need_random: bool = False,
        query_workload_weight: int = 15,
        data_update_workload_weight: int = 10,
        output_file_path: Path = None,
) -> tuple[int, list]:
    """
    Generate multiple workloads to execute (an array containing QueryWorkload and DataUpdateWorkload, where the first is always a QueryWorkload) based on weights
    """
    query_workload = dict_workload["query"]
    first_query_workload = dict_workload["first_query"]
    data_update_workload = dict_workload["data_update"]

    # Generate randomly
    if need_random:
        # Set workload weights, modify based on actual needs
        dict_from_workload_to_weight = {
            query_workload: query_workload_weight,
            data_update_workload: data_update_workload_weight,
        }
        msg = f"Workload weights: query={query_workload_weight}, data-update={data_update_workload_weight}\n"
        log_util.append_to_file(output_file_path, msg)

        # Generate args.num_workload workloads
        # Define workload generator
        workload_generator = WorkloadGenerator(
            workloads=dict_from_workload_to_weight, random_seed=args.random_seed
        )

        # The first workload is a query, the rest are selected randomly
        generated_workloads = [query_workload] + [
            workload_generator.generate() for _ in range(args.num_workload - 1)
        ]

    # Remove randomness, generate the same sequence of workloads for all experiments
    else:
        generated_workloads = []
        update_workload_idx = [
            1, 3, 5, 7, 9, 11, 13, 15,
            17, 19, 23, 27, 31, 33, 36,
            39, 41, 43, 45, 47, 49,
        ]
        for i in range(args.num_workload):
            if not i:  # The first query workload uses original data
                new_workload = first_query_workload
            elif i in update_workload_idx:
                new_workload = data_update_workload
            else:
                new_workload = query_workload
            generated_workloads += [new_workload]

    # Calculate the number of update workloads
    data_update_num = sum(
        1
        for workload in generated_workloads
        if isinstance(workload, DataUpdateWorkload)
    )

    # Print generated workloads
    workloads_description = [workload.get_type() for workload in generated_workloads]
    msg = f"Workloads: {workloads_description}\n"
    log_util.append_to_file(output_file_path, msg)

    return data_update_num, generated_workloads


def run_workloads(
        workloads: List[BaseWorkload],
        model_update_workload: BaseWorkload,
        output_file_path: Path,
        dataset: str,
) -> int:
    """
    Sequentially run all workloads,
    if a DataUpdateWorkload, check for drift,
    if drift occurs, execute the model-update workload
    """

    drift_count = 0  # Number of drifts

    # Sequentially run all workloads
    for i, workload in enumerate(workloads):
        start_message = (
            f"WORKLOAD-START | "
            f"Type: {workload.get_type()} | "
            f"Progress: {i + 1}/{len(workloads)}\n"
        )
        drift_message = f"\nDRIFT-DETECTED after {i + 1}-th workload\n"

        # Print workload start info
        log_util.append_to_file(output_file_path, start_message)

        # Before a data-update workload runs, clear the stale drift verdict so a
        # crashed child (now raising instead of failing silently) can never be
        # mistaken for a fresh 'no drift' / previous-run result.
        if isinstance(workload, DataUpdateWorkload):
            communicator.DriftCommunicator().clear()

        # Run current workload
        workload_start_time = time.time()
        workload.execute_workload()  # Execute workload
        workload_time = time.time() - workload_start_time

        # If DataUpdateWorkload, check for drift
        # If drift, execute model-update workload
        model_update_time = 0
        if isinstance(workload, DataUpdateWorkload):
            if dataset in ["stats"]:
                drift_dict = communicator.DriftCommunicator().get_dict()

                drift_flag = 0
                drift_tables = []
                for table_name in drift_dict:
                    is_drift = drift_dict[table_name]
                    if is_drift:
                        drift_flag = 1
                        drift_tables.append(table_name)
                        update_dataset = f"{dataset}_{table_name}"
                        model_update_start_time = time.time()
                        model_update_workload.args["dataset"] = update_dataset # Run update on certain table
                        model_update_workload.execute_workload()  # Run model update script
                        model_update_time_single = time.time() - model_update_start_time
                        model_update_time += model_update_time_single

                drift_count += drift_flag
                log_util.append_to_file(output_file_path, drift_message)  # Print drift detection info
                appendix_message = f"Drifted tables: {drift_tables}\n"
                log_util.append_to_file(output_file_path, appendix_message) 
                
            else:
                is_drift = communicator.DriftCommunicator().get()  # Get drift detection result from communicator
                if is_drift:
                    drift_count += 1
                    log_util.append_to_file(output_file_path, drift_message)  # Print drift detection info
                    model_update_start_time = time.time()
                    model_update_workload.execute_workload()  # Run model update script
                    model_update_time = time.time() - model_update_start_time

        # Record workload and model update times
        end_message = (
            f"\nWORKLOAD-FINISHED | "
            f"Type: {workload.get_type()} | "
            f"Progress: {i + 1}/{len(workloads)} | "
            f"Workload-time: {workload_time:.6f} | "  # Workload time rounded to six decimal places
            f"Model-update-time: {model_update_time:.6f}\n\n\n"
        )  # Model update time rounded to six decimal places

        # Print workload end info
        log_util.append_to_file(output_file_path, end_message)

    return drift_count


def main():
    start_time = time.time()

    # ************************************** Define Constants **************************************
    # >>> Extract input arguments
    args: argparse.Namespace = parse_args()  # Parse arguments
    validate_argument(args)  # Check if arguments are valid
    dataset: str = args.dataset  # Dataset name ('bjaq', 'census', 'forest', 'power')
    model: str = args.model  # Model name ('naru', 'face')
    sseed = int(JsonCommunicator().get(f"random_seed"))

    # >>> Define experiment record file path
    current_datetime = datetime.now(pytz.timezone("Asia/Shanghai"))
    formatted_datetime = current_datetime.strftime(
        "%y%m%d-%H%M%S"
    )  # Format date and time as 'yyMMdd-HHMMSS'
    output_file_name = (
        f"{model}+"  # Model
        f"{dataset}+"  # Dataset
        f"{args.model_update}+"  # Model update method (adapt/update)
        f"{args.data_update}+"  # Data update method (sample/tupleskew/valueskew)
        f"wl{args.num_workload}+"  # Number of workloads
        f"qseed{args.query_seed}+"  # Query seed
        f"sseed{sseed}+"  # Sample seed
        f"t{formatted_datetime}"  # Experiment time
        f".txt"
    )
    output_file_path = path_util.get_absolute_path(
        f"./end2end/experiment-records/{output_file_name}"
    )  # Experiment record file path
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    print("Output file path:", output_file_path)

    # Print experiment arguments
    log_util.append_to_file(output_file_path, f"Input arguments = {args}\n")
    log_util.append_to_file(output_file_path, f"JSON-passed parameters = {JsonCommunicator().get_all_in_str()}\n")

    # ************************************** Prepare Model, Dataset, Communicator **************************************
    # >>> Prepare the end2end model: copy the original model to the end2end folder, overwrite if it exists
    end2end_model_path, abs_end2end_model_path = prepare_end2end_model(
        dataset=dataset, model=model
    )

    # >>> Prepare the end2end dataset: copy the original dataset to the end2end folder, overwrite if it exists
    end2end_dataset_path, abs_end2end_dataset_path = prepare_end2end_dataset(
        dataset=dataset, model=model
    )

    # Print end2end model and dataset paths
    log_util.append_to_file(output_file_path, f"MODEL-PATH={abs_end2end_model_path}\n")
    log_util.append_to_file(
        output_file_path, f"DATASET-PATH={abs_end2end_dataset_path}\n"
    )

    # >>> Set communicators (for passing parameters between scripts)
    communicator.ModelPathCommunicator().set(end2end_model_path)  # Set end2end model path
    communicator.DatasetPathCommunicator().set(end2end_dataset_path)  # Set end2end dataset path
    communicator.RandomSeedCommunicator().set(0)  # Set initial random seed to 0

    # ************************************** Create and Execute Workloads **************************************
    # Get python paths for the three types of workloads
    workload_dict_from_name_to_script: Dict[str, Path] = get_workload_script_paths(
        model=model
    )

    # Define query/data-update/model-update workloads
    workload_dict_from_name_to_obj: Dict[str, BaseWorkload] = define_workloads(
        args=args,
        script_path_dict=workload_dict_from_name_to_script,
        output_file_path=output_file_path,
    )
    workload_model_update = workload_dict_from_name_to_obj[
        "model_update"
    ]  # Model-update workload

    # >>>>> Generate multiple workloads (composed of query and data-update workloads) <<<<<
    data_update_num, workloads_to_run = generate_workloads(
        args=args,
        dict_workload=workload_dict_from_name_to_obj,
        need_random=False,  # Whether to generate randomly
        query_workload_weight=15,  # Query weight
        data_update_workload_weight=10,  # Data-update weight
        output_file_path=output_file_path,
    )
    log_util.append_to_file(output_file_path, "\n")  # Print newline

    # >>>>> Run workloads sequentially, execute model-update workload if drift occurs <<<<<
    drift_count = run_workloads(
        workloads=workloads_to_run,
        model_update_workload=workload_model_update,  # Model-update workload
        output_file_path=output_file_path,
        dataset=dataset,
    )

    # ************************************** Experiment Summary **************************************
    # Print experiment summary
    experiment_summary = (
        f"Experiment Summary: "
        f"#data-update-times={data_update_num}, "
        f"#drift={drift_count} | "
        f"total-time={time.time() - start_time:.6f}"
    )
    log_util.append_to_file(output_file_path, experiment_summary)

    # Organize experiment records
    log_parser.parse_experiment_records(
        src_dir="./end2end/experiment-records", dst_dir="./end2end/parsed-records"
    )


if __name__ == "__main__":
    main()
