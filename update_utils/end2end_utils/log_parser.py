import ast
import re
import sys
import os
import pandas as pd

sys.path.append("./")
from pathlib import Path
from typing import List

import numpy as np

from update_utils import path_util, log_util

MODEL_NAME=['face', 'naru', 'transformer']
UPDATE_TYPE=['sample', 'permute-opt', 'single', 'value', 'tupleskew', 'valueskew']
DATASET_NAME=['bjaq', 'census', 'forest', 'power', 'bjaq_gaussian', 'census_gaussian']

def parse_lines_with_keywords(src_path: Path, dst_path: Path, start_words: List[str]):
    with src_path.open("r") as src_file, dst_path.open("w") as dst_file:
        for line in src_file:
            if any(line.startswith(word) for word in start_words):
                dst_file.write(line)
                dst_file.write("\n")


def parse_experiment_records(
    src_dir: str,
    dst_dir: str,
    start_words: List[str] = None,  
    var_names: List[str] = None,  
    list_names: List[str] = None,  #
):
    if start_words is None:
        start_words = [
            "Input arguments",
            "JSON-passed parameters",
            "Experiment Summary",
            "Distance score",
            "WORKLOAD-FINISHED",
            "ReportEsts",
        ]
    if var_names is None:
        var_names = ["Model-update-time"]
    if list_names is None:
        list_names = [
            # "ReportEsts", 
            # "Distance score",
        ]

    src_dir_path = path_util.get_absolute_path(src_dir)
    dst_dir_path = path_util.get_absolute_path(dst_dir)

    dst_dir_path.mkdir(parents=True, exist_ok=True)

    
    def err_dict_init():
        err_dict={}
        for dataset in DATASET_NAME:
            err_dict[dataset]={}
            for model_name in MODEL_NAME:
                err_dict[model_name][dataset]=[]

        return err_dict
                
    errs={}
    scores={}
    
    for src_file_path in src_dir_path.glob("*.txt"):
        dst_file_path = dst_dir_path / src_file_path.name

        if dst_file_path.exists():
            continue
        
        group=dst_file_path.name.split("+")

        model=group[0]
        dataset=group[1]
        model_update_type=group[2]
        # update_type=group[3]

        parse_lines_with_keywords(src_file_path, dst_file_path, start_words)

        # log_util.append_to_file(dst_file_path, "\n")

        for var_name in var_names:
            var_sum = sum_float_var_in_log(dst_file_path, var_name=var_name)
            log_util.append_to_file(
                dst_file_path, f"Sum of {var_name} = {var_sum:.4f}\n"
            )

        for list_name in list_names:
            concat_list, match_cnt = concat_list_var_in_log(
                dst_file_path, list_name=list_name
            )

            # log_util.append_to_file(dst_file_path, f"Concatenated {list_name} = {concat_list}")
            if list_name == "ReportEsts":
                if match_cnt == 0:
                    log_util.append_to_file(dst_file_path, f"Keyword {list_name} NOT found")
                    continue
                
                if model not in errs:
                    errs[model]={}
                if dataset not in errs[model]:
                    errs[model][dataset]=[]

                def generate_report_est_str(arr: list) -> str:
                    arr_max = np.max(arr)
                    quant99 = np.quantile(arr, 0.99)
                    quant95 = np.quantile(arr, 0.95)
                    arr_median = np.quantile(arr, 0.5)
                    arr_mean = np.mean(arr)
                    msg = (
                        f"max: {arr_max:.4f}\t"
                        f"99th: {quant99:.4f}\t"
                        f"95th: {quant95:.4f}\t"
                        f"median: {arr_median:.4f}\t"
                        f"mean: {arr_mean:.4f}\n"
                    )
                    return msg

                print(len(concat_list))
                tuple_len = int(len(concat_list) / match_cnt)
                first_query_errs = concat_list[:tuple_len]
                first_query_est_msg = (
                    "The 1st    ReportEsts -> "
                    + generate_report_est_str(first_query_errs)
                )
                log_util.append_to_file(dst_file_path, content=first_query_est_msg)

                after_query_errs = concat_list[tuple_len:]
                after_query_est_msg = (
                    "2nd to end ReportEsts -> "
                    + generate_report_est_str(after_query_errs)
                )
                log_util.append_to_file(dst_file_path, content=after_query_est_msg)

                """log for bootstrap"""
                errs[model][dataset].append(np.mean(after_query_errs))

            if list_name == "Distance score":
                """Only log for bootstrap"""
                if match_cnt == 0:
                    print(f"Keyword {list_name} NOT found")
                    continue
                if match_cnt > 1:
                    print(f"Too many {list_name}s!")
                    continue

                if model not in scores:
                    scores[model]={}
                if dataset not in scores[model]:
                    scores[model][dataset]=[]

                scores[model][dataset].append(concat_list[0])

    # log4bootstrap(errs, scores)

            
def sum_float_var_in_log(file_path: Path, var_name: str) -> float:
    total_sum = 0.0
    with file_path.open("r") as file:
        for line in file:
            if var_name in line:
                # Extract the value after the variable name and sum it up
                try:
                    value_str = line.split(var_name + ":")[1].strip()
                    total_sum += float(value_str)
                except (IndexError, ValueError):
                    # Handle cases where the line format is unexpected or the value is not a float
                    print(f"Warning: Could not parse line: {line.strip()}")
    return total_sum


def concat_list_var_in_log(file_path: Path, list_name: str):
    concatenated_list = []

    # Regular expression to find the desired list name and its contents
    if list_name == "ReportEsts":
        pattern = rf"{re.escape(list_name)}: \[([^\]]*)\]"
    elif list_name == "Distance score":
        pattern = rf"{re.escape(list_name)}: ([^\]]*)"

    match_cnt = 0

    with file_path.open("r") as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                # Extract the list and convert it to a Python list
                list_str = "[" + match.group(1) + "]"
                current_list = ast.literal_eval(list_str)

                # Concatenate to the main list
                concatenated_list.extend(current_list)
                match_cnt += 1

    return concatenated_list, match_cnt


def log4bootstrap(qerrors_all, scores_all):
    for model in scores_all:
        for dataset in scores_all[model]:
            qerrors=qerrors_all[model][dataset]
            scores=scores_all[model][dataset]

            bootstrap_dir="./end2end/end2end-bootstrap"
            bootstrap_file=f"{model}+{dataset}+dis+qerror.csv"
            bootstrap_path=os.path.join(bootstrap_dir, bootstrap_file)

            bootstrap_data=pd.DataFrame({
                "distances": scores,
                "q-errors": qerrors
            })

            bootstrap_data.to_csv(bootstrap_path, header=True, index=False, mode="w")


if __name__ == "__main__":
    src_dir: str = "./end2end/experiment-records"
    dst_dir: str = "./end2end/parsed-records"
    parse_experiment_records(src_dir=src_dir, dst_dir=dst_dir)
