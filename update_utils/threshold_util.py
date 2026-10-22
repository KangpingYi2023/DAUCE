import os
import sys
import pandas as pd
import numpy as np

sys.path.append("./")
from update_utils import log_util

def threshold_drive(dataset:str, distances, qerrors, max_qerror=2, FP=0.01):

    exceed_idx=np.where(qerrors>max_qerror)
    exceed_distances=distances[exceed_idx]
    # threshold=np.min(exceed_distances)
    threshold=np.quantile(exceed_distances, FP)
    # print(f"{dataset} recommended threshold: {threshold}")

    return threshold


def threshold_setting_per_model(dataset_paras):
    models=["naru","transformer","face"]
    # datasets=["bjaq","census","power","forest"]
    threshold_logname=f"threshold_per_model_dataset"
    threshold_logpath=f"./end2end/end2end-bootstrap/{threshold_logname}.txt"

    with open(threshold_logpath,"w") as f:
        for model in models:
            for dataset, paras in dataset_paras.items():
                max_qerror=paras["max_qerror"]
                FP=paras["FP"]

                e2e_bootstrap_filename=f"{model}+{dataset}+dis+qerror"
                e2e_bootstrap_file=f"./end2end/end2end-bootstrap/{e2e_bootstrap_filename}.csv"

                if os.path.isfile(e2e_bootstrap_file):
                    data_file=open(e2e_bootstrap_file)
                    dataframe=pd.read_csv(data_file)
                    distances=dataframe["distances"].values
                    qerrors=dataframe["q-errors"].values

                    threshold=threshold_drive(dataset, distances, qerrors, max_qerror, FP)
                    log_message=f"model-{model}, dataset-{dataset} recommended threshold: {threshold}\n"
                    # log_util.append_to_file(threshold_logpath, log_message)
                    print(log_message)
                    f.write(log_message)
                    f.flush()


def threshold_setting_per_dataset(dataset_para):
    models=["naru","transformer","face"]
    # datasets=["bjaq","census","power","forest"]
    threshold_logname=f"threshold_per_dataset"
    threshold_logpath=f"./end2end/end2end-bootstrap/{threshold_logname}.txt"

    with open(threshold_logpath, "w") as f:
        for dataset, paras in dataset_para.items():
            all_distances=[]
            all_qerrors=[]
            max_qerror=paras["max_qerror"]
            FP=paras["FP"]
            for model in models:
                e2e_bootstrap_filename=f"{model}+{dataset}+dis+qerror"
                e2e_bootstrap_file=f"./end2end/end2end-bootstrap/{e2e_bootstrap_filename}.csv"

                if os.path.isfile(e2e_bootstrap_file):
                    data_file=open(e2e_bootstrap_file)
                    dataframe=pd.read_csv(data_file)
                    distances=dataframe["distances"].tolist()
                    qerrors=dataframe["q-errors"].tolist()
                    
                    # np.hstack((all_distances, distances))
                    # np.hstack((all_qerrors, qerrors))
                    all_distances += distances
                    all_qerrors += qerrors
            
            all_distances=np.array(all_distances)
            all_qerrors=np.array(all_qerrors)
            threshold=threshold_drive(dataset, all_distances, all_qerrors, max_qerror, FP)
            log_message=f"dataset-{dataset} recommended threshold: {threshold}\n"
            print(log_message)
            f.write(log_message)
            f.flush()
            # log_util.append_to_file(threshold_logpath, log_message)


if __name__ == '__main__':
    dataset_para={
        # "bjaq": {
        #     "max_qerror": 1.3,
        #     "FP": 1e-3
        # },
        # "census": {
        #     "max_qerror": 1.5,
        #     "FP": 2e-2
        # },
        # "forest": {
        #     "max_qerror": 2,
        #     "FP": 1e-2
        # },
        # "power": {
        #     "max_qerror": 1.4,
        #     "FP": 1e-2
        # },
        "bjaq_gaussian":{
            "max_qerror": 2,
            "FP": 1e-2
        },
        "census_gaussian": {
            "max_qerror": 2,
            "FP": 1e-2
        }
    }
    threshold_setting_per_dataset(dataset_para)