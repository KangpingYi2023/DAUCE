# DAUCE

This is an open source implementation of "DAUCE: A Drift-aware Update Method For Learned Cardinality Estimators".

### Environments

```shell
# create anaconda environments
conda create --name dauce python=3.9
conda activate dauce

cd DAUCE/
# PyTorch is also contained in requirements.txt. You can also comment it and install it by yourself
pip install -r requirements.txt

# Install dependencies of FACE
cd ./FACE/torchquadMy
pip install .
```

### Continuous update experiments for each datasets

```shell
# You can run an end-to-end experiment by running this conmmand. Some config examples are stored in ./end2end/configs/. You can choose the config file from this python script
cd DAUCE/
python end2end/multi_experiments.py
```

### Results

Experimental results will be shown in the following folds:

- The detailed experiment records will be shown in ./end2end/experiment-records
- The overall performance will be shown in ./end2end/parsed-records
- The q-errors of each query workload will be shown in ./end2end/end2end-evaluations

## End-to-end update experiment

To execute the end-to-end update experiment, you need to create a new Anaconda environment by following the instructions in the README file located in the ./FactorJoin directory.

```shell
# You can run an end-to-end experiment by running this conmmand. 
cd FactorJoin/
python update_experiment.py
```


