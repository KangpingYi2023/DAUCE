# DAUCE 实验复现

本目录提供 Table IV、V、VI、VII 和 Figure 8 的复现代码与一站式入口。Table V、VI 保留 DDUp 与 DAUCE 两种更新方法；Figure 8 运行 None、Always、DAUCE 三个实验臂。

数据和模型将很快上传。

## 环境

在仓库根目录执行以下命令。Table IV–VII 使用 Linux、Python 3.11 和 CUDA GPU；FACE 需要 CUDA。

```bash
conda create -n dauce python=3.11 pip -y
conda activate dauce
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install ./FACE/torchquadMy
```

Figure 8 使用单独的 FactorJoin 环境和 Docker：

```bash
conda env create -f FactorJoin/environment.yml
conda activate dauce-factorjoin
docker info
```

Docker 只运行 PostgreSQL，Python 实验在宿主机执行。默认 PostgreSQL 配置为 `shared_buffers=12GB`、`work_mem=2GB`、`effective_cache_size=32GB`；请预留相应内存。用户需有 Docker 操作权限。

## 输入放置

保持原始文件名和列顺序。只使用可信来源的模型和数据文件。

```text
data/
  census/census_int.npy
  census/census.csv
  forest/forest.npy
  forest/forest.csv
  bjaq/bjaq.npy
  power/power.npy
  stats/{badges,votes,postHistory,posts,users,comments,postLinks,tags}.csv
models/
  naru_origin_models/origin-{census,forest,bjaq}-*.pt
  transformer_origin_models/origin-{census,forest,bjaq}-*.pt
  face_origin_models/{bjaq,power}*.t
```

每个数据集/骨干组合只放置一份匹配的原始检查点。Table IV–VII 根据所选实验检查输入；Census、Forest 的基线查询需要原始 CSV 和 NPY 两份输入。Figure 8 只需八张 STATS CSV，FactorJoin 基模型在运行时训练。SQL 工作负载已包含在 `workloads/stats_CEB/`。

## 一站式运行

Table IV–VII 在 `dauce` 环境运行：

```bash
bash scripts/run_table4.sh
bash scripts/run_table5.sh
bash scripts/run_table6.sh
bash scripts/run_table7.sh
```

配置位于 `experiments/configs/table4.yaml` 至 `table7.yaml`。支持 `--dataset`、`--model`、`--method DDUp|DAUCE`、`--drift sample|tupleskew|valueskew` 过滤实验。Table V–VII 还可使用 `--queries`、`--epochs`；`--dry-run` 仅打印实验矩阵。

快速验证入口：

```bash
bash scripts/run_table4.sh --smoke
bash scripts/run_table5.sh --smoke
bash scripts/run_table6.sh --smoke
bash scripts/run_table7.sh --smoke
```

`--smoke` 默认只选首个数据集/骨干/查询种子；Table V–VII 使用一次更新、5 条查询、1 epoch，FACE 降低积分预算。完整实验使用配置中的预算。Table IV 的检测预算保持配置值。

Figure 8 在 `dauce-factorjoin` 环境运行：

```bash
bash scripts/run_figure8.sh --smoke
bash scripts/run_figure8.sh
```

脚本自动构建 `docker/pg/Dockerfile`，启动独立容器，在宿主 `127.0.0.1:5433` 提供 PostgreSQL，初始化 STATS 表与索引，运行两个阶段并聚合 CSV：

1. `run_e2e_dauce.py --skip_sauce`：Always 与 None。
2. `run_sauce_replay.py`：重新装载相同数据批次，运行 DAUCE 检测、门控更新与查询。

完整运行七轮、每轮全部 STATS-CEB 查询；`--smoke` 运行两轮、每轮前一条查询。可用 `--port 5434` 更改独立容器端口。默认结束时删除本次容器；`--keep-container` 可保留用于检查。默认本地测试密码为 `postgres`，可通过 `PG_PASSWORD` 环境变量覆盖。

手动构建 PostgreSQL 镜像的命令：

```bash
docker build -t dauce-pg:revision docker/pg
```

## 输出与运行目录

每次运行输出到独立的 `outputs/<实验名>_<运行标识>/`，包含对应 `table4.csv` 至 `table7.csv` 或 `figure8.csv`、单元日志及运行记录。Table V–VII 另外保存逐次运行的 `runs.csv`；Table VII 的 DAUCE Total 从同次运行的总时间中扣除离线校准时间。Figure 8 的 CSV 时延单位为秒。

脚本从只读原始输入复制出工作数据和工作模型，不修改 `*_origin_models` 中的检查点。不同实验共用部分工作目录，因此通过 `.reproduce.lock` 限制同一仓库副本一次只运行一个实验。Python 路径可通过 `PYTHON`（Table IV–VII）或 `FACTORJOIN_PYTHON`（Figure 8）指定。

数据、模型、查询缓存、积分缓存、容器运行数据及输出均不纳入 Git；这些目录在运行时生成。
