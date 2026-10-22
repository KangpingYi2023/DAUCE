import pickle
import time
import numpy as np
import ast

OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "=": np.equal,
}

STATS={
	"short_name_to_full": {
		"u": "users",
		"b": "badges",
		"c": "comments",
		"ph": "postHistory",
		"v": "votes",
		"pl": "postLinks",
		"t": "tags",
    }
}


class JoinUnionFind():
    """
    generate equivalent set based on joins
    """
    def __init__(self):
        self.parent = {}  
        self.rank = {}    

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 1
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                if self.rank[root_x] == self.rank[root_y]:
                    self.rank[root_x] += 1


def generate_equivalent_sets(join_conditions):
    """
    Generate equivalence sets based on a set of join conditions.

    Paras:
    join_conditions: list of str, formatted as ["T1.A=T2.B", "T2.C=T3.D", ...]

    Return:
    dict: Equivalence sets with the root node as the key and all elements in the equivalence set as the value.
    """
    uf = JoinUnionFind()

    # 解析 join 条件并合并等价集
    for condition in join_conditions:
        left, right = condition.split('=')
        uf.union(left, right)

    # 将等价集分组
    equivalence_sets = {}
    for key in uf.parent:
        root = uf.find(key)
        if root not in equivalence_sets:
            equivalence_sets[root] = set()
        equivalence_sets[root].add(key)

    return equivalence_sets


def timestamp_transorform(time_string, start_date="2010-07-19 00:00:00"):
    start_date_int = time.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    time_array = time.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(time_array)) - int(time.mktime(start_date_int))

def process_condition(cond, tables_all=None):
    # parse a condition, either filter predicate or join operation
    start = None
    join = False
    join_keys = {}

    for i in range(len(cond)):
        s = cond[i]
        if s in OPS:
            start = i
            if cond[i + 1] in OPS:
                end = i + 2
            else:
                end = i + 1
            break
    assert start is not None
    left = cond[:start].strip()
    ops = cond[start:end].strip()
    right = cond[end:].strip()
    table1 = left.split(".")[0].strip().lower()
    if tables_all:
        cond = cond.replace(table1 + ".", tables_all[table1] + ".")
        table1 = tables_all[table1]
        left = table1 + "." + left.split(".")[-1].strip()
    if "." in right:
        table2 = right.split(".")[0].strip().lower()
        if tables_all:
            cond = cond.replace(table2 + ".", tables_all[table2] + ".")
            table2 = tables_all[table2]
            right = table2 + "." + right.split(".")[-1].strip()
        join = True
        join_keys[table1] = left
        join_keys[table2] = right
        return table1 + " " + table2, cond, join, join_keys

    value = right
    try:
        value = list(ast.literal_eval(value.strip()))
    except:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                value = value

    return table1, [left, ops, value], join, join_keys


def parse_query(query_str):
    tables_all=dict()
    table_predicates=dict()
    join_cond=[]
    true_card = int(query_str.split("||")[-1])

    query = query_str.split("||")[0][:-1]
    t = time.time()
    predicates_str=query.split(" WHERE ")[-1].split(" AND ")
    tables_str=query.split(" WHERE ")[0].split(" FROM ")[-1].split(",")
    # tables_rename=query.split(" WHERE ")[0].split(" FROM ")[0].split(",")
    
    for table_str in tables_str:
        table_str = table_str.strip()
        if " as " in table_str:
            tables_all[table_str.split(" as ")[-1]] = table_str.split(" as ")[0]
        else:
            tables_all[table_str.split(" ")[-1]] = table_str.split(" ")[0]

    for tab in tables_all.values():
        table_predicates[tab] = dict()
        table_predicates[tab]["cols"] = []
        table_predicates[tab]["ops"] = []
        table_predicates[tab]["vals"] = []
        table_predicates[tab]["join_keys"] = set()

    for predicate_str in predicates_str:
        table, cond, join, join_key = process_condition(predicate_str, tables_all)

        if not join:
            attr = cond[0]
            op = cond[1]
            value = cond[2]
            if "Date" in attr:
                assert "::timestamp" in value  # this is hardcoded for STATS-CEB workload
                value = timestamp_transorform(value.split("::timestamp")[0].strip("'"))

            table_predicates[table]["cols"].append(attr.split(".")[-1])
            table_predicates[table]["ops"].append(op)
            table_predicates[table]["vals"].append(value)   
        else:
            table1=table.split(" ")[0]
            table2=table.split(" ")[-1]
            join_cond.append(cond)
            join_key1=join_key[table1].split(".")[-1].strip()
            join_key2=join_key[table2].split(".")[-1].strip()
            table_predicates[table1]["join_keys"].add(join_key1)
            table_predicates[table2]["join_keys"].add(join_key2)

    equivalent_set = generate_equivalent_sets(join_cond)

    return tables_all, table_predicates, equivalent_set, true_card


def read_queries_on_stats(query_file, save_res=None):
    with open(query_file, "r") as f:
        queries = f.readlines()

    log_path="./Naru/checkpoints/query_parse_log.txt"
    with open(log_path, "w") as log_file:
        for i, query_str in enumerate(queries):
            tables_all, table_predicates, equivalent_sets, true_card = parse_query(query_str)
            if i%10==0:
                log_file.write(f"{tables_all}\n")
                for tab in table_predicates:
                    log_file.write(f"{tab}:{table_predicates[tab]}\n")
                for key in equivalent_sets:
                    log_file.write(f"{equivalent_sets[key]}\n")
                log_file.write(f"true card: {true_card}\n")

    print("Query parse finished!")
	        

if __name__ == "__main__":
	query_path="./workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"
	read_queries_on_stats(query_path)
