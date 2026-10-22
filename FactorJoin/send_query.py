import psycopg2
import time
import os
import argparse
import numpy as np


def connect_pg(dataset: str):
    """
    Open a PostgreSQL connection to the given database.

    Connection settings are read from the environment so that no local account
    credentials are hard-coded in the repository. Set PG_USER, PG_PASSWORD,
    PG_HOST and PG_PORT to match your own PostgreSQL deployment.
    """
    return psycopg2.connect(
        database=dataset,
        user=os.environ.get("PG_USER", "postgres"),
        password=os.environ.get("PG_PASSWORD", ""),
        host=os.environ.get("PG_HOST", "localhost"),
        port=int(os.environ.get("PG_PORT", "5432")),
    )


def send_query(dataset, method_name, query_file, save_folder, iteration=None, test_type="factorjoin", update_type="sauce", warmup=False):
    conn = connect_pg(dataset)
    conn.set_client_encoding('UTF8')
    cursor = conn.cursor()

    with open(query_file, "r") as f:
        queries = f.readlines()

    # Unified planner knobs — applied to BOTH arms (ml injection and PG-native)
    # so the ONLY difference between arms is ml_joinest_enabled. Mirrors the ASM
    # end-to-end setup verified on this container. statement_timeout=0 is
    # required because heavy-tail STATS-CEB joins can run for many minutes.
    cursor.execute("SET statement_timeout = 0; commit;")
    cursor.execute("SET enable_indexscan=on;")
    cursor.execute("SET enable_bitmapscan=on;")
    cursor.execute("SET enable_material=off;")
    cursor.execute("SET enable_hashjoin=on;")

    # cursor.execute('SET debug_card_est=true;')
    # cursor.execute('SET print_sub_queries=true')
    # cursor.execute('SET print_single_tbl_queries=true')
    if test_type in ["factorjoin", "asm"]:
        cursor.execute("SET ml_joinest_enabled=true;")
        cursor.execute("SET join_est_no=0;")
        # print(f"Cardinality path: {method_name}")
        cursor.execute(f"SET ml_joinest_fname='{method_name}';")
    elif test_type == "naive" or update_type == "naive":
        cursor.execute("SET ml_joinest_enabled=false;")


    planning_time = [] 
    execution_time = []
    for no, query in enumerate(queries):
        if "||" in query:
            query = query.split("||")[-1].strip()
        if warmup:
            cursor.execute("EXPLAIN ANALYZE " + query)
        else:
            if (no+1)%30==0:
                print(f"Executing query {no+1}")
            start = time.time()
            cursor.execute("EXPLAIN ANALYZE " + query)
            res = cursor.fetchall()
            planning_time.append(float(res[-2][0].split(":")[-1].split("ms")[0].strip()))
            execution_time.append(float(res[-1][0].split(":")[-1].split("ms")[0].strip()))
            end = time.time()
            # print(f"{no}-th query finished in {end-start}, with planning_time {planning_time[no]} ms and execution_time {execution_time[no]} ms" )

    cursor.close()
    conn.close()
    save_file_name = method_name.split(".txt")[0].split("/")[-1]
    if iteration:
        np.save(save_folder + f"plan_time_{save_file_name}_{update_type}_{test_type}_iter{iteration}", np.asarray(planning_time))
        np.save(save_folder + f"exec_time_{save_file_name}_{update_type}_{test_type}_iter{iteration}", np.asarray(execution_time))
    else:
        np.save(save_folder + f"plan_time_{save_file_name}_{update_type}_{test_type}", np.asarray(planning_time))
        np.save(save_folder + f"exec_time_{save_file_name}_{update_type}_{test_type}", np.asarray(execution_time))
    
    # print(f"Method name: {save_file_name}_{test_type}")
    # print(f"Average planning time: {np.mean(planning_time)} ms")
    # print(f"Average execution time: {np.mean(execution_time)} ms")

    total_result_path=save_folder + f"Average_result_{save_file_name}_{update_type}_{test_type}_iter{iteration}.txt"
    with open(total_result_path, "w") as total_file:
        total_file.write(f"Average planning time: {np.mean(planning_time)} ms\n")
        total_file.write(f"Total execution time: {np.sum(execution_time)} ms\n")
        total_file.write(f"Average execution time: {np.mean(execution_time)} ms\n")
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='stats', help='Which dataset to be used')
    parser.add_argument('--method_name', default='stats_CEB_sub_queries_model_stats_greedy_50.txt', help='save estimates')
    parser.add_argument('--query_file', default='../workloads/stats_CEB/stats_CEB.sql', help='Query file location')
    parser.add_argument('--with_true_card', action='store_true', help='Is true cardinality included in the query?')
    parser.add_argument('--save_folder', default='./latency/', help='Query file location')
    parser.add_argument('--iteration', type=int, default=None, help='Number of iteration to read')
    parser.add_argument('--test_type', type=str, default="factorjoin", help='estimation type (naive or factorjoin)')

    args = parser.parse_args()
    
    if args.iteration:
        for i in range(args.iteration):
            send_query(args.dataset, args.method_name, args.query_file, args.save_folder, i+1)
    else:
        send_query(args.dataset, args.method_name, args.query_file, args.save_folder, test_type=args.test_type)
    
