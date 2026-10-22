import os
import psycopg2


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


def init_pg(dataset, data_dir, table_list):
    conn = connect_pg(dataset)
    cursor = conn.cursor()

    try:
        for table_name in table_list:
            #delete all data and replace with data file
            cursor.execute(f"TRUNCATE TABLE {table_name};")

            table_filename = f"{table_name}.csv"
            table_filepath = os.path.join(data_dir, table_filename)

            with open(table_filepath, "r") as csv_file:
                cursor.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV HEADER", csv_file)
            print(f"Table {table_name} init finished!")

        conn.commit()

    except Exception as e:
        print(f"error: {e}")
        conn.rollback()

    cursor.close()
    conn.close()


def update_pg(dataset, table, delta_path):
    conn = connect_pg(dataset)
    cursor = conn.cursor()

    # related_path="../../"
    delta_path=os.path.normpath(delta_path)
    # delta_path=os.path.join(related_path, delta_path)
    table=table.lower()
    try:
        with open(delta_path, "r") as delta_file:
            cursor.copy_expert(f"COPY {table} FROM STDIN WITH CSV HEADER", delta_file)
        conn.commit()
    except Exception as e:
        print(f"error: {e}")
        conn.rollback()

    cursor.close()
    conn.close()