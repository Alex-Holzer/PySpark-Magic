import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from delta.tables import DeltaTable
from pyspark.sql import DataFrame
    df_dict: Dict[str, DataFrame],
    base_path: str = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW",
    partition_col: str = "date",
    zorder_col: str = "policy_number",
) -> None:
    """
    Saves multiple PySpark DataFrames as Delta tables in the specified path.

    Args:
        df_dict (Dict[str, DataFrame]): A dictionary where keys are DataFrame names and values are DataFrame objects.
        base_path (str, optional): The base path where Delta tables will be saved. Defaults to the given path.
        partition_col (str, optional): The column name to partition the data. Defaults to "date".
        zorder_col (str, optional): The column name to Z-Order the data. Defaults to "policy_number".

    Returns:
        None
    """
    for table_name, df in df_dict.items():
        try:
            # Build the full path
            table_path = f"{base_path}/{table_name}"

            # Write the DataFrame as Delta table
            (
                df.write.format("delta")
                .partitionBy(partition_col)
                .mode("overwrite")
                .save(table_path)
            )

            # Optimize and Z-Order the Delta table
            spark.sql(f"OPTIMIZE delta.`{table_path}` ZORDER BY ({zorder_col})")

            logging.info(f"Successfully saved and optimized table {table_name}")
        except Exception as e:
            logging.error(f"Error saving table {table_name}: {e}")
        except Exception as e:
            logging.error(f"Error saving table {table_name}: {e}")


# improved version



def save_dataframes_as_delta(
    df_dict: Dict[str, DataFrame],
    base_path: str = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW",
    partition_col: Optional[str] = None,
    partition_granularity: str = "month",  # 'year', 'month', or 'day'
    policy_cols: Dict[
        str, str
    ] = {},  # Mapping of DataFrame names to policy column names
    mode: str = "overwrite",  # 'overwrite' or 'append'
    allow_schema_evolution: bool = True,
) -> None:
    """
    Saves multiple PySpark DataFrames as optimized Delta tables in the specified base path.
    Accounts for dynamic policy column names, partition granularity, and schema evolution.

    Args:
        df_dict (Dict[str, DataFrame]): A dictionary where keys are DataFrame names and values are DataFrame objects.
        base_path (str, optional): The base path where Delta tables will be saved.
        partition_col (str, optional): Column name to partition the data by.
        partition_granularity (str, optional): Granularity of partitioning ('year', 'month', 'day').
        policy_cols (Dict[str, str], optional): Mapping of DataFrame names to policy column names.
        mode (str, optional): Write mode ('overwrite' or 'append'). Defaults to 'overwrite'.
        allow_schema_evolution (bool, optional): Whether to allow schema evolution during write. Defaults to True.

    # Example DataFrames with different policy column names
    df1 = df1.withColumn("timestamp_col", df1["timestamp"])  # Assuming df1 has a 'timestamp' column
    df2 = df2.withColumn("timestamp_col", df2["timestamp"])
    df3 = df3.withColumn("timestamp_col", df3["timestamp"])

    df_dict = {
        "table1": df1,
        "table2": df2,
        "table3": df3,
    }

    # Mapping of table names to their policy column names
    policy_cols = {
        "table1": "policy_id",
        "table2": "policy_number",
        "table3": "policy_num",
    }

    # Call the function with partitioning and schema evolution
    save_dataframes_as_delta(
        df_dict,
        partition_col="timestamp_col",
        partition_granularity="month",
        policy_cols=policy_cols,
        mode="overwrite",
        allow_schema_evolution=True
    )
    """

    def write_df(name_df_pair):
        name, df = name_df_pair
        table_path = f"{base_path}/{name}"

        # Adjust number of partitions to optimize write performance
        target_partitions = max(df.rdd.getNumPartitions() // 2, 1)
        df = df.coalesce(target_partitions)

        # Handle partitioning based on granularity
        partition_by = []
        if partition_col and partition_col in df.columns:
            if partition_granularity == "year":
                df = df.withColumn("partition_year", year(col(partition_col)))
                partition_by = ["partition_year"]
            elif partition_granularity == "month":
                df = df.withColumn(
                    "partition_year", year(col(partition_col))
                ).withColumn("partition_month", month(col(partition_col)))
                partition_by = ["partition_year", "partition_month"]
            elif partition_granularity == "day":
                df = (
                    df.withColumn("partition_year", year(col(partition_col)))
                    .withColumn("partition_month", month(col(partition_col)))
                    .withColumn("partition_day", dayofmonth(col(partition_col)))
                )
                partition_by = ["partition_year", "partition_month", "partition_day"]

        # Write the DataFrame as a Delta table with partitioning
        write_builder = df.write.format("delta").mode(mode)

        if allow_schema_evolution:
            if mode == "overwrite":
                write_builder = write_builder.option("overwriteSchema", "true")
            else:
                write_builder = write_builder.option("mergeSchema", "true")

        if partition_by:
            write_builder = write_builder.partitionBy(*partition_by)

        write_builder.save(table_path)

        # Optimize the Delta table using Z-Ordering on the policy column
        policy_col = policy_cols.get(name)
        if policy_col and policy_col in df.columns:
            delta_table = DeltaTable.forPath(df.sparkSession, table_path)
            delta_table.optimize().executeZOrderBy(policy_col)

    with ThreadPoolExecutor() as executor:
        executor.map(write_df, df_dict.items())
