import logging
from typing import Dict, Optional

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, dayofmonth, month, year

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Configure logging
logging.basicConfig(level=logging.INFO)


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
    """

    def write_df(name_df_pair):
        name, df = name_df_pair
        logging.info("Starting write process for DataFrame '%s'", name)
        table_path = f"{base_path}/{name}"

        # Adjust number of partitions to optimize write performance
        target_partitions = max(df.rdd.getNumPartitions() // 2, 1)
        df = df.coalesce(target_partitions)

        # Handle partitioning based on granularity
        partition_by = []
        if partition_col and partition_col in df.columns:
            logging.info(
                "Applying partitioning on column '%s' with granularity '%s'",
                partition_col,
                partition_granularity,
            )
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

        logging.info("Saving DataFrame '%s' to path '%s'", name, table_path)
        write_builder.save(table_path)

        # Optimize the Delta table using Z-Ordering on the policy column
        policy_col = policy_cols.get(name)
        if policy_col and policy_col in df.columns:
            try:
                delta_table = DeltaTable.forPath(df.sparkSession, table_path)
                logging.info(
                    "Optimizing Delta table '%s' with Z-ordering on column '%s'",
                    name,
                    policy_col,
                )
                delta_table.optimize().executeZOrderBy(policy_col)
            except Exception as e:
                logging.error("Failed to optimize Delta table '%s': %s", name, str(e))

        logging.info("Write process for DataFrame '%s' completed successfully", name)

    # Execute the write operation sequentially for each DataFrame
    for name_df_pair in df_dict.items():
        write_df(name_df_pair)
