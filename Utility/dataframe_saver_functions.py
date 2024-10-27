import logging
from typing import Dict, Optional

from delta.tables import DeltaTable
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, dayofmonth, month, year

# Initialize logging
logging.basicConfig(level=logging.INFO)


def save_dataframes_as_delta(
    df_dict: Dict[str, DataFrame],
    base_path: str = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW",
    partition_col: Optional[str] = None,
    partition_granularity: str = "month",  # 'year', 'month', or 'day'
    policy_col: Optional[str] = None,  # Single policy column name
    mode: str = "overwrite",  # 'overwrite' or 'append'
    allow_schema_evolution: bool = True,
    max_partitions: int = 200,  # Maximum number of partitions to prevent small files
) -> None:
    """
    Saves multiple PySpark DataFrames as optimized Delta tables in the specified base path.
    Accounts for dynamic policy column name, partition granularity, and schema evolution.

    Args:
        df_dict (Dict[str, DataFrame]): A dictionary where keys are DataFrame names and values are DataFrame objects.
        base_path (str, optional): The base path where Delta tables will be saved.
        partition_col (str, optional): Column name to partition the data by.
        partition_granularity (str, optional): Granularity of partitioning ('year', 'month', 'day').
        policy_col (str, optional): Policy column name to use for optimization.
        mode (str, optional): Write mode ('overwrite' or 'append'). Defaults to 'overwrite'.
        allow_schema_evolution (bool, optional): Whether to allow schema evolution during write. Defaults to True.
        max_partitions (int, optional): Maximum number of partitions to prevent small files.
    """
    failed_tables = {}

    for name, df in df_dict.items():
        try:
            logging.info("Starting write process for DataFrame '%s'", name)

            # Check if DataFrame is empty
            if df.rdd.isEmpty():
                logging.warning("DataFrame '%s' is empty. Skipping write.", name)
                continue

            table_path = f"{base_path}/{name}"

            # Adjust number of partitions to optimize write performance
            num_partitions = df.rdd.getNumPartitions()
            if num_partitions > max_partitions:
                df = df.coalesce(max_partitions)
                logging.info(
                    "Coalesced DataFrame '%s' to %d partitions", name, max_partitions
                )

            # Handle partitioning based on granularity
            partition_by = []
            if partition_col and partition_col in df.columns:
                logging.info(
                    "Applying partitioning on column '%s' with granularity '%s'",
                    partition_col,
                    partition_granularity,
                )
                if partition_granularity in ["year", "month", "day"]:
                    df = df.withColumn("partition_year", year(col(partition_col)))
                    partition_by.append("partition_year")
                if partition_granularity in ["month", "day"]:
                    df = df.withColumn("partition_month", month(col(partition_col)))
                    partition_by.append("partition_month")
                if partition_granularity == "day":
                    df = df.withColumn("partition_day", dayofmonth(col(partition_col)))
                    partition_by.append("partition_day")

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
            logging.info("DataFrame '%s' saved successfully.", name)

            # Optimize the Delta table using Z-Ordering on the policy column
            if policy_col and policy_col in df.columns:
                try:
                    delta_table = DeltaTable.forPath(df._jdf.sparkSession(), table_path)
                    logging.info(
                        "Optimizing Delta table '%s' with Z-ordering on column '%s'",
                        name,
                        policy_col,
                    )
                    delta_table.optimize().executeZOrderBy(policy_col)
                    logging.info(
                        "Optimization of DataFrame '%s' completed successfully.", name
                    )
                except Exception as e:
                    logging.error(
                        "Failed to optimize Delta table '%s': %s", name, str(e)
                    )
            else:
                logging.info(
                    "Policy column '%s' not found in DataFrame '%s'. Skipping optimization.",
                    policy_col,
                    name,
                )

        except Exception as e:
            logging.error("Failed to write DataFrame '%s': %s", name, str(e))
            failed_tables[name] = str(e)

    if failed_tables:
        error_message = f"Failed to write the following DataFrames: {', '.join(failed_tables.keys())}"
        logging.error(error_message)
        for table_name, error in failed_tables.items():
            logging.error("DataFrame '%s' failed due to: %s", table_name, error)
        raise RuntimeError(error_message)


df_dict = {
    "table1": df1,
    "table2": df2,
    # ... potentially up to 100 DataFrames
}

save_dataframes_as_delta(
    df_dict=df_dict,
    partition_col="date_column",
    partition_granularity="month",
    policy_col="policy_column",
    mode="overwrite",
    allow_schema_evolution=True,
    max_partitions=200,  # Adjust as needed
)


# without Z-Ordering
import logging
from typing import Dict, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, dayofmonth, month, year

# Initialize logging
logging.basicConfig(level=logging.INFO)


def save_dataframes_as_delta(
    df_dict: Dict[str, DataFrame],
    base_path: str = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW",
    partition_col: Optional[str] = None,
    partition_granularity: str = "month",  # 'year', 'month', or 'day'
    mode: str = "overwrite",  # 'overwrite' or 'append'
    allow_schema_evolution: bool = True,
    max_partitions: int = 200,  # Maximum number of partitions to prevent small files
) -> None:
    """
    Saves multiple PySpark DataFrames as Delta tables in the specified base path.
    Accounts for dynamic partition column name, partition granularity, and schema evolution.

    Args:
        df_dict (Dict[str, DataFrame]): A dictionary where keys are DataFrame names and values are DataFrame objects.
        base_path (str, optional): The base path where Delta tables will be saved.
        partition_col (str, optional): Column name to partition the data by.
        partition_granularity (str, optional): Granularity of partitioning ('year', 'month', 'day').
        mode (str, optional): Write mode ('overwrite' or 'append'). Defaults to 'overwrite'.
        allow_schema_evolution (bool, optional): Whether to allow schema evolution during write. Defaults to True.
        max_partitions (int, optional): Maximum number of partitions to prevent small files.
    """
    failed_tables = {}

    for name, df in df_dict.items():
        try:
            logging.info("Starting write process for DataFrame '%s'", name)

            # Check if DataFrame is empty
            if df.rdd.isEmpty():
                logging.warning("DataFrame '%s' is empty. Skipping write.", name)
                continue

            table_path = f"{base_path}/{name}"

            # Adjust number of partitions to optimize write performance
            num_partitions = df.rdd.getNumPartitions()
            if num_partitions > max_partitions:
                df = df.coalesce(max_partitions)
                logging.info(
                    "Coalesced DataFrame '%s' to %d partitions", name, max_partitions
                )

            # Handle partitioning based on granularity
            partition_by = []
            if partition_col and partition_col in df.columns:
                logging.info(
                    "Applying partitioning on column '%s' with granularity '%s'",
                    partition_col,
                    partition_granularity,
                )
                if partition_granularity in ["year", "month", "day"]:
                    df = df.withColumn("partition_year", year(col(partition_col)))
                    partition_by.append("partition_year")
                if partition_granularity in ["month", "day"]:
                    df = df.withColumn("partition_month", month(col(partition_col)))
                    partition_by.append("partition_month")
                if partition_granularity == "day":
                    df = df.withColumn("partition_day", dayofmonth(col(partition_col)))
                    partition_by.append("partition_day")

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
            logging.info("DataFrame '%s' saved successfully.", name)

        except Exception as e:
            logging.error("Failed to write DataFrame '%s': %s", name, str(e))
            failed_tables[name] = str(e)

    if failed_tables:
        error_message = f"Failed to write the following DataFrames: {', '.join(failed_tables.keys())}"
        logging.error(error_message)
        for table_name, error in failed_tables.items():
            logging.error("DataFrame '%s' failed due to: %s", table_name, error)
        raise RuntimeError(error_message)
