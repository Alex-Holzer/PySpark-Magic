"""

dataframe_combination_functions.py
===========================

The module provides a collection of dataframe combination functions.

"""

from typing import List, Union

from pyspark.sql import DataFrame, SparkSession, broadcast

spark: SparkSession = SparkSession.builder.getOrCreate()


def broadcast_join(
    df_large: DataFrame,
    df_small: DataFrame,
    join_cols: Union[str, List[str]],
    join_type: str = "inner",
) -> DataFrame:
    """Perform a broadcast join between two DataFrames.

    Args:
        df_large: The larger DataFrame.
        df_small: The smaller DataFrame to broadcast.
        join_cols: Column(s) to join on.
        join_type: Type of join to perform.

    Returns:
        DataFrame: Result of the broadcast join.

    Raises:
        ValueError: If DataFrames are empty or join columns invalid.
    """
    if df_large.rdd.isEmpty() or df_small.rdd.isEmpty():
        raise ValueError("Input DataFrames cannot be empty.")

    join_columns = [join_cols] if isinstance(join_cols, str) else join_cols

    for dataframe in [df_large, df_small]:
        missing_cols = set(join_columns) - set(dataframe.columns)
        if missing_cols:
            raise ValueError(f"Join columns {missing_cols} not found in DataFrame.")

    return df_large.join(broadcast(df_small), on=join_columns, how=join_type)
