from pyspark.sql.dataframe import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, expr
from typing import Union, List, Tuple, Optional, Any, Dict


def grouped_aggregation(
    df: DataFrame, group_cols: List[str], aggregations: Dict[str, Union[str, List[str]]]
) -> DataFrame:
    """
    Group a DataFrame by specified columns and calculate aggregations.

    Args:
        df (DataFrame): Input DataFrame.
        group_cols (List[str]): List of column names to group by.
        aggregations (Dict[str, Union[str, List[str]]]): Dictionary specifying the aggregations.
            Keys are column names, values are either a single aggregation function name
            or a list of aggregation function names.

    Returns:
        DataFrame: Grouped and aggregated DataFrame.

    Raises:
        ValueError: If any of the specified columns do not exist in the DataFrame.

    Feasible Aggregation Methods:
        - Basic: count, sum, avg (mean), min, max
        - First/Last: first, last
        - Collection: collect_list, collect_set
        - Distinct: countDistinct, approx_count_distinct
        - Statistical: stddev, stddev_pop, variance, var_pop, skewness, kurtosis
        - Percentile: Use expr("percentile(column, array(0.5, 0.75, 0.9))") for percentiles

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 150),
        ...     ("B", 3, 200),
        ...     ("B", 4, 250)
        ... ], ["category", "id", "value"])
        >>> group_cols = ["category"]
        >>> aggregations = {
        ...     "id": "count",
        ...     "value": ["sum", "avg", "max"]
        ... }
        >>> result = grouped_aggregation(df, group_cols, aggregations)
        >>> result.show()
        +--------+-------+----------+----------+----------+
        |category|id_count|value_sum |value_avg |value_max |
        +--------+-------+----------+----------+----------+
        |A       |2      |250       |125.0     |150       |
        |B       |2      |450       |225.0     |250       |
        +--------+-------+----------+----------+----------+
    """
    # Validate input columns
    all_cols = set(group_cols + list(aggregations.keys()))
    if not all_cols.issubset(df.columns):
        invalid_cols = all_cols - set(df.columns)
        raise ValueError(
            f"The following columns do not exist in the DataFrame: {invalid_cols}"
        )

    # Prepare aggregation expressions
    agg_exprs = []
    for col_name, agg_funcs in aggregations.items():
        if isinstance(agg_funcs, str):
            agg_funcs = [agg_funcs]
        for func in agg_funcs:
            agg_exprs.append(expr(f"{func}({col_name}) as {col_name}_{func}"))

    # Perform grouping and aggregation
    return df.groupBy(*group_cols).agg(*agg_exprs)


def add_total(
    df: DataFrame,
    value_col: str,
    partition_cols: Optional[Union[str, List[str]]] = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with the total sum of a specified column, optionally partitioned.

    This function calculates the total sum of a specified column across all rows or within
    partitions if partition columns are specified.

    Args:
        df (DataFrame): Input DataFrame.
        value_col (str): The column to sum.
        partition_cols (Optional[Union[str, List[str]]], optional): Column(s) to partition by.
            If None, calculates the grand total across all rows. Defaults to None.
        new_col_name (Optional[str], optional): Name of the new column containing the total.
            If not provided, defaults to "{value_col}_total" or "{value_col}_total_partitioned".

    Returns:
        DataFrame: DataFrame with an additional column containing the total sum.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Grand total (total over all)
        >>> result = add_total(df, "value")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+-----------+
        |group| id|value|value_total|
        +-----+---+-----+-----------+
        |    A|  1|  100|       1350|
        |    A|  2|  200|       1350|
        |    A|  3|  300|       1350|
        |    B|  1|  150|       1350|
        |    B|  2|  250|       1350|
        |    B|  3|  350|       1350|
        +-----+---+-----+-----------+
        >>>
        >>> # Partitioned total
        >>> result = add_total(df, "value", "group", "group_total")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+-----------+
        |group| id|value|group_total|
        +-----+---+-----+-----------+
        |    A|  1|  100|        600|
        |    A|  2|  200|        600|
        |    A|  3|  300|        600|
        |    B|  1|  150|        750|
        |    B|  2|  250|        750|
        |    B|  3|  350|        750|
        +-----+---+-----+-----------+

    """
    # Determine the new column name
    if new_col_name is None:
        new_col_name = (
            f"{value_col}_total_partitioned" if partition_cols else f"{value_col}_total"
        )

    # Create the window specification
    if partition_cols:
        # Ensure partition_cols is a list
        if isinstance(partition_cols, str):
            partition_cols = [partition_cols]
        window_spec = Window.partitionBy(*partition_cols)
    else:
        # For grand total, use an empty partitionBy
        window_spec = Window.partitionBy()

    # Add the total column
    return df.withColumn(new_col_name, F.sum(F.col(value_col)).over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_total = add_total


def add_running_total(
    df: DataFrame,
    value_col: str,
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    partition_cols: Optional[Union[str, List[str]]] = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with the running total of a specified column, optionally partitioned.

    This function calculates the running total of a specified column across all rows or within
    partitions if partition columns are specified.

    Args:
        df (DataFrame): Input DataFrame.
        value_col (str): The column to calculate the running total for.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        partition_cols (Optional[Union[str, List[str]]], optional): Column(s) to partition by.
            If None, calculates the running total across all rows. Defaults to None.
        new_col_name (Optional[str], optional): Name of the new column containing the running total.
            If not provided, defaults to "{value_col}_running_total" or "{value_col}_running_total_partitioned".

    Returns:
        DataFrame: DataFrame with an additional column containing the running total.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Overall running total
        >>> result = add_running_total(df, "value", "id")
        >>> result.orderBy("id", "group").show()
        +-----+---+-----+---------------------+
        |group| id|value|value_running_total  |
        +-----+---+-----+---------------------+
        |    A|  1|  100|                 100 |
        |    B|  1|  150|                 250 |
        |    A|  2|  200|                 450 |
        |    B|  2|  250|                 700 |
        |    A|  3|  300|                1000 |
        |    B|  3|  350|                1350 |
        +-----+---+-----+---------------------+
        >>>
        >>> # Partitioned running total
        >>> result = add_running_total(df, "value", "id", "group", "group_running_total")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+--------------------+
        |group| id|value|group_running_total |
        +-----+---+-----+--------------------+
        |    A|  1|  100|                100 |
        |    A|  2|  200|                300 |
        |    A|  3|  300|                600 |
        |    B|  1|  150|                150 |
        |    B|  2|  250|                400 |
        |    B|  3|  350|                750 |
        +-----+---+-----+--------------------+

    """
    # Determine the new column name
    if new_col_name is None:
        new_col_name = (
            f"{value_col}_running_total_partitioned"
            if partition_cols
            else f"{value_col}_running_total"
        )

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    if partition_cols:
        # Ensure partition_cols is a list
        if isinstance(partition_cols, str):
            partition_cols = [partition_cols]
        window_spec = (
            Window.partitionBy(*partition_cols)
            .orderBy(*order_exprs)
            .rowsBetween(Window.unboundedPreceding, 0)
        )
    else:
        # For overall running total, use an empty partitionBy
        window_spec = Window.orderBy(*order_exprs).rowsBetween(
            Window.unboundedPreceding, 0
        )

    # Add the running total column
    return df.withColumn(new_col_name, F.sum(F.col(value_col)).over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_running_total = add_running_total


def add_running_min(
    df: DataFrame,
    value_col: str,
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    partition_cols: Optional[Union[str, List[str]]] = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with the running minimum of a specified column, optionally partitioned.

    This function calculates the running minimum of a specified column across all rows or within
    partitions if partition columns are specified.

    Args:
        df (DataFrame): Input DataFrame.
        value_col (str): The column to calculate the running minimum for.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        partition_cols (Optional[Union[str, List[str]]], optional): Column(s) to partition by.
            If None, calculates the running minimum across all rows. Defaults to None.
        new_col_name (Optional[str], optional): Name of the new column containing the running minimum.
            If not provided, defaults to "{value_col}_running_min" or "{value_col}_running_min_partitioned".

    Returns:
        DataFrame: DataFrame with an additional column containing the running minimum.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 300),
        ...     ("A", 2, 100),
        ...     ("A", 3, 200),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 50)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Overall running minimum
        >>> result = add_running_min(df, "value", "id")
        >>> result.orderBy("id", "group").show()
        +-----+---+-----+------------------+
        |group| id|value|    value_running_min|
        +-----+---+-----+------------------+
        |    A|  1|  300|               300|
        |    B|  1|  150|               150|
        |    A|  2|  100|               100|
        |    B|  2|  250|               100|
        |    A|  3|  200|               100|
        |    B|  3|   50|                50|
        +-----+---+-----+------------------+
        >>>
        >>> # Partitioned running minimum
        >>> result = add_running_min(df, "value", "id", "group", "group_running_min")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+------------------+
        |group| id|value|  group_running_min|
        +-----+---+-----+------------------+
        |    A|  1|  300|               300|
        |    A|  2|  100|               100|
        |    A|  3|  200|               100|
        |    B|  1|  150|               150|
        |    B|  2|  250|               150|
        |    B|  3|   50|                50|
        +-----+---+-----+------------------+

    """
    # Determine the new column name
    if new_col_name is None:
        new_col_name = (
            f"{value_col}_running_min_partitioned"
            if partition_cols
            else f"{value_col}_running_min"
        )

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    if partition_cols:
        # Ensure partition_cols is a list
        if isinstance(partition_cols, str):
            partition_cols = [partition_cols]
        window_spec = (
            Window.partitionBy(*partition_cols)
            .orderBy(*order_exprs)
            .rowsBetween(Window.unboundedPreceding, 0)
        )
    else:
        # For overall running minimum, use an empty partitionBy
        window_spec = Window.orderBy(*order_exprs).rowsBetween(
            Window.unboundedPreceding, 0
        )

    # Add the running minimum column
    return df.withColumn(new_col_name, F.min(F.col(value_col)).over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_running_min = add_running_min


def add_running_max(
    df: DataFrame,
    value_col: str,
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    partition_cols: Optional[Union[str, List[str]]] = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with the running maximum of a specified column, optionally partitioned.

    This function calculates the running maximum of a specified column across all rows or within
    partitions if partition columns are specified.

    Args:
        df (DataFrame): Input DataFrame.
        value_col (str): The column to calculate the running maximum for.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        partition_cols (Optional[Union[str, List[str]]], optional): Column(s) to partition by.
            If None, calculates the running maximum across all rows. Defaults to None.
        new_col_name (Optional[str], optional): Name of the new column containing the running maximum.
            If not provided, defaults to "{value_col}_running_max" or "{value_col}_running_max_partitioned".

    Returns:
        DataFrame: DataFrame with an additional column containing the running maximum.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 300),
        ...     ("A", 3, 200),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Overall running maximum
        >>> result = add_running_max(df, "value", "id")
        >>> result.orderBy("id", "group").show()
        +-----+---+-----+------------------+
        |group| id|value|    value_running_max|
        +-----+---+-----+------------------+
        |    A|  1|  100|               100|
        |    B|  1|  150|               150|
        |    A|  2|  300|               300|
        |    B|  2|  250|               300|
        |    A|  3|  200|               300|
        |    B|  3|  350|               350|
        +-----+---+-----+------------------+
        >>>
        >>> # Partitioned running maximum
        >>> result = add_running_max(df, "value", "id", "group", "group_running_max")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+------------------+
        |group| id|value|  group_running_max|
        +-----+---+-----+------------------+
        |    A|  1|  100|               100|
        |    A|  2|  300|               300|
        |    A|  3|  200|               300|
        |    B|  1|  150|               150|
        |    B|  2|  250|               250|
        |    B|  3|  350|               350|
        +-----+---+-----+------------------+

    """
    # Determine the new column name
    if new_col_name is None:
        new_col_name = (
            f"{value_col}_running_max_partitioned"
            if partition_cols
            else f"{value_col}_running_max"
        )

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    if partition_cols:
        # Ensure partition_cols is a list
        if isinstance(partition_cols, str):
            partition_cols = [partition_cols]
        window_spec = (
            Window.partitionBy(*partition_cols)
            .orderBy(*order_exprs)
            .rowsBetween(Window.unboundedPreceding, 0)
        )
    else:
        # For overall running maximum, use an empty partitionBy
        window_spec = Window.orderBy(*order_exprs).rowsBetween(
            Window.unboundedPreceding, 0
        )

    # Add the running maximum column
    return df.withColumn(new_col_name, F.max(F.col(value_col)).over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_running_max = add_running_max


def add_running_count(
    df: DataFrame,
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    partition_cols: Optional[Union[str, List[str]]] = None,
    new_col_name: Optional[str] = "running_count",
) -> DataFrame:
    """
    Add a new column with the running count of rows, optionally partitioned.

    This function calculates the running count of rows across all rows or within
    partitions if partition columns are specified.

    Args:
        df (DataFrame): Input DataFrame.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        partition_cols (Optional[Union[str, List[str]]], optional): Column(s) to partition by.
            If None, calculates the running count across all rows. Defaults to None.
        new_col_name (Optional[str], optional): Name of the new column containing the running count.
            Defaults to "running_count".

    Returns:
        DataFrame: DataFrame with an additional column containing the running count.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Overall running count
        >>> result = add_running_count(df, "id")
        >>> result.orderBy("id", "group").show()
        +-----+---+-----+-------------+
        |group| id|value|running_count|
        +-----+---+-----+-------------+
        |    A|  1|  100|            1|
        |    B|  1|  150|            2|
        |    A|  2|  200|            3|
        |    B|  2|  250|            4|
        |    A|  3|  300|            5|
        |    B|  3|  350|            6|
        +-----+---+-----+-------------+
        >>>
        >>> # Partitioned running count
        >>> result = add_running_count(df, "id", "group", "group_running_count")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+-------------------+
        |group| id|value|group_running_count|
        +-----+---+-----+-------------------+
        |    A|  1|  100|                  1|
        |    A|  2|  200|                  2|
        |    A|  3|  300|                  3|
        |    B|  1|  150|                  1|
        |    B|  2|  250|                  2|
        |    B|  3|  350|                  3|
        +-----+---+-----+-------------------+

    """
    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    if partition_cols:
        # Ensure partition_cols is a list
        if isinstance(partition_cols, str):
            partition_cols = [partition_cols]
        window_spec = (
            Window.partitionBy(*partition_cols)
            .orderBy(*order_exprs)
            .rowsBetween(Window.unboundedPreceding, 0)
        )
    else:
        # For overall running count, use an empty partitionBy
        window_spec = Window.orderBy(*order_exprs).rowsBetween(
            Window.unboundedPreceding, 0
        )

    # Add the running count column
    return df.withColumn(new_col_name, F.count("*").over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_running_count = add_running_count


def add_running_average(
    df: DataFrame,
    value_col: str,
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    partition_cols: Optional[Union[str, List[str]]] = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with the running average of a specified column, optionally partitioned.

    This function calculates the running average of a specified column across all rows or within
    partitions if partition columns are specified.

    Args:
        df (DataFrame): Input DataFrame.
        value_col (str): The column to calculate the running average for.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        partition_cols (Optional[Union[str, List[str]]], optional): Column(s) to partition by.
            If None, calculates the running average across all rows. Defaults to None.
        new_col_name (Optional[str], optional): Name of the new column containing the running average.
            If not provided, defaults to "{value_col}_running_avg" or "{value_col}_running_avg_partitioned".

    Returns:
        DataFrame: DataFrame with an additional column containing the running average.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Overall running average
        >>> result = add_running_average(df, "value", "id")
        >>> result.orderBy("id", "group").show()
        +-----+---+-----+------------------+
        |group| id|value|    value_running_avg|
        +-----+---+-----+------------------+
        |    A|  1|  100|             100.0|
        |    B|  1|  150|             125.0|
        |    A|  2|  200|             150.0|
        |    B|  2|  250|             175.0|
        |    A|  3|  300|             200.0|
        |    B|  3|  350|             225.0|
        +-----+---+-----+------------------+
        >>>
        >>> # Partitioned running average
        >>> result = add_running_average(df, "value", "id", "group", "group_running_avg")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+------------------+
        |group| id|value|  group_running_avg|
        +-----+---+-----+------------------+
        |    A|  1|  100|             100.0|
        |    A|  2|  200|             150.0|
        |    A|  3|  300|             200.0|
        |    B|  1|  150|             150.0|
        |    B|  2|  250|             200.0|
        |    B|  3|  350|             250.0|
        +-----+---+-----+------------------+

    """
    # Determine the new column name
    if new_col_name is None:
        new_col_name = (
            f"{value_col}_running_avg_partitioned"
            if partition_cols
            else f"{value_col}_running_avg"
        )

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    if partition_cols:
        # Ensure partition_cols is a list
        if isinstance(partition_cols, str):
            partition_cols = [partition_cols]
        window_spec = (
            Window.partitionBy(*partition_cols)
            .orderBy(*order_exprs)
            .rowsBetween(Window.unboundedPreceding, 0)
        )
    else:
        # For overall running average, use an empty partitionBy
        window_spec = Window.orderBy(*order_exprs).rowsBetween(
            Window.unboundedPreceding, 0
        )

    # Add the running average column
    return df.withColumn(new_col_name, F.avg(F.col(value_col)).over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_running_average = add_running_average


def add_cume_dist(
    df: DataFrame,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    new_col_name: Optional[str] = "cumulative_distribution",
) -> DataFrame:
    """
    Add a new column with the cumulative distribution of values within partitions.

    This function calculates the cumulative distribution of a value within a partition,
    representing the percentage of rows with values less than or equal to the current row's value.

    Args:
        df (DataFrame): Input DataFrame.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        new_col_name (str, optional): Name of the new column containing the cumulative distribution values.
            Defaults to "cumulative_distribution".

    Returns:
        DataFrame: DataFrame with an additional column containing the cumulative distribution values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("A", 4, 300),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage
        >>> result = add_cume_dist(df, "group", "value")
        >>> result.orderBy("group", "value").show()
        +-----+---+-----+------------------------+
        |group| id|value|cumulative_distribution |
        +-----+---+-----+------------------------+
        |    A|  1|  100|                    0.25|
        |    A|  2|  200|                     0.5|
        |    A|  3|  300|                     1.0|
        |    A|  4|  300|                     1.0|
        |    B|  1|  150|   0.3333333333333333   |
        |    B|  2|  250|   0.6666666666666666   |
        |    B|  3|  350|                     1.0|
        +-----+---+-----+------------------------+
        >>>
        >>> # Advanced usage with multiple partition and order columns
        >>> result = add_cume_dist(
        ...     df,
        ...     "group",
        ...     [("value", "desc"), ("id", "asc")],
        ...     new_col_name="reverse_cume_dist"
        ... )
        >>> result.orderBy("group", F.desc("value"), "id").show()
        +-----+---+-----+------------------+
        |group| id|value| reverse_cume_dist|
        +-----+---+-----+------------------+
        |    A|  3|  300|               0.5|
        |    A|  4|  300|               0.5|
        |    A|  2|  200|              0.75|
        |    A|  1|  100|               1.0|
        |    B|  3|  350|  0.3333333333333333|
        |    B|  2|  250|  0.6666666666666666|
        |    B|  1|  150|               1.0|
        +-----+---+-----+------------------+

    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Add the cumulative distribution column
    return df.withColumn(new_col_name, F.cume_dist().over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_cume_dist = add_cume_dist


def add_dense_rank(
    df: DataFrame,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    new_col_name: str = "dense_rank",
) -> DataFrame:
    """
    Add a new column with dense ranks within partitions, allowing for ties without rank gaps.

    This function assigns a dense rank to each row within the partition. If there are ties
    in the order, they receive the same rank, and the next rank is not skipped.

    Args:
        df (DataFrame): Input DataFrame.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        new_col_name (str, optional): Name of the new column containing the dense rank.
            Defaults to "dense_rank".

    Returns:
        DataFrame: DataFrame with an additional column containing the dense rank.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 100),
        ...     ("A", 3, 200),
        ...     ("B", 1, 200),
        ...     ("B", 2, 300),
        ...     ("B", 3, 300)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage
        >>> result = add_dense_rank(df, "group", "value")
        >>> result.orderBy("group", "value", "id").show()
        +-----+---+-----+----------+
        |group| id|value|dense_rank|
        +-----+---+-----+----------+
        |    A|  1|  100|         1|
        |    A|  2|  100|         1|
        |    A|  3|  200|         2|
        |    B|  1|  200|         1|
        |    B|  2|  300|         2|
        |    B|  3|  300|         2|
        +-----+---+-----+----------+
        >>>
        >>> # Advanced usage with multiple partition and order columns
        >>> result = add_dense_rank(
        ...     df,
        ...     "group",
        ...     [("value", "desc"), ("id", "asc")],
        ...     "custom_dense_rank"
        ... )
        >>> result.orderBy("group", F.desc("value"), "id").show()
        +-----+---+-----+------------------+
        |group| id|value|custom_dense_rank |
        +-----+---+-----+------------------+
        |    A|  3|  200|                 1|
        |    A|  1|  100|                 2|
        |    A|  2|  100|                 2|
        |    B|  2|  300|                 1|
        |    B|  3|  300|                 1|
        |    B|  1|  200|                 2|
        +-----+---+-----+------------------+

    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Add the dense rank column
    return df.withColumn(new_col_name, F.dense_rank().over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_dense_rank = add_dense_rank


def add_ntile(
    df: DataFrame,
    n_buckets: int,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    new_col_name: str = "ntile",
) -> DataFrame:
    """
    Add a new column with ntile buckets within partitions.

    This function divides the result set into the specified number of roughly equal parts
    and assigns a bucket number to each row. It is useful for creating quartiles, quintiles,
    or any desired number of buckets.

    Args:
        df (DataFrame): Input DataFrame.
        n_buckets (int): Number of buckets to divide the data into.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        new_col_name (str, optional): Name of the new column containing the ntile bucket.
            Defaults to "ntile".

    Returns:
        DataFrame: DataFrame with an additional column containing the ntile bucket.

    Raises:
        ValueError: If n_buckets is less than 1.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 10),
        ...     ("A", 2, 20),
        ...     ("A", 3, 30),
        ...     ("A", 4, 40),
        ...     ("A", 5, 50),
        ...     ("B", 1, 10),
        ...     ("B", 2, 20),
        ...     ("B", 3, 30),
        ...     ("B", 4, 40)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage - create quartiles
        >>> result = add_ntile(df, 4, "group", "value")
        >>> result.orderBy("group", "value").show()
        +-----+---+-----+-----+
        |group| id|value|ntile|
        +-----+---+-----+-----+
        |    A|  1|   10|    1|
        |    A|  2|   20|    1|
        |    A|  3|   30|    2|
        |    A|  4|   40|    3|
        |    A|  5|   50|    4|
        |    B|  1|   10|    1|
        |    B|  2|   20|    2|
        |    B|  3|   30|    3|
        |    B|  4|   40|    4|
        +-----+---+-----+-----+
        >>>
        >>> # Advanced usage with multiple partition and order columns
        >>> result = add_ntile(
        ...     df,
        ...     3,
        ...     "group",
        ...     [("value", "desc"), ("id", "asc")],
        ...     "custom_ntile"
        ... )
        >>> result.orderBy("group", F.desc("value"), "id").show()
        +-----+---+-----+-----------+
        |group| id|value|custom_ntile|
        +-----+---+-----+-----------+
        |    A|  5|   50|          1|
        |    A|  4|   40|          1|
        |    A|  3|   30|          2|
        |    A|  2|   20|          2|
        |    A|  1|   10|          3|
        |    B|  4|   40|          1|
        |    B|  3|   30|          1|
        |    B|  2|   20|          2|
        |    B|  1|   10|          3|
        +-----+---+-----+-----------+

    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be a positive integer")

    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Add the ntile column
    return df.withColumn(new_col_name, F.ntile(n_buckets).over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_ntile = add_ntile


def add_percent_rank(
    df: DataFrame,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    new_col_name: str = "percent_rank",
) -> DataFrame:
    """
    Add a new column with percent ranks within partitions.

    This function calculates the relative rank of each row within the partition as a percentage.
    The result is between 0 and 1, with 0 representing the minimum rank and 1 representing the maximum rank.

    Args:
        df (DataFrame): Input DataFrame.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        new_col_name (str, optional): Name of the new column containing the percent rank.
            Defaults to "percent_rank".

    Returns:
        DataFrame: DataFrame with an additional column containing the percent rank.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("A", 4, 300),
        ...     ("B", 1, 100),
        ...     ("B", 2, 200),
        ...     ("B", 3, 300)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage
        >>> result = add_percent_rank(df, "group", "value")
        >>> result.orderBy("group", "value").show()
        +-----+---+-----+------------------+
        |group| id|value|       percent_rank|
        +-----+---+-----+------------------+
        |    A|  1|  100|               0.0|
        |    A|  2|  200|0.3333333333333333|
        |    A|  3|  300|0.6666666666666666|
        |    A|  4|  300|0.6666666666666666|
        |    B|  1|  100|               0.0|
        |    B|  2|  200|               0.5|
        |    B|  3|  300|               1.0|
        +-----+---+-----+------------------+
        >>>
        >>> # Advanced usage with multiple partition and order columns
        >>> result = add_percent_rank(
        ...     df,
        ...     "group",
        ...     [("value", "desc"), ("id", "asc")],
        ...     "custom_percent_rank"
        ... )
        >>> result.orderBy("group", F.desc("value"), "id").show()
        +-----+---+-----+--------------------+
        |group| id|value| custom_percent_rank|
        +-----+---+-----+--------------------+
        |    A|  3|  300|                 0.0|
        |    A|  4|  300|                 0.0|
        |    A|  2|  200|0.6666666666666666  |
        |    A|  1|  100|                 1.0|
        |    B|  3|  300|                 0.0|
        |    B|  2|  200|                 0.5|
        |    B|  1|  100|                 1.0|
        +-----+---+-----+--------------------+

    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Add the percent rank column
    return df.withColumn(new_col_name, F.percent_rank().over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_percent_rank = add_percent_rank


def add_rank(
    df: DataFrame,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    new_col_name: str = "rank",
) -> DataFrame:
    """
    Add a new column with ranks within partitions, allowing for ties.

    This function assigns a rank to each row within the partition. If there are ties
    in the order, they receive the same rank, and the next rank is skipped.

    Args:
        df (DataFrame): Input DataFrame.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        new_col_name (str, optional): Name of the new column containing the rank.
            Defaults to "rank".

    Returns:
        DataFrame: DataFrame with an additional column containing the rank.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 100),
        ...     ("A", 3, 200),
        ...     ("B", 1, 200),
        ...     ("B", 2, 300),
        ...     ("B", 3, 300)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage
        >>> result = add_rank(df, "group", "value")
        >>> result.orderBy("group", "value", "id").show()
        +-----+---+-----+----+
        |group| id|value|rank|
        +-----+---+-----+----+
        |    A|  1|  100|   1|
        |    A|  2|  100|   1|
        |    A|  3|  200|   3|
        |    B|  1|  200|   1|
        |    B|  2|  300|   2|
        |    B|  3|  300|   2|
        +-----+---+-----+----+
        >>>
        >>> # Advanced usage with multiple partition and order columns
        >>> result = add_rank(
        ...     df,
        ...     "group",
        ...     [("value", "desc"), ("id", "asc")],
        ...     "custom_rank"
        ... )
        >>> result.orderBy("group", F.desc("value"), "id").show()
        +-----+---+-----+-----------+
        |group| id|value|custom_rank|
        +-----+---+-----+-----------+
        |    A|  3|  200|          1|
        |    A|  1|  100|          2|
        |    A|  2|  100|          2|
        |    B|  2|  300|          1|
        |    B|  3|  300|          1|
        |    B|  1|  200|          3|
        +-----+---+-----+-----------+

    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Add the rank column
    return df.withColumn(new_col_name, F.rank().over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_rank = add_rank


def add_row_number_rank(
    df: DataFrame,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    new_col_name: str = "row_number_rank",
) -> DataFrame:
    """
    Add a new column with row number ranks within partitions.

    This function assigns a unique integer value to each row within the partition,
    with no ties. If two rows have the same values in the order columns, they will
    still receive distinct row numbers based on their position in the DataFrame.

    Args:
        df (DataFrame): Input DataFrame.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        new_col_name (str, optional): Name of the new column containing the row number rank.
            Defaults to "row_number_rank".

    Returns:
        DataFrame: DataFrame with an additional column containing the row number rank.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", "X", 1, 100),
        ...     ("A", "Y", 2, 150),
        ...     ("A", "X", 3, 200),
        ...     ("B", "X", 1, 250),
        ...     ("B", "Y", 2, 300),
        ...     ("B", "Y", 3, 350)
        ... ], ["group", "subgroup", "id", "value"])
        >>>
        >>> # Simple usage
        >>> result = add_row_number_rank(df, "group", "value")
        >>> result.orderBy("group", "value").show()
        +-----+--------+---+-----+-----------------+
        |group|subgroup| id|value|row_number_rank  |
        +-----+--------+---+-----+-----------------+
        |    A|       X|  1|  100|                1|
        |    A|       Y|  2|  150|                2|
        |    A|       X|  3|  200|                3|
        |    B|       X|  1|  250|                1|
        |    B|       Y|  2|  300|                2|
        |    B|       Y|  3|  350|                3|
        +-----+--------+---+-----+-----------------+
        >>>
        >>> # Advanced usage with multiple partition and order columns
        >>> result = add_row_number_rank(
        ...     df,
        ...     ["group", "subgroup"],
        ...     [("id", "desc"), ("value", "asc")],
        ...     "custom_rank"
        ... )
        >>> result.orderBy("group", "subgroup", "id").show()
        +-----+--------+---+-----+-----------+
        |group|subgroup| id|value|custom_rank|
        +-----+--------+---+-----+-----------+
        |    A|       X|  1|  100|          2|
        |    A|       X|  3|  200|          1|
        |    A|       Y|  2|  150|          1|
        |    B|       X|  1|  250|          1|
        |    B|       Y|  2|  300|          2|
        |    B|       Y|  3|  350|          1|
        +-----+--------+---+-----+-----------+

    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Add the row number column
    return df.withColumn(new_col_name, F.row_number().over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_row_number_rank = add_row_number_rank


def add_first_value(
    df: DataFrame,
    value_col: str,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    new_col_name: Optional[str] = None,
    ignore_nulls: bool = False,
) -> DataFrame:
    """
    Add a new column with the first value within partitions.

    This function adds a column containing the value of a specified column from the first row
    of each partition, based on the given partition and order.

    Args:
        df (DataFrame): Input DataFrame.
        value_col (str): The column to get the first value from.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        new_col_name (str, optional): Name of the new column containing the first values.
            If not provided, defaults to "{value_col}_first".
        ignore_nulls (bool, optional): If True, null values are ignored when determining
            the first value. Defaults to False.

    Returns:
        DataFrame: DataFrame with an additional column containing the first values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage
        >>> result = add_first_value(df, "value", "group", "id")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+------------+
        |group| id|value|value_first |
        +-----+---+-----+------------+
        |    A|  1|  100|         100|
        |    A|  2|  200|         100|
        |    A|  3|  300|         100|
        |    B|  1|  150|         150|
        |    B|  2|  250|         150|
        |    B|  3|  350|         150|
        +-----+---+-----+------------+
        >>>
        >>> # Advanced usage with descending order
        >>> result = add_first_value(
        ...     df,
        ...     "value",
        ...     "group",
        ...     [("id", "desc")],
        ...     new_col_name="highest_value"
        ... )
        >>> result.orderBy("group", F.desc("id")).show()
        +-----+---+-----+-------------+
        |group| id|value|highest_value|
        +-----+---+-----+-------------+
        |    A|  3|  300|          300|
        |    A|  2|  200|          300|
        |    A|  1|  100|          300|
        |    B|  3|  350|          350|
        |    B|  2|  250|          350|
        |    B|  1|  150|          350|
        +-----+---+-----+-------------+

    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{value_col}_first"

    # Add the first value column
    if ignore_nulls:
        return df.withColumn(
            new_col_name, F.first(value_col, ignorenulls=True).over(window_spec)
        )
    else:
        return df.withColumn(new_col_name, F.first(value_col).over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_first_value = add_first_value


def add_lag(
    df: DataFrame,
    lag_col: str,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    lag_offset: int = 1,
    default_value: Any = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with lagged values within partitions.

    This function adds a column containing the value of a specified column from a previous row,
    based on the given partition and order.

    Args:
        df (DataFrame): Input DataFrame.
        lag_col (str): The column to lag.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        lag_offset (int, optional): The number of rows to lag. Defaults to 1.
        default_value (Any, optional): The default value to use when there is no previous row.
            Defaults to None.
        new_col_name (str, optional): Name of the new column containing the lagged values.
            If not provided, defaults to "{lag_col}_lag_{lag_offset}".

    Returns:
        DataFrame: DataFrame with an additional column containing the lagged values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage
        >>> result = add_lag(df, "value", "group", "id")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+-----------+
        |group| id|value|value_lag_1|
        +-----+---+-----+-----------+
        |    A|  1|  100|       null|
        |    A|  2|  200|        100|
        |    A|  3|  300|        200|
        |    B|  1|  150|       null|
        |    B|  2|  250|        150|
        |    B|  3|  350|        250|
        +-----+---+-----+-----------+
        >>>
        >>> # Advanced usage
        >>> result = add_lag(
        ...     df,
        ...     "value",
        ...     "group",
        ...     [("id", "desc")],
        ...     lag_offset=2,
        ...     default_value=0,
        ...     new_col_name="value_lag_2_desc"
        ... )
        >>> result.orderBy("group", F.desc("id")).show()
        +-----+---+-----+----------------+
        |group| id|value|value_lag_2_desc|
        +-----+---+-----+----------------+
        |    A|  3|  300|               0|
        |    A|  2|  200|               0|
        |    A|  1|  100|             300|
        |    B|  3|  350|               0|
        |    B|  2|  250|               0|
        |    B|  1|  150|             350|
        +-----+---+-----+----------------+

    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{lag_col}_lag_{lag_offset}"

    # Add the lag column
    return df.withColumn(
        new_col_name, F.lag(lag_col, lag_offset, default_value).over(window_spec)
    )


# Add the function to DataFrame class for use with transform method
DataFrame.add_lag = add_lag


def add_last_value(
    df: DataFrame,
    value_col: str,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    new_col_name: Optional[str] = None,
    ignore_nulls: bool = False,
) -> DataFrame:
    """
    Add a new column with the last value within partitions.

    This function adds a column containing the value of a specified column from the last row
    of each partition, based on the given partition and order.

    Args:
        df (DataFrame): Input DataFrame.
        value_col (str): The column to get the last value from.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        new_col_name (str, optional): Name of the new column containing the last values.
            If not provided, defaults to "{value_col}_last".
        ignore_nulls (bool, optional): If True, null values are ignored when determining
            the last value. Defaults to False.

    Returns:
        DataFrame: DataFrame with an additional column containing the last values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage
        >>> result = add_last_value(df, "value", "group", "id")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+-----------+
        |group| id|value|value_last |
        +-----+---+-----+-----------+
        |    A|  1|  100|        300|
        |    A|  2|  200|        300|
        |    A|  3|  300|        300|
        |    B|  1|  150|        350|
        |    B|  2|  250|        350|
        |    B|  3|  350|        350|
        +-----+---+-----+-----------+
        >>>
        >>> # Advanced usage with descending order
        >>> result = add_last_value(
        ...     df,
        ...     "value",
        ...     "group",
        ...     [("id", "desc")],
        ...     new_col_name="lowest_value"
        ... )
        >>> result.orderBy("group", F.desc("id")).show()
        +-----+---+-----+------------+
        |group| id|value|lowest_value|
        +-----+---+-----+------------+
        |    A|  3|  300|         100|
        |    A|  2|  200|         100|
        |    A|  1|  100|         100|
        |    B|  3|  350|         150|
        |    B|  2|  250|         150|
        |    B|  1|  150|         150|
        +-----+---+-----+------------+

    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = (
        Window.partitionBy(*partition_cols)
        .orderBy(*order_exprs)
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{value_col}_last"

    # Add the last value column
    if ignore_nulls:
        return df.withColumn(
            new_col_name, F.last(value_col, ignorenulls=True).over(window_spec)
        )
    else:
        return df.withColumn(new_col_name, F.last(value_col).over(window_spec))


# Add the function to DataFrame class for use with transform method
DataFrame.add_last_value = add_last_value


def add_lead(
    df: DataFrame,
    lead_col: str,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    lead_offset: int = 1,
    default_value: Any = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with lead values within partitions.

    This function adds a column containing the value of a specified column from a subsequent row,
    based on the given partition and order.

    Args:
        df (DataFrame): Input DataFrame.
        lead_col (str): The column to lead.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        lead_offset (int, optional): The number of rows to lead. Defaults to 1.
        default_value (Any, optional): The default value to use when there is no subsequent row.
            Defaults to None.
        new_col_name (str, optional): Name of the new column containing the lead values.
            If not provided, defaults to "{lead_col}_lead_{lead_offset}".

    Returns:
        DataFrame: DataFrame with an additional column containing the lead values.

    Raises:
        ValueError: If lead_offset is less than 1.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage
        >>> result = add_lead(df, "value", "group", "id")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+------------+
        |group| id|value|value_lead_1|
        +-----+---+-----+------------+
        |    A|  1|  100|         200|
        |    A|  2|  200|         300|
        |    A|  3|  300|        null|
        |    B|  1|  150|         250|
        |    B|  2|  250|         350|
        |    B|  3|  350|        null|
        +-----+---+-----+------------+
        >>>
        >>> # Advanced usage
        >>> result = add_lead(
        ...     df,
        ...     "value",
        ...     "group",
        ...     [("id", "desc")],
        ...     lead_offset=2,
        ...     default_value=0,
        ...     new_col_name="value_lead_2_desc"
        ... )
        >>> result.orderBy("group", F.desc("id")).show()
        +-----+---+-----+-----------------+
        |group| id|value|value_lead_2_desc|
        +-----+---+-----+-----------------+
        |    A|  3|  300|              100|
        |    A|  2|  200|                0|
        |    A|  1|  100|                0|
        |    B|  3|  350|              150|
        |    B|  2|  250|                0|
        |    B|  1|  150|                0|
        +-----+---+-----+-----------------+

    """
    if lead_offset < 1:
        raise ValueError("lead_offset must be a positive integer")

    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{lead_col}_lead_{lead_offset}"

    # Add the lead column
    return df.withColumn(
        new_col_name, F.lead(lead_col, lead_offset, default_value).over(window_spec)
    )


# Add the function to DataFrame class for use with transform method
DataFrame.add_lead = add_lead


def add_nth_value(
    df: DataFrame,
    value_col: str,
    n: int,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    new_col_name: Optional[str] = None,
    ignore_nulls: bool = False,
) -> DataFrame:
    """
    Add a new column with the Nth value within partitions.

    This function adds a column containing the value of a specified column from the Nth row
    of each partition, based on the given partition and order.

    Args:
        df (DataFrame): Input DataFrame.
        value_col (str): The column to get the Nth value from.
        n (int): The position of the row to get the value from (1-based index).
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]): Column(s) to order by.
            Can be a string, list of strings, or list of tuples where each tuple is
            (column_name, sort_order) with sort_order being either "asc" or "desc".
        new_col_name (str, optional): Name of the new column containing the Nth values.
            If not provided, defaults to "{value_col}_nth_{n}".
        ignore_nulls (bool, optional): If True, null values are ignored when determining
            the Nth value. Defaults to False.

    Returns:
        DataFrame: DataFrame with an additional column containing the Nth values.

    Raises:
        ValueError: If n is less than 1.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 300),
        ...     ("A", 4, 400),
        ...     ("B", 1, 150),
        ...     ("B", 2, 250),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>>
        >>> # Simple usage - get the 2nd value
        >>> result = add_nth_value(df, "value", 2, "group", "id")
        >>> result.orderBy("group", "id").show()
        +-----+---+-----+------------+
        |group| id|value|value_nth_2 |
        +-----+---+-----+------------+
        |    A|  1|  100|         200|
        |    A|  2|  200|         200|
        |    A|  3|  300|         200|
        |    A|  4|  400|         200|
        |    B|  1|  150|         250|
        |    B|  2|  250|         250|
        |    B|  3|  350|         250|
        +-----+---+-----+------------+
        >>>
        >>> # Advanced usage with descending order
        >>> result = add_nth_value(
        ...     df,
        ...     "value",
        ...     3,
        ...     "group",
        ...     [("id", "desc")],
        ...     new_col_name="third_highest_value"
        ... )
        >>> result.orderBy("group", F.desc("id")).show()
        +-----+---+-----+--------------------+
        |group| id|value|third_highest_value |
        +-----+---+-----+--------------------+
        |    A|  4|  400|                 200|
        |    A|  3|  300|                 200|
        |    A|  2|  200|                 200|
        |    A|  1|  100|                 200|
        |    B|  3|  350|                 150|
        |    B|  2|  250|                 150|
        |    B|  1|  150|                 150|
        +-----+---+-----+--------------------+

    """
    if n < 1:
        raise ValueError("n must be a positive integer")

    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window_spec = (
        Window.partitionBy(*partition_cols)
        .orderBy(*order_exprs)
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{value_col}_nth_{n}"

    # Add the nth value column
    if ignore_nulls:
        return df.withColumn(
            new_col_name, F.expr(f"nth_value({value_col}, {n}, true)").over(window_spec)
        )
    else:
        return df.withColumn(
            new_col_name, F.expr(f"nth_value({value_col}, {n})").over(window_spec)
        )


# Add the function to DataFrame class for use with transform method
DataFrame.add_nth_value = add_nth_value


def add_window_collect_bottom_n(
    df: DataFrame,
    collect_col: str,
    n: int,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    window_spec: Optional[str] = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with the bottom N values collected within a window.

    This function collects the bottom N values of a specified column within
    the defined window, based on the given ordering.

    Args:
        df (DataFrame): Input DataFrame.
        collect_col (str): The column to collect bottom N values from.
        n (int): The number of bottom values to collect.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]):
            Column(s) to order by. Can be a string, list of strings, or list of tuples
            where each tuple is (column_name, sort_order) with sort_order being either "asc" or "desc".
        window_spec (Optional[str]): A string specifying the window frame, e.g.,
            "2 preceding and 2 following". If None, the entire partition is used.
        new_col_name (Optional[str]): Name of the new column containing the collected bottom N values.
            If not provided, defaults to "{collect_col}_bottom_{n}".


    Returns:
        DataFrame: DataFrame with an additional column containing the collected bottom N values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 150),
        ...     ("A", 4, 120),
        ...     ("B", 1, 300),
        ...     ("B", 2, 400),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>> result = add_window_collect_bottom_n(df, "value", 2, "group", [("value", "asc")])
        >>> result.orderBy("group", "value").show()
        +-----+---+-----+---------------+
        |group| id|value| value_bottom_2|
        +-----+---+-----+---------------+
        |    A|  1|  100|    [100, 120] |
        |    A|  4|  120|    [100, 120] |
        |    A|  3|  150|    [100, 120] |
        |    A|  2|  200|    [100, 120] |
        |    B|  1|  300|    [300, 350] |
        |    B|  3|  350|    [300, 350] |
        |    B|  2|  400|    [300, 350] |
        +-----+---+-----+---------------+
    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Apply the window frame specification if provided
    if window_spec:
        frame_parts = window_spec.split()
        if (
            len(frame_parts) == 4
            and frame_parts[1] == "preceding"
            and frame_parts[3] == "following"
        ):
            preceding = (
                int(frame_parts[0])
                if frame_parts[0] != "unbounded"
                else Window.unboundedPreceding
            )
            following = (
                int(frame_parts[2])
                if frame_parts[2] != "unbounded"
                else Window.unboundedFollowing
            )
            window = window.rowsBetween(preceding, following)
    else:
        # If no window_spec provided, use the entire partition
        window = window.rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{collect_col}_bottom_{n}"

    # Create a temporary rank column
    df_with_rank = df.withColumn("temp_rank", F.rank().over(window))

    # Collect the bottom N values
    df_with_bottom_n = df_with_rank.withColumn(
        new_col_name,
        F.collect_list(F.when(F.col("temp_rank") <= n, F.col(collect_col))).over(
            window
        ),
    )

    # Remove the temporary rank column
    return df_with_bottom_n.drop("temp_rank")


# Add the function to DataFrame class for use with transform method
DataFrame.add_window_collect_bottom_n = add_window_collect_bottom_n


def add_window_collect_distinct(
    df: DataFrame,
    collect_col: str,
    partition_cols: Union[str, List[str]],
    order_cols: Optional[Union[str, List[str], List[Tuple[str, str]]]] = None,
    window_spec: Optional[str] = None,
    new_col_name: Optional[str] = None,
    max_distinct: Optional[int] = None,
) -> DataFrame:
    """
    Add a new column with distinct values collected within a window.

    This function collects unique values of a specified column within
    the defined window, based on the given partitioning and ordering.

    Args:
        df (DataFrame): Input DataFrame.
        collect_col (str): The column to collect distinct values from.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Optional[Union[str, List[str], List[Tuple[str, str]]]]):
            Column(s) to order by. Can be None if no ordering is required.
        window_spec (Optional[str]): A string specifying the window frame, e.g.,
            "2 preceding and 2 following". If None, the entire partition is used.
        new_col_name (Optional[str]): Name of the new column containing the collected distinct values.
            If not provided, defaults to "{collect_col}_distinct".
        max_distinct (Optional[int]): Maximum number of distinct values to collect.
            If None, all distinct values are collected.

    Returns:
        DataFrame: DataFrame with an additional column containing the collected distinct values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, "X"),
        ...     ("A", 2, "Y"),
        ...     ("A", 3, "X"),
        ...     ("A", 4, "Z"),
        ...     ("B", 1, "P"),
        ...     ("B", 2, "Q"),
        ...     ("B", 3, "P")
        ... ], ["group", "id", "value"])
        >>> result = add_window_collect_distinct(df, "value", "group", "id", "2 preceding and current row")
        >>> result.orderBy("group", "id").show(truncate=False)
        +-----+---+-----+---------------+
        |group|id |value|value_distinct |
        +-----+---+-----+---------------+
        |A    |1  |X    |[X]            |
        |A    |2  |Y    |[X, Y]         |
        |A    |3  |X    |[X, Y]         |
        |A    |4  |Z    |[X, Y, Z]      |
        |B    |1  |P    |[P]            |
        |B    |2  |Q    |[P, Q]         |
        |B    |3  |P    |[P, Q]         |
        +-----+---+-----+---------------+
    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols if provided
    if order_cols:
        if isinstance(order_cols, str):
            order_cols = [order_cols]

        order_exprs = []
        for col in order_cols:
            if isinstance(col, tuple):
                order_exprs.append(
                    F.col(col[0]).cast("string").asc()
                    if col[1].lower() == "asc"
                    else F.col(col[0]).cast("string").desc()
                )
            else:
                order_exprs.append(F.col(col).cast("string").asc())

        # Create the window specification
        window = Window.partitionBy(*partition_cols).orderBy(*order_exprs)
    else:
        # If no order_cols provided, create a partitioned window without ordering
        window = Window.partitionBy(*partition_cols)

    # Apply the window frame specification if provided
    if window_spec:
        frame_parts = window_spec.split()
        if (
            len(frame_parts) == 4
            and frame_parts[1] == "preceding"
            and frame_parts[3] == "following"
        ):
            preceding = (
                int(frame_parts[0])
                if frame_parts[0] != "unbounded"
                else Window.unboundedPreceding
            )
            following = (
                int(frame_parts[2])
                if frame_parts[2] != "unbounded"
                else Window.unboundedFollowing
            )
            window = window.rowsBetween(preceding, following)
    else:
        # If no window_spec provided, use the entire partition
        window = window.rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{collect_col}_distinct"

    # Collect distinct values
    if max_distinct is not None:
        # If max_distinct is specified, use array_distinct and slice to limit the number of elements
        df = df.withColumn(
            new_col_name,
            F.expr(
                f"slice(array_distinct(collect_list({collect_col})), 1, {max_distinct})"
            ).over(window),
        )
    else:
        # If max_distinct is not specified, use collect_set to get all distinct values
        df = df.withColumn(new_col_name, F.collect_set(collect_col).over(window))

    return df


# Add the function to DataFrame class for use with transform method
DataFrame.add_window_collect_distinct = add_window_collect_distinct


def add_window_collect_first_n_distinct(
    df: DataFrame,
    collect_col: str,
    n: int,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    window_spec: Optional[str] = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with the first N distinct values collected within a window.

    This function collects the first N unique values of a specified column within
    the defined window, based on the given partitioning and ordering.

    Args:
        df (DataFrame): Input DataFrame.
        collect_col (str): The column to collect first N distinct values from.
        n (int): The number of distinct values to collect.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]):
            Column(s) to order by. Can be a string, list of strings, or list of tuples
            where each tuple is (column_name, sort_order) with sort_order being either "asc" or "desc".
        window_spec (Optional[str]): A string specifying the window frame, e.g.,
            "2 preceding and 2 following". If None, the entire partition is used.
        new_col_name (Optional[str]): Name of the new column containing the collected first N distinct values.
            If not provided, defaults to "{collect_col}_first_{n}_distinct".

    Returns:
        DataFrame: DataFrame with an additional column containing the collected first N distinct values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, "X"),
        ...     ("A", 2, "Y"),
        ...     ("A", 3, "X"),
        ...     ("A", 4, "Z"),
        ...     ("A", 5, "Y"),
        ...     ("B", 1, "P"),
        ...     ("B", 2, "Q"),
        ...     ("B", 3, "P"),
        ...     ("B", 4, "R")
        ... ], ["group", "id", "value"])
        >>> result = add_window_collect_first_n_distinct(df, "value", 2, "group", "id")
        >>> result.orderBy("group", "id").show(truncate=False)
        +-----+---+-----+-------------------------+
        |group|id |value|value_first_2_distinct   |
        +-----+---+-----+-------------------------+
        |A    |1  |X    |[X]                      |
        |A    |2  |Y    |[X, Y]                   |
        |A    |3  |X    |[X, Y]                   |
        |A    |4  |Z    |[X, Y]                   |
        |A    |5  |Y    |[X, Y]                   |
        |B    |1  |P    |[P]                      |
        |B    |2  |Q    |[P, Q]                   |
        |B    |3  |P    |[P, Q]                   |
        |B    |4  |R    |[P, Q]                   |
        +-----+---+-----+-------------------------+
    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Apply the window frame specification if provided
    if window_spec:
        frame_parts = window_spec.split()
        if (
            len(frame_parts) == 4
            and frame_parts[1] == "preceding"
            and frame_parts[3] == "following"
        ):
            preceding = (
                int(frame_parts[0])
                if frame_parts[0] != "unbounded"
                else Window.unboundedPreceding
            )
            following = (
                int(frame_parts[2])
                if frame_parts[2] != "unbounded"
                else Window.unboundedFollowing
            )
            window = window.rowsBetween(preceding, following)
    else:
        # If no window_spec provided, use the entire partition
        window = window.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{collect_col}_first_{n}_distinct"

    # Create a temporary column with row numbers for each distinct value
    df_with_row_num = df.withColumn(
        "temp_row_num",
        F.row_number().over(
            Window.partitionBy(*partition_cols, collect_col).orderBy(*order_exprs)
        ),
    )

    # Collect the first N distinct values
    df_with_first_n_distinct = df_with_row_num.withColumn(
        new_col_name,
        F.expr(
            f"slice(array_distinct(collect_list(case when temp_row_num = 1 then {collect_col} end)), 1, {n})"
        ).over(window),
    )

    # Remove the temporary row number column
    return df_with_first_n_distinct.drop("temp_row_num")


# Add the function to DataFrame class for use with transform method
DataFrame.add_window_collect_first_n_distinct = add_window_collect_first_n_distinct


def add_window_collect_top_n(
    df: DataFrame,
    collect_col: str,
    n: int,
    partition_cols: Union[str, List[str]],
    order_cols: Union[str, List[str], List[Tuple[str, str]]],
    window_spec: Optional[str] = None,
    new_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Add a new column with the top N values collected within a window.

    This function collects the top N values of a specified column within
    the defined window, based on the given ordering.

    Args:
        df (DataFrame): Input DataFrame.
        collect_col (str): The column to collect top N values from.
        n (int): The number of top values to collect.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Union[str, List[str], List[Tuple[str, str]]]):
            Column(s) to order by. Can be a string, list of strings, or list of tuples
            where each tuple is (column_name, sort_order) with sort_order being either "asc" or "desc".
        window_spec (Optional[str]): A string specifying the window frame, e.g.,
            "2 preceding and 2 following". If None, the entire partition is used.
        new_col_name (Optional[str]): Name of the new column containing the collected top N values.
            If not provided, defaults to "{collect_col}_top_{n}".

    Returns:
        DataFrame: DataFrame with an additional column containing the collected top N values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, 100),
        ...     ("A", 2, 200),
        ...     ("A", 3, 150),
        ...     ("A", 4, 120),
        ...     ("B", 1, 300),
        ...     ("B", 2, 400),
        ...     ("B", 3, 350)
        ... ], ["group", "id", "value"])
        >>> result = add_window_collect_top_n(df, "value", 2, "group", [("value", "desc")])
        >>> result.orderBy("group", F.desc("value")).show()
        +-----+---+-----+--------------+
        |group| id|value|  value_top_2 |
        +-----+---+-----+--------------+
        |    A|  2|  200|   [200, 150] |
        |    A|  3|  150|   [200, 150] |
        |    A|  4|  120|   [200, 150] |
        |    A|  1|  100|   [200, 150] |
        |    B|  2|  400|   [400, 350] |
        |    B|  3|  350|   [400, 350] |
        |    B|  1|  300|   [400, 350] |
        +-----+---+-----+--------------+
    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols
    if isinstance(order_cols, str):
        order_cols = [order_cols]

    # Create the ordering expressions
    order_exprs = []
    for col in order_cols:
        if isinstance(col, tuple):
            order_exprs.append(
                F.col(col[0]).cast("string").asc()
                if col[1].lower() == "asc"
                else F.col(col[0]).cast("string").desc()
            )
        else:
            order_exprs.append(F.col(col).cast("string").asc())

    # Create the window specification
    window = Window.partitionBy(*partition_cols).orderBy(*order_exprs)

    # Apply the window frame specification if provided
    if window_spec:
        frame_parts = window_spec.split()
        if (
            len(frame_parts) == 4
            and frame_parts[1] == "preceding"
            and frame_parts[3] == "following"
        ):
            preceding = (
                int(frame_parts[0])
                if frame_parts[0] != "unbounded"
                else Window.unboundedPreceding
            )
            following = (
                int(frame_parts[2])
                if frame_parts[2] != "unbounded"
                else Window.unboundedFollowing
            )
            window = window.rowsBetween(preceding, following)
    else:
        # If no window_spec provided, use the entire partition
        window = window.rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{collect_col}_top_{n}"

    # Create a temporary rank column
    df_with_rank = df.withColumn("temp_rank", F.rank().over(window))

    # Collect the top N values
    df_with_top_n = df_with_rank.withColumn(
        new_col_name,
        F.collect_list(F.when(F.col("temp_rank") <= n, F.col(collect_col))).over(
            window
        ),
    )

    # Remove the temporary rank column
    return df_with_top_n.drop("temp_rank")


# Add the function to DataFrame class for use with transform method
DataFrame.add_window_collect_top_n = add_window_collect_top_n


def add_window_collect_list(
    df: DataFrame,
    collect_col: str,
    partition_cols: Union[str, List[str]],
    order_cols: Optional[Union[str, List[str], List[Tuple[str, str]]]] = None,
    window_spec: Optional[str] = None,
    new_col_name: Optional[str] = None,
    max_list_size: Optional[int] = None,
) -> DataFrame:
    """
    Add a new column with a list of values collected within a window.

    This function collects values of a specified column into a list within
    the defined window, preserving duplicates and order.

    Args:
        df (DataFrame): Input DataFrame.
        collect_col (str): The column to collect values from.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Optional[Union[str, List[str], List[Tuple[str, str]]]]):
            Column(s) to order by. Can be None if no ordering is required.
        window_spec (Optional[str]): A string specifying the window frame, e.g.,
            "2 preceding and 2 following". If None, the entire partition is used.
        new_col_name (Optional[str]): Name of the new column containing the collected list.
            If not provided, defaults to "{collect_col}_list".
        max_list_size (Optional[int]): Maximum number of elements to include in the list.
            If None, all elements are included.

    Returns:
        DataFrame: DataFrame with an additional column containing the collected list of values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, "X"),
        ...     ("A", 2, "Y"),
        ...     ("A", 3, "X"),
        ...     ("A", 4, "Z"),
        ...     ("B", 1, "P"),
        ...     ("B", 2, "Q"),
        ...     ("B", 3, "P")
        ... ], ["group", "id", "value"])
        >>> result = add_window_collect_list(df, "value", "group", "id", "1 preceding and 1 following")
        >>> result.orderBy("group", "id").show(truncate=False)
        +-----+---+-----+-------------+
        |group|id |value|value_list   |
        +-----+---+-----+-------------+
        |A    |1  |X    |[X, Y]       |
        |A    |2  |Y    |[X, Y, X]    |
        |A    |3  |X    |[Y, X, Z]    |
        |A    |4  |Z    |[X, Z]       |
        |B    |1  |P    |[P, Q]       |
        |B    |2  |Q    |[P, Q, P]    |
        |B    |3  |P    |[Q, P]       |
        +-----+---+-----+-------------+
    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols if provided
    if order_cols:
        if isinstance(order_cols, str):
            order_cols = [order_cols]

        order_exprs = []
        for col in order_cols:
            if isinstance(col, tuple):
                order_exprs.append(
                    F.col(col[0]).cast("string").asc()
                    if col[1].lower() == "asc"
                    else F.col(col[0]).cast("string").desc()
                )
            else:
                order_exprs.append(F.col(col).cast("string").asc())

        # Create the window specification
        window = Window.partitionBy(*partition_cols).orderBy(*order_exprs)
    else:
        # If no order_cols provided, create a partitioned window without ordering
        window = Window.partitionBy(*partition_cols)

    # Apply the window frame specification if provided
    if window_spec:
        frame_parts = window_spec.split()
        if (
            len(frame_parts) == 4
            and frame_parts[1] == "preceding"
            and frame_parts[3] == "following"
        ):
            preceding = (
                int(frame_parts[0])
                if frame_parts[0] != "unbounded"
                else Window.unboundedPreceding
            )
            following = (
                int(frame_parts[2])
                if frame_parts[2] != "unbounded"
                else Window.unboundedFollowing
            )
            window = window.rowsBetween(preceding, following)
    else:
        # If no window_spec provided, use the entire partition
        window = window.rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{collect_col}_list"

    # Collect the list of values
    if max_list_size is not None:
        # If max_list_size is specified, use slice to limit the number of elements
        df = df.withColumn(
            new_col_name,
            F.expr(f"slice(collect_list({collect_col}), 1, {max_list_size})").over(
                window
            ),
        )
    else:
        # If max_list_size is not specified, use collect_list to get all values
        df = df.withColumn(new_col_name, F.collect_list(collect_col).over(window))

    return df


# Add the function to DataFrame class for use with transform method
DataFrame.add_window_collect_list = add_window_collect_list


def add_window_collect_set(
    df: DataFrame,
    collect_col: str,
    partition_cols: Union[str, List[str]],
    order_cols: Optional[Union[str, List[str], List[Tuple[str, str]]]] = None,
    window_spec: Optional[str] = None,
    new_col_name: Optional[str] = None,
    max_set_size: Optional[int] = None,
) -> DataFrame:
    """
    Add a new column with a set of unique values collected within a window.

    This function collects unique values of a specified column into a set within
    the defined window. The result is returned as an array column.

    Args:
        df (DataFrame): Input DataFrame.
        collect_col (str): The column to collect unique values from.
        partition_cols (Union[str, List[str]]): Column(s) to partition by.
        order_cols (Optional[Union[str, List[str], List[Tuple[str, str]]]]):
            Column(s) to order by. Can be None if no ordering is required.
        window_spec (Optional[str]): A string specifying the window frame, e.g.,
            "2 preceding and 2 following". If None, the entire partition is used.
        new_col_name (Optional[str]): Name of the new column containing the collected set.
            If not provided, defaults to "{collect_col}_set".
        max_set_size (Optional[int]): Maximum number of elements to include in the set.
            If None, all unique elements are included.

    Returns:
        DataFrame: DataFrame with an additional column containing the collected set of unique values.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("A", 1, "X"),
        ...     ("A", 2, "Y"),
        ...     ("A", 3, "X"),
        ...     ("A", 4, "Z"),
        ...     ("B", 1, "P"),
        ...     ("B", 2, "Q"),
        ...     ("B", 3, "P")
        ... ], ["group", "id", "value"])
        >>> result = add_window_collect_set(df, "value", "group", "id", "1 preceding and 1 following")
        >>> result.orderBy("group", "id").show(truncate=False)
        +-----+---+-----+-----------+
        |group|id |value|value_set  |
        +-----+---+-----+-----------+
        |A    |1  |X    |[X, Y]     |
        |A    |2  |Y    |[X, Y, Z]  |
        |A    |3  |X    |[X, Y, Z]  |
        |A    |4  |Z    |[X, Z]     |
        |B    |1  |P    |[P, Q]     |
        |B    |2  |Q    |[P, Q]     |
        |B    |3  |P    |[P, Q]     |
        +-----+---+-----+-----------+
    """
    # Ensure partition_cols is a list
    if isinstance(partition_cols, str):
        partition_cols = [partition_cols]

    # Process order_cols if provided
    if order_cols:
        if isinstance(order_cols, str):
            order_cols = [order_cols]

        order_exprs = []
        for col in order_cols:
            if isinstance(col, tuple):
                order_exprs.append(
                    F.col(col[0]).cast("string").asc()
                    if col[1].lower() == "asc"
                    else F.col(col[0]).cast("string").desc()
                )
            else:
                order_exprs.append(F.col(col).cast("string").asc())

        # Create the window specification
        window = Window.partitionBy(*partition_cols).orderBy(*order_exprs)
    else:
        # If no order_cols provided, create a partitioned window without ordering
        window = Window.partitionBy(*partition_cols)

    # Apply the window frame specification if provided
    if window_spec:
        frame_parts = window_spec.split()
        if (
            len(frame_parts) == 4
            and frame_parts[1] == "preceding"
            and frame_parts[3] == "following"
        ):
            preceding = (
                int(frame_parts[0])
                if frame_parts[0] != "unbounded"
                else Window.unboundedPreceding
            )
            following = (
                int(frame_parts[2])
                if frame_parts[2] != "unbounded"
                else Window.unboundedFollowing
            )
            window = window.rowsBetween(preceding, following)
    else:
        # If no window_spec provided, use the entire partition
        window = window.rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )

    # Determine the new column name
    if new_col_name is None:
        new_col_name = f"{collect_col}_set"

    # Collect the set of unique values
    if max_set_size is not None:
        # If max_set_size is specified, use array_distinct and slice to limit the number of elements
        df = df.withColumn(
            new_col_name,
            F.expr(
                f"slice(array_distinct(collect_list({collect_col})), 1, {max_set_size})"
            ).over(window),
        )
    else:
        # If max_set_size is not specified, use collect_set to get all unique values
        df = df.withColumn(new_col_name, F.collect_set(collect_col).over(window))

    return df


# Add the function to DataFrame class for use with transform method
DataFrame.add_window_collect_set = add_window_collect_set
