from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, trim, lower, lit
from pyspark.sql.column import Column
from typing import Optional, Union, List, Dict, Any


def fill_missing_values(
    df: DataFrame,
    fill_value_map: Dict[str, Union[str, int, float]],
) -> DataFrame:
    """
    Fill missing values in specified columns with defined values.

    This function replaces null values in the specified columns with the provided
    fill values. It supports string, integer, and float fill values.

    Args:
        df (DataFrame): Input PySpark DataFrame.
        fill_value_map (Dict[str, Union[str, int, float]]): A dictionary mapping
            column names to their respective fill values.

    Returns:
        DataFrame: DataFrame with missing values filled according to the fill_value_map.

    Raises:
        ValueError: If the input DataFrame is empty or if specified columns don't exist.
        TypeError: If a fill value's type is not string, int, or float.

    Example:
        >>> data = [("John", None, None), ("Jane", 25, None), ("Bob", None, 70.5)]
        >>> df = spark.createDataFrame(data, ["name", "age", "weight"])
        >>> fill_map = {"age": 30, "weight": 65.0}
        >>> result_df = fill_missing_values(df, fill_map)
        >>> result_df.show()
        +----+---+------+
        |name|age|weight|
        +----+---+------+
        |John| 30|  65.0|
        |Jane| 25|  65.0|
        | Bob| 30|  70.5|
        +----+---+------+
    """
    if df.rdd.isEmpty():
        raise ValueError("Input DataFrame is empty.")

    non_existent_cols = set(fill_value_map.keys()) - set(df.columns)
    if non_existent_cols:
        raise ValueError(f"Columns not found in DataFrame: {non_existent_cols}")

    for col_name, fill_value in fill_value_map.items():
        if not isinstance(fill_value, (str, int, float)):
            raise TypeError(
                f"Fill value for column '{col_name}' must be a string, int, or float."
            )

        df = df.withColumn(
            col_name,
            when(col(col_name).isNull(), lit(fill_value)).otherwise(col(col_name)),
        )

    return df


def replace_missing_values(
    df: DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    missing_patterns: Optional[Dict[str, List[str]]] = None,
) -> DataFrame:
    """
    Replace values indicating missing data with None in specified columns of a PySpark DataFrame.

    This function identifies and replaces various forms of missing values, including:
    - Cells containing only special characters
    - Empty strings
    - Variations of the word "null" (including when surrounded by other characters)
    - Custom patterns provided by the user

    Args:
        df (DataFrame): Input PySpark DataFrame.
        columns (Optional[Union[str, List[str]]]): Column(s) to process. If None, all columns are processed.
        missing_patterns (Optional[Dict[str, List[str]]]): Custom patterns to identify missing values.
            Keys are column names, values are lists of regex patterns.

    Returns:
        DataFrame: DataFrame with missing values replaced by None.

    Raises:
        ValueError: If the input DataFrame is empty or if specified columns don't exist.

    Example:
        >>> data = [("John", " "), ("Jane", "(null)"), ("Bob", "@#$"), ("Alice", "N/A")]
        >>> df = spark.createDataFrame(data, ["name", "value"])
        >>> custom_patterns = {"value": [r"N/A"]}
        >>> result_df = replace_missing_values(df, "value", custom_patterns)
        >>> result_df.show()
        +-----+-----+
        | name|value|
        +-----+-----+
        | John| NULL|
        | Jane| NULL|
        |  Bob| NULL|
        |Alice| NULL|
        +-----+-----+
    """
    if df.rdd.isEmpty():
        raise ValueError("Input DataFrame is empty.")

    if columns is None:
        columns = df.columns
    elif isinstance(columns, str):
        columns = [columns]

    non_existent_cols = set(columns) - set(df.columns)
    if non_existent_cols:
        raise ValueError(f"Columns not found in DataFrame: {non_existent_cols}")

    missing_patterns = missing_patterns or {}

    def get_missing_value_expression(column: str) -> Column:
        base_patterns = [
            r"^[\s\W]*$",  # Only special characters or whitespace
            r"^\s*$",  # Empty string
            r".*null.*",  # "null" surrounded by any characters
            r".*n/a.*",  # n/a surrounded by any characters
        ]
        custom_patterns = missing_patterns.get(column, [])
        all_patterns = base_patterns + custom_patterns

        return when(
            trim(lower(col(column))).rlike(
                "|".join(f"({pattern})" for pattern in all_patterns)
            )
            | col(column).isNull(),
            lit(None),
        ).otherwise(col(column))
