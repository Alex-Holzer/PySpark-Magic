from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    count,
    col,
    isnan,
    min,
    max,
    length,
    regexp_extract,
    countDistinct,
)
from typing import Union, List, Dict, Tuple
from pyspark.sql.types import DataType, StructField, StringType


def validate_unique_combination(df: DataFrame, columns: Union[str, List[str]]) -> None:
    """
    Check for uniqueness in the specified column(s) of a PySpark DataFrame.

    This function validates whether the combination of values in the specified column(s)
    is unique across all rows in the DataFrame. If non-unique combinations are found,
    it raises a ValueError with details about the duplicates.

    Args:
        df (DataFrame): The input PySpark DataFrame to check.
        columns (Union[str, List[str]]): A single column name or a list of column names to check for uniqueness.

    Raises:
        ValueError: If the input types are incorrect or if non-unique combinations are found.

    Example:
        >>> df = spark.createDataFrame([
        ...     (1, "A", "X"),
        ...     (2, "B", "Y"),
        ...     (3, "A", "X"),  # Duplicate combination
        ...     (4, "C", "Z")
        ... ], ["id", "col1", "col2"])
        >>>
        >>> # Check for uniqueness in a single column
        >>> validate_unique_combination(df, "col1")
        >>>
        >>> # Check for uniqueness in multiple columns
        >>> validate_unique_combination(df, ["col1", "col2"])
        Traceback (most recent call last):
            ...
        ValueError: Non-unique combinations found in columns ['col1', 'col2']. Duplicates: [('A', 'X')]
    """
    if not isinstance(df, DataFrame):
        raise ValueError("Input 'df' must be a PySpark DataFrame.")

    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list) or not all(
        isinstance(col, str) for col in columns
    ):
        raise ValueError("'columns' must be a string or a list of strings.")

    # Check if all specified columns exist in the DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    # Count occurrences of each combination
    grouped = df.groupBy(columns).agg(count("*").alias("count"))
    duplicates = grouped.filter(col("count") > 1)

    # If duplicates are found, raise an error with details
    if duplicates.count() > 0:
        dup_combinations = duplicates.select(columns).collect()
        dup_list = [tuple(row) for row in dup_combinations]
        raise ValueError(
            f"Non-unique combinations found in columns {columns}. Duplicates: {dup_list}"
        )


def validate_column_existence(df: DataFrame, columns: List[str]) -> None:
    """
    Check if all specified columns exist in the DataFrame.

    Args:
        df (DataFrame): The input PySpark DataFrame to check.
        columns (List[str]): A list of column names to verify.

    Raises:
        ValueError: If any of the specified columns are missing from the DataFrame.

    Example:
        >>> df = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "value"])
        >>> validate_column_existence(df, ["id", "value", "nonexistent"])
        Traceback (most recent call last):
            ...
        ValueError: Columns ['nonexistent'] not found in the DataFrame.
    """
    if not isinstance(df, DataFrame):
        raise ValueError("Input 'df' must be a PySpark DataFrame.")
    if not isinstance(columns, list) or not all(
        isinstance(col, str) for col in columns
    ):
        raise ValueError("'columns' must be a list of strings.")

    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")


def validate_column_types(df: DataFrame, column_type_map: Dict[str, DataType]) -> None:
    """
    Verify that specified columns have the expected data types.

    Args:
        df (DataFrame): The input PySpark DataFrame to check.
        column_type_map (Dict[str, DataType]): A dictionary mapping column names to their expected DataType.

    Raises:
        ValueError: If any column has an unexpected data type.

    Example:
        >>> from pyspark.sql.types import IntegerType, StringType
        >>> df = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "value"])
        >>> validate_column_types(df, {"id": IntegerType(), "value": StringType()})
        >>> validate_column_types(df, {"id": StringType()})
        Traceback (most recent call last):
            ...
        ValueError: Column 'id' has unexpected type. Expected: StringType, Found: IntegerType
    """
    if not isinstance(df, DataFrame):
        raise ValueError("Input 'df' must be a PySpark DataFrame.")
    if not isinstance(column_type_map, dict):
        raise ValueError("'column_type_map' must be a dictionary.")

    for col_name, expected_type in column_type_map.items():
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in the DataFrame.")

        actual_type = df.schema[col_name].dataType
        if not isinstance(actual_type, expected_type.__class__):
            raise ValueError(
                f"Column '{col_name}' has unexpected type. Expected: {expected_type}, Found: {actual_type}"
            )


def validate_value_range(
    df: DataFrame,
    column: str,
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> None:
    """
    Check if numeric values in a column fall within a specified range.

    Args:
        df (DataFrame): The input PySpark DataFrame to check.
        column (str): The name of the column to validate.
        min_value (Union[int, float]): The minimum allowed value (inclusive).
        max_value (Union[int, float]): The maximum allowed value (inclusive).

    Raises:
        ValueError: If any values in the column are outside the specified range.

    Example:
        >>> df = spark.createDataFrame([(1,), (5,), (10,)], ["value"])
        >>> validate_value_range(df, "value", 0, 9)
        Traceback (most recent call last):
            ...
        ValueError: Column 'value' contains values outside the range [0, 9]. Min: 1, Max: 10
    """
    if not isinstance(df, DataFrame):
        raise ValueError("Input 'df' must be a PySpark DataFrame.")
    if not isinstance(column, str):
        raise ValueError("'column' must be a string.")
    if not isinstance(min_value, (int, float)) or not isinstance(
        max_value, (int, float)
    ):
        raise ValueError("'min_value' and 'max_value' must be numeric.")
    if min_value > max_value:
        raise ValueError("'min_value' must be less than or equal to 'max_value'.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    stats = df.agg(
        min(col(column)).alias("min"), max(col(column)).alias("max")
    ).collect()[0]
    col_min, col_max = stats["min"], stats["max"]

    if col_min < min_value or col_max > max_value:
        raise ValueError(
            f"Column '{column}' contains values outside the range [{min_value}, {max_value}]. Min: {col_min}, Max: {col_max}"
        )


def validate_string_length(
    df: DataFrame, column: str, min_length: int, max_length: int
) -> None:
    """
    Verify that string values in a column have lengths within a specified range.

    Args:
        df (DataFrame): The input PySpark DataFrame to check.
        column (str): The name of the column to validate.
        min_length (int): The minimum allowed string length (inclusive).
        max_length (int): The maximum allowed string length (inclusive).

    Raises:
        ValueError: If any string values in the column have lengths outside the specified range.

    Example:
        >>> df = spark.createDataFrame([("A",), ("ABC",), ("ABCDE",)], ["value"])
        >>> validate_string_length(df, "value", 1, 3)
        Traceback (most recent call last):
            ...
        ValueError: Column 'value' contains strings with lengths outside the range [1, 3]. Min length: 1, Max length: 5
    """
    if not isinstance(df, DataFrame):
        raise ValueError("Input 'df' must be a PySpark DataFrame.")
    if not isinstance(column, str):
        raise ValueError("'column' must be a string.")
    if not isinstance(min_length, int) or not isinstance(max_length, int):
        raise ValueError("'min_length' and 'max_length' must be integers.")
    if min_length > max_length:
        raise ValueError("'min_length' must be less than or equal to 'max_length'.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    length_stats = df.select(
        min(length(col(column))).alias("min_length"),
        max(length(col(column))).alias("max_length"),
    ).collect()[0]
    min_str_length, max_str_length = (
        length_stats["min_length"],
        length_stats["max_length"],
    )

    if min_str_length < min_length or max_str_length > max_length:
        raise ValueError(
            f"Column '{column}' contains strings with lengths outside the range [{min_length}, {max_length}]. "
            f"Min length: {min_str_length}, Max length: {max_str_length}"
        )


def validate_foreign_key(
    df: DataFrame, column: str, reference_df: DataFrame, reference_column: str
) -> None:
    """
    Verify that values in a column exist in a reference DataFrame's column (foreign key constraint).

    Args:
        df (DataFrame): The input PySpark DataFrame to check.
        column (str): The name of the column to validate.
        reference_df (DataFrame): The reference DataFrame containing the valid values.
        reference_column (str): The name of the column in the reference DataFrame to check against.

    Raises:
        ValueError: If any values in the column do not exist in the reference column.

    Example:
        >>> df = spark.createDataFrame([(1,), (2,), (3,)], ["id"])
        >>> ref_df = spark.createDataFrame([(1,), (2,)], ["valid_id"])
        >>> validate_foreign_key(df, "id", ref_df, "valid_id")
        Traceback (most recent call last):
            ...
        ValueError: Column 'id' contains 1 values that do not exist in the reference column 'valid_id'.
    """
    if not isinstance(df, DataFrame) or not isinstance(reference_df, DataFrame):
        raise ValueError("Input 'df' and 'reference_df' must be PySpark DataFrames.")
    if not isinstance(column, str) or not isinstance(reference_column, str):
        raise ValueError("'column' and 'reference_column' must be strings.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the input DataFrame.")
    if reference_column not in reference_df.columns:
        raise ValueError(
            f"Column '{reference_column}' not found in the reference DataFrame."
        )

    # Perform a left anti join to find values that don't exist in the reference DataFrame
    invalid_values = df.join(
        reference_df, df[column] == reference_df[reference_column], "left_anti"
    )
    invalid_count = invalid_values.count()

    if invalid_count > 0:
        raise ValueError(
            f"Column '{column}' contains {invalid_count} values that do not exist in the reference column '{reference_column}'."
        )


def validate_contract_number_format(df: DataFrame, column: str) -> DataFrame:
    """
    Validates if all contract numbers in the specified column match the expected format.
    The expected format is "XXCC-NNNNNNN", where:
    - XX are two digits
    - CC are two characters
    - NNNNNNN is a seven-digit number
    - The dash (-) separates the first four characters from the last seven

    Args:
        df (DataFrame): Input PySpark DataFrame.
        column (str): Name of the column containing the contract numbers.

    Returns:
        DataFrame: A new DataFrame with an additional boolean column 'is_valid_format'
                   indicating whether each contract number matches the expected format.

    Example:
        >>> df = spark.createDataFrame([
        ...     ("01FV-0685210",),
        ...     ("35FV-9532111",),
        ...     ("99LV-0012340",),
        ...     ("00AV-4912338",),
        ...     ("01FV0685210",),  # Invalid format (missing dash)
        ...     ("0685210-01FV",), # Invalid format (wrong order)
        ...     ("AB12-3456789",)  # Invalid format (letters at start)
        ... ], ["contract_number"])
        >>> result = validate_contract_number_format(df, "contract_number")
        >>> result.show()
        +---------------+----------------+
        |contract_number|is_valid_format |
        +---------------+----------------+
        |  01FV-0685210 |           true |
        |  35FV-9532111 |           true |
        |  99LV-0012340 |           true |
        |  00AV-4912338 |           true |
        |  01FV0685210  |          false |
        |  0685210-01FV |          false |
        |  AB12-3456789 |          false |
        +---------------+----------------+
    """
    # Define the regex pattern for the expected format
    pattern = r"^(\d{2})([A-Z]{2})-(\d{7})$"

    # Add a new column 'is_valid_format' to indicate if the format is valid
    df_with_validation = df.withColumn(
        "is_valid_format",
        (length(col(column)) == 12)
        & (regexp_extract(col(column), pattern, 0) == col(column)),
    )

    return df_with_validation


def validate_optional_string_column(df: DataFrame, column: str, threshold: int) -> bool:
    """
    Validate if a column in a PySpark DataFrame has enough distinct values to be considered an optional string column.

    This function counts the distinct values in the specified column and compares it to the given threshold.
    If the number of distinct values is larger than the threshold, the column is considered an optional string column.

    Args:
        df (DataFrame): The input PySpark DataFrame.
        column (str): The name of the column to validate.
        threshold (int): The minimum number of distinct values required to consider the column as an optional string column.

    Returns:
        bool: True if the column is an optional string column (distinct values > threshold), False otherwise.

    Raises:
        ValueError: If the input types are incorrect or if the specified column doesn't exist in the DataFrame.

    Example:
        >>> df = spark.createDataFrame([("A",), ("B",), ("C",), ("A",), ("D",)], ["value"])
        >>> is_optional = validate_optional_string_column(df, "value", 3)
        >>> print(is_optional)
        True
        >>> is_optional = validate_optional_string_column(df, "value", 5)
        >>> print(is_optional)
        False
    """
    # Input validation
    if not isinstance(df, DataFrame):
        raise ValueError("Input 'df' must be a PySpark DataFrame.")
    if not isinstance(column, str):
        raise ValueError("'column' must be a string.")
    if not isinstance(threshold, int) or threshold < 0:
        raise ValueError("'threshold' must be a non-negative integer.")

    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # Check if the column is of StringType
    if not isinstance(df.schema[column].dataType, StringType):
        raise ValueError(f"Column '{column}' must be of StringType.")

    # Count distinct values in the column
    distinct_count = df.select(
        countDistinct(col(column)).alias("distinct_count")
    ).collect()[0]["distinct_count"]

    # Compare distinct count to threshold
    return distinct_count > threshold


def validate_missing_value(df: DataFrame, column: str) -> Tuple[float, int]:
    """
    Calculate the ratio (percentage) and count of missing values in a specified column of a PySpark DataFrame.

    Missing values are considered to be None, Null, NaN, or empty strings.

    Args:
        df (DataFrame): The input PySpark DataFrame.
        column (str): The name of the column to check for missing values.

    Returns:
        Tuple[float, int]: A tuple containing:
            - The percentage of missing values (float between 0 and 100).
            - The total number of missing values (int).

    Raises:
        ValueError: If the input types are incorrect or if the specified column doesn't exist in the DataFrame.

    Example:
        >>> df = spark.createDataFrame([("",), ("A",), (None,), ("B",), (None,)], ["value"])
        >>> missing_ratio, missing_count = validate_missing_value(df, "value")
        >>> print(f"Missing value ratio: {missing_ratio:.2f}%")
        Missing value ratio: 60.00%
        >>> print(f"Missing value count: {missing_count}")
        Missing value count: 3
    """
    if not isinstance(df, DataFrame):
        raise ValueError("Input 'df' must be a PySpark DataFrame.")
    if not isinstance(column, str):
        raise ValueError("'column' must be a string.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    total_count = df.count()
    missing_count = df.filter(
        (col(column).isNull()) | (col(column) == "") | (isnan(col(column)))
    ).count()

    missing_ratio = (missing_count / total_count) * 100 if total_count > 0 else 0.0

    return missing_ratio, missing_count
