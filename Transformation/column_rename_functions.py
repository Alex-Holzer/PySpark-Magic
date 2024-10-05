"""

column_rename_functions.py
===========================

This module provides a collection of utility functions for renaming, modifying,
and validating column names in PySpark DataFrames. These functions are designed
to be efficient, reusable, and compatible with large-scale data processing in
Databricks environments.

Author: Alex Holzer
Date: 22.09.2024
Version: 1.0
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lower, regexp_replace

# Your SparkSession creation
spark: SparkSession = SparkSession.builder.getOrCreate()


def rename_columns_using_mapping(
    dataframe: DataFrame, columns_map: Dict[str, str]
) -> DataFrame:
    """
    Rename columns of a DataFrame based on a provided dictionary mapping.

    Args:
        dataframe (DataFrame): Input DataFrame.
        columns_map (Dict[str, str]): Dictionary mapping old column names to new column names.

    Returns:
        DataFrame: DataFrame with renamed columns.
    """
    new_columns = [columns_map.get(c, c) for c in dataframe.columns]
    return dataframe.toDF(*new_columns)


def rename_columns_with_prefix(dataframe: DataFrame, prefix: str) -> DataFrame:
    """
    Add a prefix to all column names in the DataFrame.

    Args:
        dataframe (DataFrame): Input DataFrame.
        prefix (str): Prefix to add to column names.

    Returns:
        DataFrame: DataFrame with prefixed column names.
    """
    return dataframe.toDF(*[f"{prefix}{c}" for c in dataframe.columns])


def rename_columns_with_suffix(dataframe: DataFrame, suffix: str) -> DataFrame:
    """
    Add a suffix to all column names in the DataFrame.

    Args:
        dataframe (DataFrame): Input DataFrame.
        suffix (str): Suffix to add to column names.

    Returns:
        DataFrame: DataFrame with suffixed column names.
    """
    return dataframe.toDF(*[f"{c}{suffix}" for c in dataframe.columns])


def remove_prefix_from_columns(dataframe: DataFrame, prefix: str) -> DataFrame:
    """Remove specified prefix from column names in a PySpark DataFrame.

    Args:
        dataframe: Input PySpark DataFrame.
        prefix: Prefix to remove from column names.

    Returns:
        DataFrame with prefix removed from applicable column names.

    Example:
        >>> df = spark.createDataFrame([(1, 2)], ["prefix_col1", "prefix_col2"])
        >>> remove_prefix_from_columns(dataframe, "prefix_").columns
        ['col1', 'col2']
    """
    return dataframe.select(
        [
            col(c).alias(c[len(prefix) :] if c.startswith(prefix) else c)
            for c in dataframe.columns
        ]
    )


def remove_suffix_from_columns(dataframe: DataFrame, suffix: str) -> DataFrame:
    """Remove specified suffix from column names in a PySpark DataFrame.

    Args:
        dataframe: Input PySpark DataFrame.
        suffix: Suffix to remove from column names.

    Returns:
        DataFrame with suffix removed from applicable column names.

    Example:
        >>> df = spark.createDataFrame([(1, 2)], ["col1_suffix", "col2_suffix"])
        >>> remove_suffix_from_columns(dataframe, "_suffix").columns
        ['col1', 'col2']
    """
    return dataframe.select(
        [
            col(c).alias(c[: -len(suffix)] if c.endswith(suffix) else c)
            for c in dataframe.columns
        ]
    )


def rename_columns_replace_spaces_with_underscores(dataframe: DataFrame) -> DataFrame:
    """
    Replace spaces with underscores in all column names.

    Args:
        dataframe (DataFrame): Input DataFrame.

    Returns:
        DataFrame: DataFrame with spaces in column names replaced by underscores.
    """
    return dataframe.toDF(*[c.replace(" ", "_") for c in dataframe.columns])


def rename_columns_to_lowercase(dataframe: DataFrame) -> DataFrame:
    """
    Convert all column names to lowercase.

    Args:
        dataframe (DataFrame): Input DataFrame.

    Returns:
        DataFrame: DataFrame with lowercase column names.
    """
    return dataframe.toDF(*[c.lower() for c in dataframe.columns])


def rename_columns_to_uppercase(dataframe: DataFrame) -> DataFrame:
    """
    Convert all column names to uppercase.

    Args:
        dataframe (DataFrame): Input DataFrame.

    Returns:
        DataFrame: DataFrame with uppercase column names.
    """
    return dataframe.toDF(*[c.upper() for c in dataframe.columns])


def rename_columns_to_snake_case(dataframe: DataFrame) -> DataFrame:
    """
    Rename all columns in a PySpark DataFrame to snake_case.
    This function converts all column names to lowercase and replaces spaces with underscores.

    Args:
        dataframe (DataFrame): The input PySpark DataFrame.

    Returns:
        DataFrame: A new DataFrame with all column names converted to snake_case.

    Raises:
        ValueError: If the input is not a PySpark DataFrame.

    Example:
        >>> df = spark.createDataFrame([(1, 2, 3)], ["Column One", "Column Two", "Column Three"])
        >>> result_df = rename_columns_to_snake_case(dataframe)
        >>> result_df.show()
        +----------+----------+-------------+
        |column_one|column_two|column_three |
        +----------+----------+-------------+
        |         1|         2|            3|
        +----------+----------+-------------+
    """
    return dataframe.select(
        [lower(regexp_replace(col(c), " ", "_")).alias(c) for c in dataframe.columns]
    )


def remove_substring_from_column_names(
    dataframe: DataFrame,
    columns: Optional[List[str]] = None,
    substring: Union[str, int] = "",
) -> DataFrame:
    """
    Removes a specific substring (character or number) from the column names in the given DataFrame.

    Parameters
    ----------
    dataframe : DataFrame
        The input PySpark DataFrame.
    columns : list, optional
        List of columns from which the substring should be removed.
    substring : str or int
        The substring (character or number) to remove from the column names.

    Returns
    -------
    DataFrame
        The DataFrame with updated column names.
    """

    substring = str(substring)
    columns_to_process = columns if columns is not None else dataframe.columns

    new_column_names = [
        col_name.replace(substring, "") if col_name in columns_to_process else col_name
        for col_name in dataframe.columns
    ]

    # Return a DataFrame with renamed columns
    return dataframe.toDF(*new_column_names)


def remove_digits_from_columns(df: DataFrame) -> DataFrame:
    """
    Removes digits from the column names of the given PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input PySpark DataFrame with columns potentially containing digits.

    Returns
    -------
    DataFrame
        A new DataFrame with digits removed from the column names.
    """
    # Use list comprehension to remove digits from the column names
    new_column_names = [re.sub(r"\d", "", col_name) for col_name in df.columns]

    # Apply the new column names to the DataFrame
    df_renamed = df.toDF(*new_column_names)

    return df_renamed


def remove_special_characters_from_columns(
    dataframe: DataFrame, columns: Optional[List[str]] = None
) -> DataFrame:
    """
    Removes all special characters from the column names in the given DataFrame.

    Parameters
    ----------
    dataframe : DataFrame
        The input PySpark DataFrame.
    columns : list, optional
        List of columns from which special characters should be removed.
        If None, all columns will be processed.

    Returns
    -------
    DataFrame
        A new DataFrame with special characters removed from the specified column names.
    """
    pattern = r"[^a-zA-Z0-9_]"

    columns_to_process = columns if columns is not None else dataframe.columns

    new_column_names = [
        re.sub(pattern, "", col_name) if col_name in columns_to_process else col_name
        for col_name in dataframe.columns
    ]

    # Return a new DataFrame with updated column names
    return dataframe.toDF(*new_column_names)


def assert_unique_column_names(dataframe: DataFrame) -> None:
    """Assert that all column names in the DataFrame are unique.

    Raises ValueError if duplicate column names are found.

    Args:
        dataframe: PySpark DataFrame to check for unique column names.

    Raises:
        ValueError: If duplicate column names are found.

    Example:
        >>> dataframe = spark.createDataFrame([(1, 2)], ["col1", "col1"])
        >>> assert_unique_column_names(dataframe)
        Traceback (most recent call last):
            ...
        ValueError: Duplicate column names found: ['col1']
    """
    duplicates = [col for col, count in Counter(dataframe.columns).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate column names found: {duplicates}")


def assert_valid_column_names(
    dataframe: DataFrame, pattern: str = r"^[a-zA-Z][a-zA-Z0-9_]*$"
) -> None:
    """Assert that all column names match the specified pattern.

    Args:
        dataframe: PySpark DataFrame to check.
        pattern: Regex pattern for valid column names. Default allows
                 names starting with a letter, followed by letters,
                 numbers, or underscores.

    Raises:
        ValueError: If input is not a PySpark DataFrame.
        AssertionError: If invalid column names are found.

    Example:
        >>> df = spark.createDataFrame([(1, 2)], ["valid", "1invalid"])
        >>> assert_valid_column_names(dataframe)
        Traceback (most recent call last):
            ...
        AssertionError: Invalid column names found: '1invalid'
    """
    if not isinstance(dataframe, DataFrame):
        raise ValueError("Input must be a PySpark DataFrame")

    invalid_cols = [col for col in dataframe.columns if not re.match(pattern, col)]
    if invalid_cols:
        raise AssertionError(f"Invalid column names found: {invalid_cols}")
