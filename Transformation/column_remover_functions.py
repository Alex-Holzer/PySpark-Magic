"""
columns_remover_functions.py
===========================


This module contains a set of utility functions for removing specific types of columns from a PySpark DataFrame.
Each function is designed to handle a particular column type or content pattern, allowing for efficient data cleaning 
and preprocessing in large datasets. 
"""

from typing import List

from pyspark.sql import DataFrame, SparkSession

spark: SparkSession = SparkSession.builder.getOrCreate()

from pyspark.sql.functions import col, count, isnan, when
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructType,
    TimestampType,
)


def remove_empty_string_columns(df: DataFrame) -> DataFrame:
    """
    Removes columns that contain only empty strings.
    """
    non_empty_columns = [
        col_name
        for col_name in df.columns
        if df.filter(col(col_name) != "").count() > 0
    ]
    return df.select(*non_empty_columns)


def remove_boolean_columns(df: DataFrame) -> DataFrame:
    """
    Removes all BooleanType columns from the DataFrame.
    """
    non_boolean_columns = [
        field.name
        for field in df.schema.fields
        if not isinstance(field.dataType, BooleanType)
    ]
    return df.select(*non_boolean_columns)


def remove_complex_columns(df: DataFrame) -> DataFrame:
    """
    Removes all complex columns (ArrayType, MapType, StructType) from the given PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input PySpark DataFrame from which complex columns need to be removed.

    Returns
    -------
    DataFrame
        A new DataFrame without complex columns.
    """
    complex_types = (ArrayType, MapType, StructType)

    non_complex_columns = [
        field.name
        for field in df.schema.fields
        if not isinstance(field.dataType, complex_types)
    ]

    return df.select(*non_complex_columns)


def remove_date_columns(df: DataFrame) -> DataFrame:
    """
    Removes all date-related columns (DateType, TimestampType) from the given PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input PySpark DataFrame from which date columns need to be removed.

    Returns
    -------
    DataFrame
        A new DataFrame without date-related columns.
    """
    # List of date-related types to remove
    date_types = (DateType, TimestampType)

    # Filter out columns that are of DateType or TimestampType
    non_date_columns = [
        field.name
        for field in df.schema.fields
        if not isinstance(field.dataType, date_types)
    ]

    # Select only the non-date columns
    return df.select(*non_date_columns)


def remove_numeric_columns(df: DataFrame) -> DataFrame:
    """
    Removes all numeric columns (IntegerType, FloatType, DoubleType, etc.) from the given PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input PySpark DataFrame from which numeric columns need to be removed.

    Returns
    -------
    DataFrame
        A new DataFrame without numeric columns.
    """
    # List of numeric types to remove
    numeric_types = (
        IntegerType,
        FloatType,
        DoubleType,
        LongType,
        ShortType,
        ByteType,
        DecimalType,
    )

    # Filter out columns that are of numeric types
    non_numeric_columns = [
        field.name
        for field in df.schema.fields
        if not isinstance(field.dataType, numeric_types)
    ]

    # Select only the non-numeric columns
    return df.select(*non_numeric_columns)


def remove_string_columns(df: DataFrame) -> DataFrame:
    """
    Removes all columns of StringType from the given PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input PySpark DataFrame from which string columns need to be removed.

    Returns
    -------
    DataFrame
        A new DataFrame without StringType columns.
    """
    # Filter out columns that are of StringType
    non_string_columns = [
        field.name
        for field in df.schema.fields
        if not isinstance(field.dataType, StringType)
    ]

    # Select only the non-string columns
    return df.select(*non_string_columns)


def remove_empty_columns(df: DataFrame) -> DataFrame:
    """
    Remove all columns from a PySpark DataFrame that contain only null or NaN values.

    Parameters
    ----------
    df : DataFrame
        Input PySpark DataFrame from which empty columns need to be removed.

    Returns
    -------
    DataFrame
        A new DataFrame without columns that are entirely null or NaN.
    """
    # Explicitly cast all columns to DoubleType to ensure type consistency
    df_casted = df.select([col(column).cast(DoubleType()) for column in df.columns])

    # Aggregate to count non-null and non-NaN values in each column
    agg_result = df_casted.agg(
        *[
            (
                count(when(~col(column).isNull() & ~isnan(col(column)), column)).alias(
                    column
                )
            )
            for column in df_casted.columns
        ]
    ).first()

    # Handle the case where first() might return None
    if agg_result is None:
        return df

    # Convert aggregation result to a dictionary
    agg_dict = agg_result.asDict()

    # Get only the columns that are not entirely null or NaN
    non_empty_columns = [
        column for column, non_null_count in agg_dict.items() if non_null_count > 0
    ]

    # Select only non-empty columns
    return df.select(*non_empty_columns)


def remove_columns_with_prefix(dataframe: DataFrame, prefix: str) -> DataFrame:
    """
    Drop all columns from a PySpark DataFrame that start with the specified prefix.

    Args:
        dataframe (DataFrame): The input PySpark DataFrame.
        prefix (str): The prefix to match against column names.

    Returns:
        DataFrame: A new DataFrame with the specified columns dropped.

    Raises:
        ValueError: If the input is not a PySpark DataFrame or if the prefix is not a string.

    Example:
        >>> df = spark.createDataFrame([(1, 2, 3, 4)], ["pre_col1", "pre_col2", "col3", "Col4"])
        >>> result_dataframe = remove_columns_with_prefix(dataframe, "pre_")
        >>> result_df.show()
        +----+----+
        |col3|Col4|
        +----+----+
        |   3|   4|
        +----+----+
    """
    columns_to_keep = [col for col in dataframe.columns if not col.startswith(prefix)]
    return dataframe.select(*columns_to_keep)


def remove_columns_with_suffix(dataframe: DataFrame, suffix: str) -> DataFrame:
    """
    Drop all columns from a PySpark DataFrame that end with the specified suffix.

    Args:
        dataframe (DataFrame): The input PySpark DataFrame.
        suffix (str): The suffix to match against column names.

    Returns:
        DataFrame: A new DataFrame with the specified columns dropped.

    Raises:
        ValueError: If the input is not a PySpark DataFrame or if the suffix is not a string.

    Example:
        >>> df = spark.createDataFrame([(1, 2, 3, 4)], ["col1_suf", "col2_suf", "col3", "Col4"])
        >>> result_df = remove_columns_with_suffix(dataframe, "_suf")
        >>> result_df.show()
        +----+----+
        |col3|Col4|
        +----+----+
        |   3|   4|
        +----+----+
    """
    columns_to_keep: List[str] = [
        col for col in dataframe.columns if not col.endswith(suffix)
    ]
    return dataframe.select(*columns_to_keep)
