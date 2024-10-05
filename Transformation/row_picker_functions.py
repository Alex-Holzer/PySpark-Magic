"""
row_picker_functions.py
===========================

This module provides a collection of reusable functions designed to filter rows in PySpark DataFrames
based on various conditions. The functions cover a wide range of filtering scenarios, such as filtering
based on numeric ranges, date ranges, string patterns, null checks, and more. These functions are 
intended to simplify the process of cleaning, transforming, and analyzing data in PySpark by providing 
concise, flexible, and reusable filters for common use cases.

Each function takes a PySpark DataFrame as input along with filtering criteria and returns a new DataFrame
with the filtered results. The functions are designed to be modular and can be combined for more complex
filtering logic.
"""

from datetime import date
from functools import reduce
from typing import List, Union

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()


def filter_rows_by_and_conditions(
    df: DataFrame, conditions: List[Union[Column, str]]
) -> DataFrame:
    """
    Filters a PySpark DataFrame by applying multiple conditions combined with AND logic.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame to be filtered.
    conditions : List[Union[Column, str]]
        A list of conditions, where each condition is either a PySpark Column or a string
        representing a condition (e.g., "age > 30").

    Returns
    -------
    DataFrame
        The filtered DataFrame after applying all conditions combined with AND.

    Examples
    --------
    # Example 1: Filter for active users in New York with age >= 30
    conditions = [
        col("status") == "active",
        col("age") >= 30,
        col("city") == "New York"
    ]
    df_filtered = filter_rows_by_and_conditions(df, conditions)

    # Example 2: Filter for users whose city is either New York or Los Angeles
    conditions_isin = [
        col("city").isin(["New York", "Los Angeles"]),
        col("status") == "active"
    ]
    df_filtered_isin = filter_rows_by_and_conditions(df, conditions_isin)

    # Example 3: Filter for users whose city is NOT New York or Los Angeles
    conditions_is_not_in = [
        ~col("city").isin(["New York", "Los Angeles"]),
        col("status") == "active"
    ]
    df_filtered_is_not_in = filter_rows_by_and_conditions(df, conditions_is_not_in)

    # Example 4: If no conditions are provided, the original DataFrame is returned
    df_filtered_empty = filter_rows_by_and_conditions(df, [])
    """

    if not conditions:
        # If no conditions are provided, return the original DataFrame unfiltered
        return df

    # Convert any string conditions to PySpark column expressions
    condition_exprs = [
        col(condition) if isinstance(condition, str) else condition
        for condition in conditions
    ]

    # Combine all conditions with AND using reduce
    combined_condition = reduce(lambda cond1, cond2: cond1 & cond2, condition_exprs)

    # Apply the combined condition to filter the DataFrame
    return df.filter(combined_condition)


def filter_rows_by_or_conditions(
    df: DataFrame, conditions: List[Union[Column, str]]
) -> DataFrame:
    """
    Filters a PySpark DataFrame by applying multiple conditions combined with OR logic.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame to be filtered.
    conditions : List[Union[Column, str]]
        A list of conditions, where each condition is either a PySpark Column or a string
        representing a condition (e.g., "age > 30").

    Returns
    -------
    DataFrame
        The filtered DataFrame after applying all conditions combined with OR.

    Examples
    --------
    # Example 1: Filter for users who are either active OR live in New York
    conditions = [
        col("status") == "active",
        col("city") == "New York"
    ]
    df_filtered = filter_rows_by_or_conditions(df, conditions)

    # Example 2: Filter for users who live in New York OR Los Angeles
    conditions_isin = [
        col("city").isin(["New York", "Los Angeles"])
    ]
    df_filtered_isin = filter_rows_by_or_conditions(df, conditions_isin)

    # Example 3: Filter for users whose city is NOT New York OR NOT Los Angeles
    conditions_is_not_in = [
        ~col("city").isin(["New York", "Los Angeles"])
    ]
    df_filtered_is_not_in = filter_rows_by_or_conditions(df, conditions_is_not_in)

    # Example 4: If no conditions are provided, the original DataFrame is returned
    df_filtered_empty = filter_rows_by_or_conditions(df, [])
    """

    if not conditions:
        # If no conditions are provided, return the original DataFrame unfiltered
        return df

    # Convert any string conditions to PySpark column expressions
    condition_exprs = [
        col(condition) if isinstance(condition, str) else condition
        for condition in conditions
    ]

    # Combine all conditions with OR using reduce
    combined_condition = reduce(lambda cond1, cond2: cond1 | cond2, condition_exprs)

    # Apply the combined condition to filter the DataFrame
    return df.filter(combined_condition)


def filter_rows_by_isin_conditions(df: DataFrame, column_value_map: dict) -> DataFrame:
    """
    Filters a PySpark DataFrame where each column matches one of the values specified in a dictionary using the 'isin' condition.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame to be filtered.
    column_value_map : dict
        A dictionary where keys are column names and values are lists of acceptable values for that column (e.g., {"city": ["New York", "San Francisco"]}).

    Returns
    -------
    DataFrame
        The filtered DataFrame where each column matches one of the specified values.

    Examples
    --------
    # Example: Filter rows where city is in ["New York", "Los Angeles"] AND status is in ["active"]
    column_value_map = {
        "city": ["New York", "Los Angeles"],
        "status": ["active"]
    }
    df_filtered = filter_rows_by_isin_conditions(df, column_value_map)
    """

    if not column_value_map:
        return df

    conditions = [
        col(column).isin(values) for column, values in column_value_map.items()
    ]
    combined_condition = reduce(lambda cond1, cond2: cond1 & cond2, conditions)

    return df.filter(combined_condition)


def filter_rows_by_date_range(
    df: DataFrame,
    date_column: str,
    start_date: Union[str, date],
    end_date: Union[str, date],
) -> DataFrame:
    """
    Filters a PySpark DataFrame to include only rows where the date_column is within the specified date range.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame to be filtered.
    date_column : str
        The name of the column containing date values to be filtered.
    start_date : Union[str, date]
        The start date of the range (inclusive). Can be a string in 'YYYY-MM-DD' format or a datetime.date object.
    end_date : Union[str, date]
        The end date of the range (inclusive). Can be a string in 'YYYY-MM-DD' format or a datetime.date object.

    Returns
    -------
    DataFrame
        The filtered DataFrame where the date_column is within the specified range.

    Examples
    --------
    # Filter rows where the 'event_date' is between '2023-01-01' and '2023-12-31'
    df_filtered = filter_rows_by_date_range(df, "event_date", "2023-01-01", "2023-12-31")
    """

    return df.filter((col(date_column) >= start_date) & (col(date_column) <= end_date))


def filter_rows_by_column_in_range(
    df: DataFrame,
    column_name: str,
    lower_bound: Union[int, float],
    upper_bound: Union[int, float],
) -> DataFrame:
    """
    Filters a PySpark DataFrame to include only rows where the column's value falls within a specified numeric range.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame to be filtered.
    column_name : str
        The name of the column containing numeric values to be filtered.
    lower_bound : Union[int, float]
        The lower bound of the range (inclusive).
    upper_bound : Union[int, float]
        The upper bound of the range (inclusive).

    Returns
    -------
    DataFrame
        The filtered DataFrame where the column's value is within the specified range.

    Examples
    --------
    # Filter rows where 'age' is between 30 and 40
    df_filtered = filter_rows_by_column_in_range(df, "age", 30, 40)
    """

    return df.filter(
        (col(column_name) >= lower_bound) & (col(column_name) <= upper_bound)
    )
