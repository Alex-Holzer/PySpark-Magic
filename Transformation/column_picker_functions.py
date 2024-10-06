"""
column_picker_functions.py
===========================

This module provides utility functions for selecting columns 
based on their data types from a PySpark DataFrame.
The functions include selecting string and numeric columns, 
and also performing column-based operations utilizing 
PySpark's `StringIndexer`, `VectorAssembler`, and `ChiSquareTest`.
"""

import math
from collections import defaultdict
from itertools import combinations
from typing import Dict, List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, lit, when
from pyspark.sql.functions import sum as _sum
from pyspark.sql.types import StringType


def select_column_with_prefix(df: DataFrame, prefix: str) -> DataFrame:
    """
    Select columns from the DataFrame that start with the specified prefix.

    Parameters
    ----------
    df : DataFrame
        Input PySpark DataFrame from which columns are to be selected.
    prefix : str
        Prefix string to filter columns by their names.

    Returns
    -------
    DataFrame
        A DataFrame containing only the columns that start with the given prefix.
    """
    # Get a list of columns that start with the specified prefix
    columns_with_prefix: List[str] = [
        col for col in df.columns if col.startswith(prefix)
    ]

    # Select the columns with the specified prefix
    return df.select(*columns_with_prefix)


def select_column_with_suffix(df: DataFrame, suffix: str) -> DataFrame:
    """
    Select columns from the DataFrame that end with the specified suffix.

    Parameters
    ----------
    df : DataFrame
        Input PySpark DataFrame from which columns are to be selected.
    suffix : str
        Suffix string to filter columns by their names.

    Returns
    -------
    DataFrame
        A DataFrame containing only the columns that end with the given suffix.
    """
    # Get a list of columns that end with the specified suffix
    columns_with_suffix: List[str] = [col for col in df.columns if col.endswith(suffix)]

    # Select the columns with the specified suffix
    return df.select(*columns_with_suffix)


def select_string_columns(df: DataFrame) -> DataFrame:
    """
    Selects all columns with data type 'string' from the input PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the string columns from the input DataFrame.
    """
    # Identify string columns
    string_columns: List[str] = [
        col_name for col_name, dtype in df.dtypes if dtype == "string"
    ]

    # Select and return the DataFrame with only string columns
    return df.select(*string_columns)


def select_numeric_columns(df: DataFrame) -> DataFrame:
    """
    Selects all columns with numeric data types from the input PySpark DataFrame.

    Numeric types considered:
    - 'int', 'bigint', 'float', 'double', 'decimal'

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the numeric columns from the input DataFrame.
    """
    # Define numeric types
    numeric_types: List[str] = ["int", "bigint", "float", "double", "decimal"]

    # Identify numeric columns
    numeric_columns: List[str] = [
        col_name for col_name, dtype in df.dtypes if dtype in numeric_types
    ]

    # Select and return the DataFrame with only numeric columns
    return df.select(*numeric_columns)


def select_date_columns(df: DataFrame) -> DataFrame:
    """
    Selects all columns with date-related data types ('date', 'timestamp')
    from the input PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the date and timestamp columns
        from the input DataFrame.
    """
    # Define date-related types
    date_types: List[str] = ["date", "timestamp"]

    # Identify date-related columns
    date_columns: List[str] = [
        col_name for col_name, dtype in df.dtypes if dtype in date_types
    ]

    # Select and return the DataFrame with only date-related columns
    return df.select(*date_columns)


def select_complex_columns(df: DataFrame) -> DataFrame:
    """
    Selects all columns with complex data types ('array', 'map', 'struct')
    from the input PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the complex columns (array, map, struct)
        from the input DataFrame.
    """
    # Define complex types
    complex_types: List[str] = ["array", "map", "struct"]

    # Identify complex columns
    complex_columns: List[str] = [
        col_name
        for col_name, dtype in df.dtypes
        if any(ct in dtype for ct in complex_types)
    ]

    # Select and return the DataFrame with only complex columns
    return df.select(*complex_columns)


def select_binary_columns(df: DataFrame) -> DataFrame:
    """
    Selects all columns with binary data types ('binary') from the input PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the binary columns from the input DataFrame.
    """
    # Define binary type
    binary_type: str = "binary"

    # Identify binary columns
    binary_columns: List[str] = [
        col_name for col_name, dtype in df.dtypes if dtype == binary_type
    ]

    # Select and return the DataFrame with only binary columns
    return df.select(*binary_columns)


def select_struct_columns(df: DataFrame) -> DataFrame:
    """
    Selects all columns with 'struct' data types from the input PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the 'struct' columns from the input DataFrame.
    """
    # Define struct type
    struct_type: str = "struct"

    # Identify struct columns
    struct_columns: List[str] = [
        col_name for col_name, dtype in df.dtypes if struct_type in dtype
    ]

    # Select and return the DataFrame with only struct columns
    return df.select(*struct_columns)


def select_array_columns(df: DataFrame) -> DataFrame:
    """
    Selects all columns with 'array' data types from the input PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the 'array' columns from the input DataFrame.
    """
    # Define array type
    array_type: str = "array"

    # Identify array columns
    array_columns: List[str] = [
        col_name for col_name, dtype in df.dtypes if array_type in dtype
    ]

    # Select and return the DataFrame with only array columns
    return df.select(*array_columns)


def select_map_columns(df: DataFrame) -> DataFrame:
    """
    Selects all columns with 'map' data types from the input PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the 'map' columns from the input DataFrame.
    """
    # Define map type
    map_type: str = "map"

    # Identify map columns
    map_columns: List[str] = [
        col_name for col_name, dtype in df.dtypes if map_type in dtype
    ]

    # Select and return the DataFrame with only map columns
    return df.select(*map_columns)


def select_columns_by_name_pattern(df: DataFrame, pattern: str) -> DataFrame:
    """
    Selects all columns from the DataFrame whose column names contain the given pattern (substring).

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.
    pattern : str
        The pattern (substring) to search for in the column names.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the columns whose names contain the given pattern.
    """
    # Identify columns that contain the pattern in their names (case-insensitive)
    selected_columns: List[str] = [
        col_name for col_name in df.columns if pattern.lower() in col_name.lower()
    ]

    # Select and return the DataFrame with the filtered columns
    return df.select(*selected_columns)


def select_columns_by_value_pattern(df: DataFrame, pattern: str) -> DataFrame:
    """
    Selects all columns from the DataFrame where any of the rows contain the given pattern.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.
    pattern : str
        The pattern (substring) to search for in the column values.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the columns where the pattern exists in any row.
    """
    # Initialize an empty list to store columns that contain the pattern
    matching_columns: List[str] = []

    # Iterate over each column and check if any value contains the pattern
    for column in df.columns:
        if df.filter(col(column).cast("string").contains(pattern)).count() > 0:
            matching_columns.append(column)

    # Select and return the DataFrame with the filtered columns
    return df.select(*matching_columns)


def select_non_null_columns(df: DataFrame) -> DataFrame:
    """
    Selects all columns from the DataFrame that contain no null values.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the columns without any null values.
    """
    non_null_columns = [
        col_name
        for col_name in df.columns
        if df.filter(col(col_name).isNotNull()).count() == df.count()
    ]
    return df.select(*non_null_columns)


def select_columns_with_nulls(df: DataFrame) -> DataFrame:
    """
    Selects columns that contain at least one null value.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame with only the columns that contain null values.
    """
    columns_with_nulls = [
        col_name
        for col_name in df.columns
        if df.filter(col(col_name).isNull()).count() > 0
    ]
    return df.select(*columns_with_nulls)


def select_columns_with_unique_values(df: DataFrame) -> DataFrame:
    """
    Selects columns where all values are unique (no duplicates).

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the columns where all values are unique.
    """
    unique_columns = [
        col_name
        for col_name in df.columns
        if df.select(col_name).distinct().count() == df.count()
    ]
    return df.select(*unique_columns)


def select_columns_with_low_cardinality(
    df: DataFrame, max_unique_count: int
) -> DataFrame:
    """
    Selects columns from the DataFrame where the number of unique values is less
    than or equal to the specified threshold.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.
    max_unique_count : int
        The maximum number of unique values a column must have to be considered low cardinality.

    Returns
    -------
    DataFrame
        A new DataFrame containing only the low cardinality columns.
    """
    # Identify low cardinality columns
    low_cardinality_columns: List[str] = [
        col_name
        for col_name in df.columns
        if df.select(col_name).distinct().count() <= max_unique_count
    ]

    # Select and return the DataFrame with only the low cardinality columns
    return df.select(*low_cardinality_columns)


# Initialize Spark Session (if not already initialized)
spark = SparkSession.builder.getOrCreate()


def find_highly_correlated_string_columns(
    df: DataFrame, threshold: float = 0.7, missing_placeholder: str = "__MISSING__"
) -> List[List[str]]:
    """
    Identifies groups of string (categorical) columns in a PySpark DataFrame that are highly correlated.
    Correlation is measured using Cramér's V statistic.

    Parameters:
    -----------
    df : DataFrame
        The input PySpark DataFrame containing string columns to analyze.
    threshold : float, optional
        The Cramér's V threshold above which columns are considered highly correlated.
        Defaults to 0.7.
    missing_placeholder : str, optional
        The string to replace null values with in string columns.
        Defaults to "__MISSING__".

    Returns:
    --------
    List[List[str]]
        A list of groups, where each group is a list of column names that are highly correlated.

    Raises:
    -------
    ValueError
        If no string columns are found in the DataFrame.
    """

    # Step 1: Identify string columns
    string_cols = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, StringType)
    ]

    if not string_cols:
        raise ValueError("No string columns found in the DataFrame.")

    # Step 2: Handle missing values by replacing nulls with a placeholder
    df_filled = df.select(
        [
            when(col(c).isNull(), lit(missing_placeholder)).otherwise(col(c)).alias(c)
            for c in string_cols
        ]
    )

    # Cache the filled DataFrame as it will be used multiple times
    df_filled.cache()
    df_filled_count = df_filled.count()  # Trigger caching

    # Precompute the number of unique categories for each column
    unique_counts = {}
    for c in string_cols:
        unique_counts[c] = df_filled.select(c).distinct().count()

    # Broadcast the unique counts for efficiency
    spark = df_filled.sparkSession
    bc_unique_counts = spark.sparkContext.broadcast(unique_counts)

    # Initialize Union-Find structure for grouping
    parent = {col_name: col_name for col_name in string_cols}

    def find(x: str) -> str:
        # Path Compression
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Step 3: Compute Cramér's V for each pair of string columns
    for col1, col2 in combinations(string_cols, 2):
        # Create contingency table
        contingency = df_filled.groupBy(col1, col2).agg(count(lit(1)).alias("count"))

        # Compute chi-squared statistic
        # Step 3.1: Compute the total number of observations
        n = df_filled_count

        # Step 3.2: Compute the observed frequencies
        # To compute chi-squared, we need the observed counts and expected counts
        # However, computing expected counts directly is expensive
        # Instead, we'll use an approximation for large datasets

        # Compute row sums and column sums
        row_sums = df_filled.groupBy(col1).agg(_sum(lit(1)).alias("row_sum"))
        col_sums = df_filled.groupBy(col2).agg(_sum(lit(1)).alias("col_sum"))

        # Join contingency with row_sums and col_sums to compute expected counts
        contingency_with_totals = contingency.join(row_sums, on=col1, how="left").join(
            col_sums, on=col2, how="left"
        )

        # Calculate expected counts and chi-squared components
        # E_ij = (row_sum_i * col_sum_j) / n
        # (O_ij - E_ij)^2 / E_ij
        contingency_with_calc = contingency_with_totals.withColumn(
            "expected", (col("row_sum") * col("col_sum")) / lit(n)
        ).withColumn(
            "chi_sq_component",
            ((col("count") - col("expected")) ** 2) / col("expected"),
        )

        # Sum all chi_sq_components to get chi-squared statistic
        chi2 = contingency_with_calc.agg(
            _sum("chi_sq_component").alias("chi2")
        ).collect()[0]["chi2"]

        # Calculate Cramér's V
        k1 = bc_unique_counts.value[col1]
        k2 = bc_unique_counts.value[col2]
        min_dim = min(k1, k2) - 1
        if min_dim > 0 and n > 0:
            cramer_v = math.sqrt(chi2 / (n * min_dim))
        else:
            cramer_v = 0.0

        if cramer_v >= threshold:
            union(col1, col2)

    # Step 4: Group columns that are highly correlated
    group_dict: Dict[str, List[str]] = defaultdict(list)
    for col_name in string_cols:
        root = find(col_name)
        group_dict[root].append(col_name)

    # Filter out groups with only one column
    groups = [group for group in group_dict.values() if len(group) > 1]

    # Unpersist the cached DataFrame
    df_filled.unpersist()

    return groups
