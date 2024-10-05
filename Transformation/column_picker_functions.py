"""
column_picker_functions.py
===========================

This module provides utility functions for selecting columns 
based on their data types from a PySpark DataFrame.
The functions include selecting string and numeric columns, 
and also performing column-based operations utilizing 
PySpark's `StringIndexer`, `VectorAssembler`, and `ChiSquareTest`.
"""

from typing import List

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.stat import ChiSquareTest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()


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


def select_highly_correlated_columns(
    df: DataFrame, target_col: str, threshold: float
) -> DataFrame:
    """
    Selects categorical columns highly correlated with the target column using Chi-square test.

     Parameters
     ----------
     df : DataFrame
         Input DataFrame.
     target_col : str
         Target column for correlation.
     threshold : float
         Max p-value to consider a column as correlated.

     Returns
     -------
     DataFrame
         DataFrame with only highly correlated columns.


    """
    # Step 1: Identify string columns (categorical columns)
    string_columns = [
        col_name
        for col_name, dtype in df.dtypes
        if dtype == "string" and col_name != target_col
    ]

    # Step 2: Index the string columns and the target column
    indexers = [
        StringIndexer(inputCol=col_name, outputCol=col_name + "_indexed").fit(df)
        for col_name in string_columns + [target_col]
    ]

    # Apply the indexers and transform the DataFrame
    for indexer in indexers:
        df = indexer.transform(df)

    # List of indexed string columns (excluding target column)
    indexed_columns = [col_name + "_indexed" for col_name in string_columns]

    # Assemble all indexed columns into a feature vector
    assembler = VectorAssembler(inputCols=indexed_columns, outputCol="features")
    df_vector = assembler.transform(df)

    # Step 3: Perform Chi-Square Test to compute correlations
    chi_sq_result = ChiSquareTest.test(
        df_vector, "features", target_col + "_indexed"
    ).head()

    # Step 4: Identify highly correlated columns based on p-values
    correlated_columns = []
    for i, p_value in enumerate(chi_sq_result.pValues):
        if p_value < threshold:
            correlated_columns.append(string_columns[i])  # Get the original column name

    # Step 5: Select the correlated columns and return the resulting DataFrame
    return df.select(*correlated_columns)
