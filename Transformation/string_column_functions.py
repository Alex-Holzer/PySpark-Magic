"""
String Manipulation Functions for PySpark DataFrames.

This module provides a collection of utility functions for transforming, 
and manipulating string columns in PySpark DataFrames. These functions are optimized 
for large-scale data processing and are particularly useful in Databricks environments 
where efficient string operations are needed.

Usage:
    These functions are designed to be used within PySpark DataFrame operations. 
    Ensure that a valid Spark session is active before utilizing these functions.

Author: Alex Holzer
Date: 22.09.2024
Version: 1.0
"""

import re
from itertools import chain
from typing import Dict, List, Literal, Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.column import Column
from pyspark.sql.functions import (
    coalesce,
    col,
    concat,
    create_map,
    initcap,
    length,
    lit,
    lower,
    ltrim,
)
from pyspark.sql.functions import max as spark_max
from pyspark.sql.functions import min as spark_min
from pyspark.sql.functions import regexp_replace, rtrim, trim, udf, upper, when
from pyspark.sql.types import StringType

spark = SparkSession.builder.getOrCreate()


def _validate_column_existence(df: DataFrame, columns: Optional[List[str]]) -> None:
    """
    Validates the existence of specified columns in the DataFrame.

    Args:
        df: Input PySpark DataFrame.
        columns: List of columns to validate.

    Raises:
        ValueError: If the DataFrame is empty or if any specified columns don't exist.
    """
    if df.rdd.isEmpty():
        raise ValueError("Input DataFrame is empty.")

    if columns:
        non_existent_cols = set(columns) - set(df.columns)
        if non_existent_cols:
            raise ValueError(f"Columns not found in DataFrame: {non_existent_cols}")


def _get_string_columns(df: DataFrame) -> List[str]:
    """
    Retrieves all string columns from the DataFrame.

    Args:
        df: Input PySpark DataFrame.

    Returns:
        List of string column names.
    """
    return [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, StringType)
    ]


def _create_missing_value_expression(column: str, custom_patterns: List[str]) -> Column:
    """
    Creates an expression to identify missing values based on default and custom patterns.

    Args:
        column: Column name for which the missing value expression is created.
        custom_patterns: List of custom regex patterns to identify missing values.

    Returns:
        Column expression with missing values replaced by None.
    """
    base_patterns = [r"^[\s\W]*$", r"^\s*$", r".*null.*", r".*n/a.*"]
    all_patterns = base_patterns + custom_patterns
    pattern = "|".join(f"({p})" for p in all_patterns)

    return when(
        trim(lower(col(column))).rlike(pattern) | col(column).isNull(), lit(None)
    ).otherwise(col(column))


def replace_missing_values_with_none(
    df: DataFrame,
    columns: Optional[Union[str, List[str]]] = None,
    missing_patterns: Optional[Dict[str, List[str]]] = None,
) -> DataFrame:
    """
    Replace missing values with None in specified columns of a PySpark DataFrame.

    Args:
        df: Input PySpark DataFrame.
        columns: Column(s) to process. If None, all string columns are processed.
        missing_patterns: Custom patterns to identify missing values for specific columns.

    Returns:
        DataFrame with missing values replaced by None.

    Raises:
        ValueError: If the input DataFrame is empty or if specified columns don't exist.
    """
    columns_to_validate = [columns] if isinstance(columns, str) else columns
    _validate_column_existence(df, columns_to_validate)

    columns = (
        _get_string_columns(df)
        if columns is None
        else [columns] if isinstance(columns, str) else columns
    )
    missing_patterns = missing_patterns or {}

    for column in columns:
        custom_patterns = missing_patterns.get(column, [])
        df = df.withColumn(
            column, _create_missing_value_expression(column, custom_patterns)
        )

    return df


def add_string_to_column(
    df: DataFrame,
    column: str,
    string_to_add: str,
    position: Literal["start", "end"] = "end",
) -> DataFrame:
    """
    Add a string to a specified string column in a PySpark DataFrame.

    This function adds a given string to the specified column, either at the
    beginning or end of the existing string values.

    Args:
        df (DataFrame): Input PySpark DataFrame.
        column (str): Name of the column to modify.
        string_to_add (str): The string to add to the column values.
        position (Literal["start", "end"], optional): Where to add the string.
            Must be either "start" or "end". Defaults to "end".

    Returns:
        DataFrame: DataFrame with the modified column.

    Raises:
        ValueError: If the column doesn't exist or if position is invalid.

    Example:
        >>> df = spark.createDataFrame([("John",), ("Jane",)], ["name"])
        >>> result = add_string_to_column(df, "name", " Doe", "end")
        >>> result.show()
        +---------+
        |     name|
        +---------+
        |John Doe |
        |Jane Doe |
        +---------+

        >>> result = add_string_to_column(df, "name", "Mr. ", "start")
        >>> result.show()
        +---------+
        |     name|
        +---------+
        |Mr. John |
        |Mr. Jane |
        +---------+
    """
    # Validate the input column
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    # Validate position
    if position not in {"start", "end"}:
        raise ValueError("Position must be either 'start' or 'end'.")

    # Add string at the start or end of the column values
    if position == "start":
        return df.withColumn(column, concat(lit(string_to_add), col(column)))
    else:  # position == "end"
        return df.withColumn(column, concat(col(column), lit(string_to_add)))


def add_map_string_column(
    df: DataFrame, source_column: str, target_column: str, string_map: Dict[str, str]
) -> DataFrame:
    """
    Add a new column based on mapping values from an existing column.

    Args:
        df (DataFrame): Input DataFrame.
        source_column (str): Name of the source column to map from.
        target_column (str): Name of the new column to create with mapped values.
        string_map (Dict[str, str]): Dictionary mapping source strings to target strings.

    Returns:
        DataFrame: DataFrame with a new column containing mapped strings.

    Raises:
        ValueError: If the specified source column does not exist in the DataFrame.
        ValueError: If the target column already exists in the DataFrame.

    Example:
        >>> df = spark.createDataFrame([("apple",), ("banana",), ("cherry",)], ["fruit"])
        >>> string_map = {"apple": "red", "banana": "yellow", "cherry": "red"}
        >>> result = add_map_string_column(df, "fruit", "color", string_map)
        >>> result.show()
        +------+------+
        | fruit| color|
        +------+------+
        | apple|   red|
        |banana|yellow|
        |cherry|   red|
        +------+------+
    """
    # Validate source column existence
    if source_column not in df.columns:
        raise ValueError(
            f"Source column '{source_column}' does not exist in the DataFrame."
        )

    # Validate target column does not already exist
    if target_column in df.columns:
        raise ValueError(
            f"Target column '{target_column}' already exists in the DataFrame."
        )

    # Create a mapping expression from the string_map dictionary
    mapping_expr = create_map([lit(k) for k in chain(*string_map.items())])

    # Apply the mapping to create the new column
    return df.withColumn(target_column, mapping_expr[col(source_column)])


def map_strings(df: DataFrame, column: str, string_map: Dict[str, str]) -> DataFrame:
    """
    Change strings in a specified column according to a provided dictionary mapping.

    Args:
        df (DataFrame): Input DataFrame.
        column (str): Name of the column to modify.
        string_map (Dict[str, str]): Dictionary mapping old strings to new strings.

    Returns:
        DataFrame: DataFrame with modified strings in the specified column.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.

    Example:
        >>> df = spark.createDataFrame([("apple",), ("banana",), ("cherry",)], ["fruit"])
        >>> string_map = {"apple": "red", "banana": "yellow", "cherry": "red"}
        >>> result = map_strings(df, "fruit", string_map)
        >>> result.show()
        +------+
        | fruit|
        +------+
        |   red|
        |yellow|
        |   red|
        +------+
    """
    # Validate column existence
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    # Create a mapping expression from the string_map dictionary
    mapping_expr = create_map([lit(k) for k in chain(*string_map.items())])

    # Apply the mapping to the specified column, using the original value if not in the map
    return df.withColumn(column, coalesce(mapping_expr[col(column)], col(column)))


def capitalize_string_column(df: DataFrame, column: str) -> DataFrame:
    """
    Capitalizes the first letter of each word in the specified string column.

    Args:
        df (DataFrame): Input DataFrame.
        column (str): Name of the column to capitalize.

    Returns:
        DataFrame: DataFrame with the specified column capitalized.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.

    Example:
        >>> df = spark.createDataFrame([("john doe", "new york")], ["name", "city"])
        >>> result = capitalize_string_column(df, "name")
        >>> result.show()
        +---------+---------+
        |     name|     city|
        +---------+---------+
        |John Doe |new york |
        +---------+---------+
    """
    return df.withColumn(column, initcap(col(column)))


def lowercase_string_column(df: DataFrame, column: str) -> DataFrame:
    """
    Converts all characters in the specified string column to lowercase.

    Args:
        df (DataFrame): Input DataFrame.
        column (str): Name of the column to convert to lowercase.

    Returns:
        DataFrame: DataFrame with the specified column converted to lowercase.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.

    Example:
        >>> df = spark.createDataFrame([("JOHN DOE", "NEW YORK")], ["name", "city"])
        >>> result = lowercase_string_column(df, "name")
        >>> result.show()
        +---------+---------+
        |     name|     city|
        +---------+---------+
        |john doe |NEW YORK |
        +---------+---------+
    """
    return df.withColumn(column, lower(col(column)))


def uppercase_string_column(df: DataFrame, column: str) -> DataFrame:
    """
    Converts all characters in the specified string column to uppercase.

    Args:
        df (DataFrame): Input DataFrame.
        column (str): Name of the column to convert to uppercase.

    Returns:
        DataFrame: DataFrame with the specified column converted to uppercase.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.

    Example:
        >>> df = spark.createDataFrame([("John Doe", "New York")], ["name", "city"])
        >>> result = uppercase_string_column(df, "name")
        >>> result.show()
        +---------+---------+
        |     name|     city|
        +---------+---------+
        |JOHN DOE |New York |
        +---------+---------+
    """
    return df.withColumn(column, upper(col(column)))


def trim_whitespace(
    df: DataFrame,
    columns: Union[str, List[str]],
    trim_type: Literal["both", "left", "right"] = "both",
) -> DataFrame:
    """
    Trim whitespace from specified string columns.

    This function efficiently trims whitespace from the specified string columns
    in a PySpark DataFrame. It supports trimming from both sides, left side only,
    or right side only.

    Args:
        df (DataFrame): Input DataFrame.
        columns (Union[str, List[str]]): Column name or list of column names to trim.
        trim_type (Literal['both', 'left', 'right']): Type of trim operation.
            Options: 'both', 'left', 'right'. Default is 'both'.

    Returns:
        DataFrame: DataFrame with trimmed string columns.

    Raises:
        ValueError: If an invalid trim_type is provided.
        TypeError: If input types are incorrect.

    Example:
        >>> df = spark.createDataFrame([("  John  ", "  Doe  ")], ["first_name", "last_name"])
        >>> result = df.trim_whitespace(["first_name", "last_name"])
        >>> result.show()
        +----------+---------+
        |first_name|last_name|
        +----------+---------+
        |      John|      Doe|
        +----------+---------+
    """
    if trim_type not in ["both", "left", "right"]:
        raise ValueError("trim_type must be 'both', 'left', or 'right'")

    trim_func = {"both": trim, "left": ltrim, "right": rtrim}[trim_type]

    trim_exprs = [
        trim_func(col(c)).alias(c) if c in columns else col(c) for c in df.columns
    ]

    return df.select(*trim_exprs)


def trim_all_string_columns(
    df: DataFrame, trim_type: Literal["both", "left", "right"] = "both"
) -> DataFrame:
    """
    Trim whitespace from all string columns in the DataFrame.

    Args:
        df (DataFrame): Input DataFrame.
        trim_type (Literal['both', 'left', 'right']): Type of trim operation,
            either 'both', 'left', or 'right'. Default is 'both'.

    Returns:
        DataFrame: DataFrame with string columns trimmed.

    Raises:
        ValueError: If an invalid trim_type is provided.
    """
    # Validate trim_type
    if trim_type not in {"both", "left", "right"}:
        raise ValueError("trim_type must be 'both', 'left', or 'right'")

    # Select the appropriate trim function based on the trim_type
    trim_func = {"both": trim, "left": ltrim, "right": rtrim}[trim_type]

    # Get all string column names
    string_columns = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, StringType)
    ]

    # Apply the selected trim function only to string columns, leave other columns unchanged
    return df.select(
        [
            trim_func(col(c)).alias(c) if c in string_columns else col(c)
            for c in df.columns
        ]
    )


def replace_substring(
    df: DataFrame,
    column: str,
    replacements: Dict[str, Union[str, None]],
    whole_word: bool = False,
) -> DataFrame:
    """
    Replace or remove substrings in a specified column.

    Args:
        df (DataFrame): Input DataFrame.
        column (str): Column name to process.
        replacements (Dict[str, Union[str, None]]): Substrings to replace or remove.
        whole_word (bool): If True, replace whole words only. Defaults to False.

    Returns:
        DataFrame: Updated DataFrame.
    """
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    # Validate replacements dictionary
    for key, value in replacements.items():
        if not isinstance(key, str):
            raise TypeError(f"Invalid key type: {type(key)}")
        if value is not None and not isinstance(value, str):
            raise TypeError(f"Invalid value type: {type(value)}")

    # Start with the original column
    replaced_col = col(column)

    # Apply replacements sequentially
    for old, new in replacements.items():
        # Escape special characters to safely use in regex
        escaped_old = re.escape(old)

        # If whole_word is True, use word boundaries
        if whole_word:
            pattern = f"\\b{escaped_old}\\b"
        else:
            pattern = escaped_old

        # If new is None or an empty string, we're removing the substring
        new_value = new if new is not None else ""

        # Apply the replacement
        replaced_col = regexp_replace(replaced_col, pattern, new_value)

    # Return DataFrame with replaced column
    return df.withColumn(column, replaced_col)


def replace_substring_at_occurrence(
    df: DataFrame,
    column: str,
    replacements: Dict[str, Union[str, None]],
    at_occurrence: int,
    whole_word: bool = False,
) -> DataFrame:
    """
    Replace or remove a substring at a specific occurrence in a specified string column.

    This function replaces a substring in a single column of a PySpark DataFrame
    at a specified occurrence. It supports whole word replacements and character removal.

    Args:
        df (DataFrame): Input DataFrame.
        column (str): Name of the column to process.
        replacements (Dict[str, Union[str, None]]): Dictionary with a single key-value pair.
            The key is the substring to replace, and the value is its replacement.
            Use an empty string or None as the value to remove the substring.
        at_occurrence (int): The occurrence of the substring to replace (1-based index).
        whole_word (bool): If True, only whole words will be replaced. Defaults to False.

    Returns:
        DataFrame: DataFrame with the substring replaced or removed at the specified occurrence.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame,
                    if at_occurrence is less than 1, or if replacements dict has more than one item.
        TypeError: If the replacements dictionary contains non-string keys or values that are neither strings nor None.
    """
    # Error checking
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if at_occurrence < 1:
        raise ValueError("at_occurrence must be greater than or equal to 1.")

    if len(replacements) != 1:
        raise ValueError(
            "replacements dictionary must contain exactly one key-value pair."
        )

    substring, replacement = list(replacements.items())[0]

    if not isinstance(substring, str):
        raise TypeError("Replacement key must be a string.")

    if not (isinstance(replacement, str) or replacement is None):
        raise TypeError("Replacement value must be a string or None.")

    if replacement is None:
        replacement = ""

    # Define the UDF function
    def replace_at_occurrence(text):
        # If text is None, return None
        if text is None:
            return None

        occurrence_counter = 0

        # If whole_word is True, use word boundaries
        if whole_word:
            # Build the regex pattern with word boundaries
            pattern = r"\b" + re.escape(substring) + r"\b"
        else:
            # Use substring as is
            pattern = re.escape(substring)

        # Split the text with capturing groups
        parts = re.split("(" + pattern + ")", text)

        # Iterate over parts and replace at the specified occurrence
        for idx, part in enumerate(parts):
            if re.fullmatch(pattern, part):
                occurrence_counter += 1
                if occurrence_counter == at_occurrence:
                    parts[idx] = replacement

        # Reconstruct the text
        modified_text = "".join(parts)
        return modified_text

    # Register the UDF
    replace_udf = udf(replace_at_occurrence, StringType())

    # Apply the UDF to the DataFrame
    df = df.withColumn(column, replace_udf(col(column)))

    return df


def replace_substring_after_string(
    df: DataFrame,
    column: str,
    target: str,
    after_string: str,
    replacement: Union[str, None],
    whole_word: bool = False,
) -> DataFrame:
    """
    Replace or remove a substring, but only after a specified string occurs in the text.

    This function replaces a substring in a single column of a PySpark DataFrame
    after a specified string. It supports whole word replacements and character removal.

    Args:
        df (DataFrame): Input DataFrame.
        column (str): Name of the column to process.
        target (str): The substring to replace.
        after_string (str): The string after which the replacement should occur.
        replacement (Union[str, None]): The replacement string. Use an empty string or None to remove the substring.
        whole_word (bool): If True, only whole words will be replaced. Defaults to False.

    Returns:
        DataFrame: DataFrame with the substring replaced or removed after the specified string.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.
        TypeError: If the input types are incorrect.

    Examples:
        >>> df = spark.createDataFrame([
        ...     ("The quick brown fox jumps over the quick dog. The fox is quick.",)
        ... ], ["text"])
        >>> # Example 1: Replace "quick" with "slow" after "over the"
        >>> result1 = replace_substring_after_string(
        ...     df=df,
        ...     column="text",
        ...     target="quick",
        ...     after_string="over the",
        ...     replacement="slow"
        ... )
        >>> result1.show(truncate=False)
        +-----------------------------------------------------------------------------------+
        |text                                                                               |
        +-----------------------------------------------------------------------------------+
        |The quick brown fox jumps over the slow dog. The fox is quick.                     |
        +-----------------------------------------------------------------------------------+
    """
    # Error checking
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if not isinstance(target, str):
        raise TypeError("The 'target' must be a string.")

    if not isinstance(after_string, str):
        raise TypeError("The 'after_string' must be a string.")

    if not (isinstance(replacement, str) or replacement is None):
        raise TypeError("The 'replacement' must be a string or None.")

    if replacement is None:
        replacement = ""

    # Define the UDF function
    def replace_after(text):
        if text is None:
            return None

        # Find the last occurrence of after_string
        after_pos = text.find(after_string)
        if after_pos == -1:
            # after_string not found, return original text
            return text

        # Position after after_string
        start_pos = after_pos + len(after_string)

        before = text[:start_pos]
        after = text[start_pos:]

        # Prepare the pattern for target
        if whole_word:
            pattern = r"\b" + re.escape(target) + r"\b"
        else:
            pattern = re.escape(target)

        # Replace the target in the after part
        after_modified = re.sub(pattern, replacement, after)

        # Return the concatenation
        return before + after_modified

    # Register the UDF
    replace_udf = udf(replace_after, StringType())

    # Apply the UDF to the DataFrame
    df = df.withColumn(column, replace_udf(col(column)))

    return df


def concatenate_columns(
    df: DataFrame, columns: List[str], separator: str, new_column: str
) -> DataFrame:
    """
    Concatenate multiple string columns into a new column.

    Args:
        df (DataFrame): Input DataFrame.
        columns (List[str]): List of column names to concatenate.
        separator (str): Separator to use between concatenated values.
        new_column (str): Name of the new column to create.

    Returns:
        DataFrame: DataFrame with a new column containing concatenated strings.

    Example:
        >>> df = spark.createDataFrame([("John", "Doe", "Smith")], ["first_name", "middle_name", "last_name"])
        >>> result = concatenate_columns(df, ["first_name", "middle_name", "last_name"], " ", "full_name")
        >>> result.show()
        +----------+-----------+---------+------------------+
        |first_name|middle_name|last_name|         full_name|
        +----------+-----------+---------+------------------+
        |      John|        Doe|    Smith|John Doe Smith    |
        +----------+-----------+---------+------------------+
    """
    df = df.withColumn(new_column, concat(*[col(c) for c in columns], separator))
    return df


def swap_number_format(
    df: DataFrame, column: str, to_german: bool = False
) -> DataFrame:
    """
    Swap the number format in a string column between German and standard format.

    This function converts numbers in the format "3.200,50" to "3,200.50" (German to standard)
    or vice versa (standard to German). It handles numbers with or without decimal places.

    Args:
        df (DataFrame): Input PySpark DataFrame.
        column (str): Name of the column containing the numbers as strings.
        to_german (bool, optional): If True, convert from standard to German format.
                                    If False (default), convert from German to standard format.

    Returns:
        DataFrame: A new DataFrame with the specified column modified in-place.

    Raises:
        ValueError: If the input parameters are invalid.

    Example:
        >>> df = spark.createDataFrame([("3.200,50",), ("1.000",), ("500,75",)], ["amount"])
        >>> # Convert from German to standard format
        >>> result = swap_number_format(df, "amount")
        >>> result.show()
        +---------+
        |   amount|
        +---------+
        |3,200.50 |
        |1,000.00 |
        |  500.75 |
        +---------+
        >>>
        >>> # Convert back to German format
        >>> result = swap_number_format(result, "amount", to_german=True)
        >>> result.show()
        +---------+
        |   amount|
        +---------+
        |3.200,50 |
        |1.000,00 |
        |  500,75 |
        +---------+
    """
    if to_german:
        # Convert from standard to German format
        return df.withColumn(
            column,
            regexp_replace(
                regexp_replace(
                    # Add decimal zeros if missing
                    when(col(column).contains("."), col(column)).otherwise(
                        concat(col(column), ".00")
                    ),
                    r"(\d),(\d)",
                    r"\1~\2",  # Replace , with ~ temporarily
                ),
                r"\.",
                ",",  # Replace . with ,
            ),
        ).withColumn(
            column, regexp_replace(col(column), r"~", ".")  # Replace ~ with .
        )
    else:
        # Convert from German to standard format
        return df.withColumn(
            column,
            regexp_replace(
                regexp_replace(
                    # Add decimal zeros if missing
                    when(col(column).contains(","), col(column)).otherwise(
                        concat(col(column), ",00")
                    ),
                    r"(\d)\.(\d)",
                    r"\1~\2",  # Replace . with ~ temporarily
                ),
                r",",
                ".",  # Replace , with .
            ),
        ).withColumn(
            column, regexp_replace(col(column), r"~", ",")  # Replace ~ with ,
        )


def validate_equal_string_length(
    df: DataFrame, columns: Union[str, List[str]], raise_error: bool = False
) -> DataFrame:
    """
    Validates if all non-null strings in the specified column(s) have equal length.

    This function checks if all non-null values in the specified column(s) have the
    same length. It adds a new column for each checked column, indicating whether
    all strings have equal length. Optionally, it can raise an error if any
    inconsistent lengths are found.

    Args:
        df (DataFrame): Input PySpark DataFrame.
        columns (Union[str, List[str]]): Column name or list of column names to validate.
        raise_error (bool): If True, raises a ValueError when inconsistent lengths are found.
                            Defaults to False.

    Returns:
        DataFrame: The input DataFrame with additional boolean columns indicating
                   whether all strings in each checked column have equal length.

    Raises:
        ValueError: If any column has strings with inconsistent lengths and raise_error is True.

    Example:
        >>> df = spark.createDataFrame([("123", "ab"), ("456", "cd"), ("789", "ef")], ["col1", "col2"])
        >>> result = validate_equal_string_length(df, ["col1", "col2"])
        >>> result.show()
        +----+----+-------------------------+-------------------------+
        |col1|col2|col1_equal_string_length |col2_equal_string_length |
        +----+----+-------------------------+-------------------------+
        | 123|  ab|                    true |                    true |
        | 456|  cd|                    true |                    true |
        | 789|  ef|                    true |                    true |
        +----+----+-------------------------+-------------------------+
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        # Calculate min and max lengths for non-null values
        length_stats = df.agg(
            spark_min(length(col(column))).alias("min_length"),
            spark_max(length(col(column))).alias("max_length"),
        ).collect()[0]

        min_length, max_length = length_stats["min_length"], length_stats["max_length"]

        # Check if all non-null strings have equal length
        is_equal_length = min_length == max_length

        # Add a column indicating if all strings have equal length
        df = df.withColumn(
            f"{column}_equal_string_length",
            when(col(column).isNull(), lit(None)).otherwise(lit(is_equal_length)),
        )

        if not is_equal_length:
            error_message = f"Column '{column}' inconsistent lengths. Min length: {min_length}, Max length: {max_length}"
            print(f"⚠️ {error_message}")
            if raise_error:
                raise ValueError(error_message)

    return df
