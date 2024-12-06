"""

date_transformation_functions.py
===========================

This module provides a collection of utility functions for transforming date columns



"""

from datetime import datetime
from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    coalesce,
    col,
    current_timestamp,
    expr,
    regexp_replace,
    to_date,
    to_timestamp,
)
from pyspark.sql.types import TimestampType

spark: SparkSession = SparkSession.builder.getOrCreate()


def cast_columns_to_date(
    df: DataFrame, columns: List[str], date_format: str
) -> DataFrame:
    """
    Casts the specified string columns to a date format in the given DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the string columns to be cast.
    columns : list of str
        A list of column names in the DataFrame that need to be cast to date format.
    date_format : str
        The desired date format to cast the string columns to.

    Returns
    -------
    DataFrame
        The DataFrame with the specified columns cast to date format.

    Example
    -------
    >>> df = cast_columns_to_date(df, ['column1', 'column2'], 'yyyy-MM-dd')
    """
    for column in columns:
        df = df.withColumn(column, to_date(col(column), date_format))
    return df


def cast_string_to_timestamp(
    df: DataFrame, columns: list[str], timestamp_format: str
) -> DataFrame:
    """
    Casts the specified string columns to a timestamp format in the given DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the string columns to be cast.
    columns : list of str
        A list of column names in the DataFrame that need to be cast to timestamp format.
    timestamp_format : str
        The desired timestamp format to cast the string columns to.

    Returns
    -------
    DataFrame
        The DataFrame with the specified columns cast to timestamp format.
    """
    for column in columns:
        df = df.withColumn(column, to_timestamp(col(column), timestamp_format))
    return df


def cast_timestamp_to_date(df: DataFrame, columns: list[str]) -> DataFrame:
    """
    Casts the specified timestamp columns to date format in the given DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the timestamp columns to be cast.
    columns : list of str
        A list of column names in the DataFrame that need to be cast to date format.

    Returns
    -------
    DataFrame
        The DataFrame with the specified timestamp columns cast to date format.
    """
    for column in columns:
        df = df.withColumn(column, to_date(col(column)))
    return df


def add_current_timestamp_column(
    df: DataFrame, column_name: str = "timestamp", time_zone: Optional[str] = None
) -> DataFrame:
    """
    Adds a column with the current timestamp to the DataFrame.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    column_name : str, optional
        Name of the new timestamp column, by default "timestamp".
    time_zone : Optional[str], optional
        Time zone for the timestamp. If None, uses the system's local time zone.
        Example values: "UTC", "America/New_York", "Europe/London".

    Returns
    -------
    DataFrame
        DataFrame with the new timestamp column added.

    Examples
    --------
    >>> df = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "value"])
    >>> df_with_timestamp = add_current_timestamp_column(df, "created_at", "UTC")
    >>> df_with_timestamp.show(truncate=False)
    +---+-----+-----------------------+
    | id|value|created_at             |
    +---+-----+-----------------------+
    |  1|    A|2023-09-11 12:34:56.789|
    |  2|    B|2023-09-11 12:34:56.789|
    +---+-----+-----------------------+
    """
    if time_zone:
        # Use the specified time zone with expression-based casting
        timestamp_col = expr(f"current_timestamp() at time zone '{time_zone}'")
        df = df.withColumn(column_name, timestamp_col.cast(TimestampType()))
    else:
        # Use the system's local time zone
        df = df.withColumn(column_name, current_timestamp())

    return df


def detect_timestamp_format(timestamp: str) -> Optional[str]:
    """
    Detect the format of a given timestamp string.

    This function attempts to parse the input string using various common timestamp formats
    and returns the format string that successfully parses the input.

    Args:
        timestamp (str): The timestamp string to analyze.

    Returns:
        Optional[str]: The format string if detected, None if no matching format is found.

    Examples:
        >>> detect_timestamp_format("2023-05-15 14:30:00")
        'yyyy-MM-dd HH:mm:ss'

        >>> detect_timestamp_format("2023-05-15 14:30:00.123")
        'yyyy-MM-dd HH:mm:ss.SSS'

        >>> detect_timestamp_format("2023-05-15T14:30:00")
        "yyyy-MM-dd'T'HH:mm:ss"

        >>> detect_timestamp_format("2023-05-15T14:30:00.123")
        "yyyy-MM-dd'T'HH:mm:ss.SSS"

        >>> detect_timestamp_format("2023-05-15T14:30:00Z")
        "yyyy-MM-dd'T'HH:mm:ss'Z'"

        >>> detect_timestamp_format("2023-05-15T14:30:00.123Z")
        "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"

        >>> detect_timestamp_format("2023-05-15")
        'yyyy-MM-dd'

        >>> detect_timestamp_format("05/15/2023")
        'MM/dd/yyyy'

        >>> detect_timestamp_format("15/05/2023")
        'dd/MM/yyyy'

        >>> detect_timestamp_format("15.05.2023")
        'dd.MM.yyyy'

        >>> detect_timestamp_format("15-05-2023")
        'dd-MM-yyyy'

        >>> detect_timestamp_format("2023/05/15")
        'yyyy/MM/dd'

        >>> detect_timestamp_format("14:30:00")
        'HH:mm:ss'

        >>> detect_timestamp_format("14:30:00.123")
        'HH:mm:ss.SSS'

        >>> detect_timestamp_format("15.05.2023 14:30:00")
        'dd.MM.yyyy HH:mm:ss'

        >>> detect_timestamp_format("15.05.2023 14:30:00.123")
        'dd.MM.yyyy HH:mm:ss.SSS'

        >>> detect_timestamp_format("05/15/2023 14:30:00")
        'MM/dd/yyyy HH:mm:ss'

        >>> detect_timestamp_format("05/15/2023 14:30:00.123")
        'MM/dd/yyyy HH:mm:ss.SSS'

        >>> detect_timestamp_format("15/05/2023 14:30:00")
        'dd/MM/yyyy HH:mm:ss'

        >>> detect_timestamp_format("15/05/2023 14:30:00.123")
        'dd/MM/yyyy HH:mm:ss.SSS'

        >>> detect_timestamp_format("2023/05/15 14:30:00")
        'yyyy/MM/dd HH:mm:ss'

        >>> detect_timestamp_format("2023/05/15 14:30:00.123")
        'yyyy/MM/dd HH:mm:ss.SSS'

        >>> detect_timestamp_format("15-May-2023")
        'dd-MMM-yyyy'

        >>> detect_timestamp_format("15 May 2023")
        'dd MMM yyyy'

        >>> detect_timestamp_format("May 15, 2023")
        'MMM dd, yyyy'

        >>> detect_timestamp_format("15-May-2023")
        'dd-MMMM-yyyy'

        >>> detect_timestamp_format("15 May 2023")
        'dd MMMM yyyy'

        >>> detect_timestamp_format("May 15, 2023")
        'MMMM dd, yyyy'

        >>> detect_timestamp_format("Mon, 15 May 2023")
        'E, dd MMM yyyy'

        >>> detect_timestamp_format("Mon, 15 May 2023 14:30:00")
        'E, dd MMM yyyy HH:mm:ss'

        >>> detect_timestamp_format("Monday, 15 May 2023")
        'EEEE, dd MMMM yyyy'

        >>> detect_timestamp_format("Monday, 15 May 2023 14:30:00")
        'EEEE, dd MMMM yyyy HH:mm:ss'

        >>> detect_timestamp_format("1684159800")
        'unix_seconds'

        >>> detect_timestamp_format("1684159800000")
        'unix_milliseconds'

        >>> detect_timestamp_format("2023-05-15T14:30:00+01:00")
        "yyyy-MM-dd'T'HH:mm:ssXXX"

        >>> detect_timestamp_format("2023-05-15T14:30:00.123+01:00")
        "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"

        >>> detect_timestamp_format("20230515")
        'yyyyMMdd'

        >>> detect_timestamp_format("20230515143000")
        'yyyyMMddHHmmss'

        >>> detect_timestamp_format("20230515 143000")
        'yyyyMMdd HHmmss'

        >>> detect_timestamp_format("2023-05-15 14.30.00")
        'yyyy-MM-dd HH.mm.ss'

        >>> detect_timestamp_format("15-05-2023 14:30:00")
        'dd-MM-yyyy HH:mm:ss'

    Note:
        - This function attempts to match the input against a predefined set of formats.
        - If no matching format is found, it returns None.
        - For Unix timestamps, it checks if the string represents a valid Unix timestamp.
    """
    formats = [
        ("%Y-%m-%d %H:%M:%S", "yyyy-MM-dd HH:mm:ss"),
        ("%Y-%m-%d %H:%M:%S.%f", "yyyy-MM-dd HH:mm:ss.SSS"),
        ("%Y-%m-%dT%H:%M:%S", "yyyy-MM-dd'T'HH:mm:ss"),
        ("%Y-%m-%dT%H:%M:%S.%f", "yyyy-MM-dd'T'HH:mm:ss.SSS"),
        ("%Y-%m-%dT%H:%M:%SZ", "yyyy-MM-dd'T'HH:mm:ss'Z'"),
        ("%Y-%m-%dT%H:%M:%S.%fZ", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),
        ("%Y-%m-%d", "yyyy-MM-dd"),
        ("%m/%d/%Y", "MM/dd/yyyy"),
        ("%d/%m/%Y", "dd/MM/yyyy"),
        ("%d.%m.%Y", "dd.MM.yyyy"),
        ("%d-%m-%Y", "dd-MM-yyyy"),
        ("%Y/%m/%d", "yyyy/MM/dd"),
        ("%H:%M:%S", "HH:mm:ss"),
        ("%H:%M:%S.%f", "HH:mm:ss.SSS"),
        ("%d.%m.%Y %H:%M:%S", "dd.MM.yyyy HH:mm:ss"),
        ("%d.%m.%Y %H:%M:%S.%f", "dd.MM.yyyy HH:mm:ss.SSS"),
        ("%m/%d/%Y %H:%M:%S", "MM/dd/yyyy HH:mm:ss"),
        ("%m/%d/%Y %H:%M:%S.%f", "MM/dd/yyyy HH:mm:ss.SSS"),
        ("%d/%m/%Y %H:%M:%S", "dd/MM/yyyy HH:mm:ss"),
        ("%d/%m/%Y %H:%M:%S.%f", "dd/MM/yyyy HH:mm:ss.SSS"),
        ("%Y/%m/%d %H:%M:%S", "yyyy/MM/dd HH:mm:ss"),
        ("%Y/%m/%d %H:%M:%S.%f", "yyyy/MM/dd HH:mm:ss.SSS"),
        ("%d-%b-%Y", "dd-MMM-yyyy"),
        ("%d %b %Y", "dd MMM yyyy"),
        ("%b %d, %Y", "MMM dd, yyyy"),
        ("%d-%B-%Y", "dd-MMMM-yyyy"),
        ("%d %B %Y", "dd MMMM yyyy"),
        ("%B %d, %Y", "MMMM dd, yyyy"),
        ("%a, %d %b %Y", "E, dd MMM yyyy"),
        ("%a, %d %b %Y %H:%M:%S", "E, dd MMM yyyy HH:mm:ss"),
        ("%A, %d %B %Y", "EEEE, dd MMMM yyyy"),
        ("%A, %d %B %Y %H:%M:%S", "EEEE, dd MMMM yyyy HH:mm:ss"),
        ("%Y-%m-%dT%H:%M:%S%z", "yyyy-MM-dd'T'HH:mm:ssXXX"),
        ("%Y-%m-%dT%H:%M:%S.%f%z", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"),
        ("%Y%m%d", "yyyyMMdd"),
        ("%Y%m%d%H%M%S", "yyyyMMddHHmmss"),
        ("%Y%m%d %H%M%S", "yyyyMMdd HHmmss"),
        ("%Y-%m-%d %H.%M.%S", "yyyy-MM-dd HH.mm.ss"),
        ("%d-%m-%Y %H:%M:%S", "dd-MM-yyyy HH:mm:ss"),
    ]

    for python_format, spark_format in formats:
        try:
            datetime.strptime(timestamp, python_format)
            return spark_format
        except ValueError:
            continue

    # Check for Unix timestamp (seconds)
    try:
        int_timestamp = int(timestamp)
        if (
            1000000000 <= int_timestamp <= 9999999999
        ):  # Reasonable range for Unix timestamps
            return "unix_seconds"
    except ValueError:
        pass

    # Check for Unix timestamp (milliseconds)
    try:
        int_timestamp = int(timestamp)
        if (
            1000000000000 <= int_timestamp <= 9999999999999
        ):  # Reasonable range for Unix timestamps in milliseconds
            return "unix_milliseconds"
    except ValueError:
        pass

    return None


def normalize_fractional_seconds(col_name):
    # This regex captures timestamps and normalizes the fractional part to at most 9 digits
    # by truncating if it's longer.
    return regexp_replace(col_name, r"(\.\d{9})\d+", r"\1")


def parse_iso_timestamp(df, column_name, new_column_name):
    """
    Converts a column containing ISO 8601-like string timestamps with varying fractional second precision
    and time zone offsets into a Spark TimestampType column. Returns a DataFrame with an added or replaced
    column containing the converted timestamps.

    This function tries multiple timestamp formats to accommodate different lengths of fractional seconds.
    It uses `coalesce` to return the first successfully parsed result.

    Parameters
    ----------
    df : pyspark.sql.dataframe.DataFrame
        Input DataFrame
    column_name : str
        Name of the column containing the string timestamps
    new_column_name : str
        Name of the new column to store the parsed timestamps

    Returns
    -------
    pyspark.sql.dataframe.DataFrame
        A DataFrame with an additional column containing parsed timestamps.
    """

    # Define multiple formats to handle variable fractions of a second
    # and different lengths of timestamp strings.
    # The patterns assume ISO 8601 structure like:
    # yyyy-MM-dd'T'HH:mm:ss[.fractional_seconds]+HH:MM
    # We start from no fractional second up to a high count of fractional digits.
    patterns = [
        "yyyy-MM-dd'T'HH:mm:ssXXX",
        "yyyy-MM-dd'T'HH:mm:ss.SXXX",
        "yyyy-MM-dd'T'HH:mm:ss.SSXXX",
        "yyyy-MM-dd'T'HH:mm:ss.SSSXXX",
        "yyyy-MM-dd'T'HH:mm:ss.SSSSXXX",
        "yyyy-MM-dd'T'HH:mm:ss.SSSSSXXX",
        "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
        "yyyy-MM-dd'T'HH:mm:ss.SSSSSSSXXX",
    ]

    # Apply to_timestamp with each pattern and coalesce results
    parsed_col = coalesce(*[to_timestamp(col(column_name), fmt) for fmt in patterns])

    # Return DataFrame with the new parsed timestamp column
    return df.withColumn(new_column_name, parsed_col)
