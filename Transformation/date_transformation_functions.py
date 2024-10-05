from pyspark.sql import DataFrame, SparkSession

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()
from datetime import datetime
from typing import List, Literal, Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql.functions import current_timestamp, expr
from pyspark.sql.types import TimestampType
from typing import Optional


from pyspark.sql.functions import (
    col,
    current_timestamp,
    date_format,
    dayofmonth,
    dayofweek,
    dayofyear,
    expr,
    from_unixtime,
    hour,
    last_day,
    lit,
    minute,
    month,
    quarter,
    second,
    to_date,
    to_timestamp,
    weekofyear,
    when,
    year,
    concat,
)
from pyspark.sql.types import TimestampType


def cast_string_to_date(
    df: DataFrame, columns: list[str], date_format: str
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



def format_date_column(df: DataFrame, column: str, new_column_name: str, format: str = "MM.yyyy") -> DataFrame:
    """
    Formats a date column to the specified format, defaulting to "MM.yyyy".

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the date column.
    column : str
        The name of the date column to be formatted.
    new_column_name : str
        The name of the new column that will contain the formatted date.
    format : str, optional
        The desired date format. Defaults to "MM.yyyy".

    Returns
    -------
    DataFrame
        DataFrame with a new column containing the formatted date.

    Examples
    --------
    >>> df = spark.createDataFrame([("2020-12-15",), ("2021-01-10",)], ["date_col"])
    >>> df_formatted = format_date_column(df, "date_col", "formatted_date", "dd/MM/yyyy")
    >>> df_formatted.show(truncate=False)
    +----------+--------------+
    |date_col  |formatted_date|
    +----------+--------------+
    |2020-12-15|15/12/2020    |
    |2021-01-10|10/01/2021    |
    +----------+--------------+
    """
    return df.withColumn(new_column_name, date_format(col(column), format))




def add_day_of_week_column(df: DataFrame, date_column: str, new_column_name: str = "day_of_week") -> DataFrame:
    """
    Adds a new column to the DataFrame with the day of the week as long text (e.g., 'Monday') 
    derived from a date column.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the date column.
    date_column : str
        The name of the date column from which the day of the week is derived.
    new_column_name : str, optional
        The name of the new column containing the day of the week. Defaults to "day_of_week".

    Returns
    -------
    DataFrame
        DataFrame with a new column containing the day of the week as long text.

    Examples
    --------
    >>> df = spark.createDataFrame([("2023-09-28",), ("2024-01-01",)], ["date_col"])
    >>> df_with_day_of_week = add_day_of_week_column(df, "date_col")
    >>> df_with_day_of_week.show(truncate=False)
    +----------+------------+
    |date_col  |day_of_week |
    +----------+------------+
    |2023-09-28|Thursday    |
    |2024-01-01|Monday      |
    +----------+------------+
    """
    return df.withColumn(new_column_name, date_format(col(date_column), "EEEE"))



def add_month_column(df: DataFrame, date_column: str, new_column_name: str = "month") -> DataFrame:
    """
    Adds a new column to the DataFrame with the month extracted from a date column.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the date column.
    date_column : str
        The name of the date column from which the month is extracted.
    new_column_name : str, optional
        The name of the new column containing the month. Defaults to "month".

    Returns
    -------
    DataFrame
        DataFrame with a new column containing the extracted month.

    Examples
    --------
    >>> df = spark.createDataFrame([("2023-09-28",), ("2024-01-01",)], ["date_col"])
    >>> df_with_month = add_month_column(df, "date_col")
    >>> df_with_month.show(truncate=False)
    +----------+-----+
    |date_col  |month|
    +----------+-----+
    |2023-09-28|9    |
    |2024-01-01|1    |
    +----------+-----+
    """
    return df.withColumn(new_column_name, month(col(date_column)))




def add_year_column(df: DataFrame, date_column: str, new_column_name: str = "year") -> DataFrame:
    """
    Adds a new column to the DataFrame with the year extracted from a date column.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the date column.
    date_column : str
        The name of the date column from which the year is extracted.
    new_column_name : str, optional
        The name of the new column containing the year. Defaults to "year".

    Returns
    -------
    DataFrame
        DataFrame with a new column containing the extracted year.

    Examples
    --------
    >>> df = spark.createDataFrame([("2023-09-28",), ("2024-01-01",)], ["date_col"])
    >>> df_with_year = add_year_column(df, "date_col")
    >>> df_with_year.show(truncate=False)
    +----------+----+
    |date_col  |year|
    +----------+----+
    |2023-09-28|2023|
    |2024-01-01|2024|
    +----------+----+
    """
    return df.withColumn(new_column_name, year(col(date_column)))


def add_quarter_column(df: DataFrame, date_column: str, new_column_name: str = "quarter") -> DataFrame:
    """
    Adds a new column to the DataFrame with the quarter of the year extracted from a date column.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the date column.
    date_column : str
        The name of the date column from which the quarter is extracted.
    new_column_name : str, optional
        The name of the new column containing the quarter. Defaults to "quarter".

    Returns
    -------
    DataFrame
        DataFrame with a new column containing the extracted quarter.

    Examples
    --------
    >>> df = spark.createDataFrame([("2023-09-28",), ("2024-01-01",)], ["date_col"])
    >>> df_with_quarter = add_quarter_column(df, "date_col")
    >>> df_with_quarter.show(truncate=False)
    +----------+-------+
    |date_col  |quarter|
    +----------+-------+
    |2023-09-28|3      |
    |2024-01-01|1      |
    +----------+-------+
    """
    return df.withColumn(new_column_name, quarter(col(date_column)))




def add_quarter_year_column(df: DataFrame, date_column: str, new_column_name: str = "quarter_year") -> DataFrame:
    """
    Adds a new column to the DataFrame that combines the quarter and year extracted from a date column.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the date column.
    date_column : str
        The name of the date column from which the quarter and year are extracted.
    new_column_name : str, optional
        The name of the new column containing the combined quarter and year. Defaults to "quarter_year".

    Returns
    -------
    DataFrame
        DataFrame with a new column containing the combined quarter and year.

    Examples
    --------
    >>> df = spark.createDataFrame([("2023-09-28",), ("2024-01-01",)], ["date_col"])
    >>> df_with_quarter_year = add_quarter_year_column(df, "date_col")
    >>> df_with_quarter_year.show(truncate=False)
    +----------+------------+
    |date_col  |quarter_year|
    +----------+------------+
    |2023-09-28|Q3 2023     |
    |2024-01-01|Q1 2024     |
    +----------+------------+
    """
    return df.withColumn(new_column_name, concat(lit("Q"), quarter(col(date_column)), lit(" "), year(col(date_column))))


-----------

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


DateFeature = Literal[
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "weekofyear",
    "quarter",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_year_start",
    "is_year_end",
]

TimeFeature = Literal[
    "hour",
    "minute",
    "second",
    "am_pm",
    "is_morning",
    "is_afternoon",
    "is_evening",
    "is_night",
    "day_period",
]


def add_date_feature(
    df: DataFrame,
    date_col: str,
    date_feature: DateFeature,
    output_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Extract a specific date feature from a date column and add it as a new column.

    This function adds a new column to the DataFrame, representing a specific
    feature extracted from the specified date column.

    Args:
        df (DataFrame): Input DataFrame.
        date_col (str): Name of the column containing the date.
        date_feature (DateFeature): Feature to extract. Available features are:
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear',
            'quarter', 'is_weekend', 'is_month_start', 'is_month_end',
            'is_year_start', 'is_year_end'
        output_col_name (Optional[str]): Name of the new column for the extracted feature.
            If None, defaults to the name of the date_feature.

    Returns:
        DataFrame: DataFrame with an additional column for the extracted date feature.

    Raises:
        ValueError: If the specified date_col is not present in the DataFrame.

    Example:
        >>> df = spark.createDataFrame([
        ...     ('2023-01-01',),
        ...     ('2023-06-15',),
        ...     ('2023-12-31',)
        ... ], ['date'])
        >>> result = add_date_feature(df, 'date', 'is_weekend', 'is_weekend_flag')
        >>> result.show()
        +----------+---------------+
        |      date|is_weekend_flag|
        +----------+---------------+
        |2023-01-01|           true|
        |2023-06-15|          false|
        |2023-12-31|           true|
        +----------+---------------+
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in the DataFrame.")

    if output_col_name is None:
        output_col_name = date_feature

    date_features = {
        "year": year(col(date_col)),
        "month": month(col(date_col)),
        "day": dayofmonth(col(date_col)),
        "dayofweek": dayofweek(col(date_col)),
        "dayofyear": dayofyear(col(date_col)),
        "weekofyear": weekofyear(col(date_col)),
        "quarter": quarter(col(date_col)),
        "is_weekend": (dayofweek(col(date_col)).isin([1, 7])).cast("boolean"),
        "is_month_start": (dayofmonth(col(date_col)) == lit(1)).cast("boolean"),
        "is_month_end": (last_day(col(date_col)) == col(date_col)).cast("boolean"),
        "is_year_start": (
            (month(col(date_col)) == lit(1)) & (dayofmonth(col(date_col)) == lit(1))
        ).cast("boolean"),
        "is_year_end": (
            (month(col(date_col)) == lit(12)) & (dayofmonth(col(date_col)) == lit(31))
        ).cast("boolean"),
    }

    return df.withColumn(output_col_name, date_features[date_feature])


def add_time_feature(
    df: DataFrame,
    timestamp_col: str,
    time_feature: TimeFeature,
    output_col_name: Optional[str] = None,
) -> DataFrame:
    """
    Extract a specific time feature from a timestamp column and add it as a new column.

    This function adds a new column to the DataFrame, representing a specific
    time feature extracted from the specified timestamp column.

    Args:
        df (DataFrame): Input DataFrame.
        timestamp_col (str): Name of the column containing the timestamp.
        time_feature (TimeFeature): Feature to extract. Available features are:
            'hour', 'minute', 'second', 'am_pm', 'is_morning', 'is_afternoon',
            'is_evening', 'is_night', 'day_period'
        output_col_name (Optional[str]): Name of the new column for the extracted feature.
            If None, defaults to the name of the time_feature.

    Returns:
        DataFrame: DataFrame with an additional column for the extracted time feature.

    Raises:
        ValueError: If the specified timestamp_col is not present in the DataFrame.

    Example:
        >>> df = spark.createDataFrame([
        ...     ('2023-01-01 09:30:00',),
        ...     ('2023-06-15 14:45:30',),
        ...     ('2023-12-31 23:59:59',)
        ... ], ['timestamp'])
        >>> result = add_time_feature(df, 'timestamp', 'am_pm', 'period')
        >>> result.show()
        +-------------------+------+
        |          timestamp|period|
        +-------------------+------+
        |2023-01-01 09:30:00|    AM|
        |2023-06-15 14:45:30|    PM|
        |2023-12-31 23:59:59|    PM|
        +-------------------+------+
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in the DataFrame.")

    if output_col_name is None:
        output_col_name = time_feature

    time_features = {
        "hour": hour(col(timestamp_col)),
        "minute": minute(col(timestamp_col)),
        "second": second(col(timestamp_col)),
        "am_pm": date_format(col(timestamp_col), "a"),
        "is_morning": (hour(col(timestamp_col)).between(6, 11)).cast("boolean"),
        "is_afternoon": (hour(col(timestamp_col)).between(12, 17)).cast("boolean"),
        "is_evening": (hour(col(timestamp_col)).between(18, 23)).cast("boolean"),
        "is_night": (
            (hour(col(timestamp_col)) >= 0) & (hour(col(timestamp_col)) < 6)
        ).cast("boolean"),
        "day_period": (
            when(hour(col(timestamp_col)).between(6, 11), "Morning")
            .when(hour(col(timestamp_col)).between(12, 17), "Afternoon")
            .when(hour(col(timestamp_col)).between(18, 23), "Evening")
            .otherwise("Night")
        ),
    }

    return df.withColumn(output_col_name, time_features[time_feature])


