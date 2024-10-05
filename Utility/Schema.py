from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when, sum as f_sum
from typing import List, Tuple, Dict, Optional, Union
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    FloatType,
    BooleanType,
    DateType,
    TimestampType,
)


def define_dataframe_schema(
    schema_dict: Dict[str, Union[str, Dict[str, Union[str, bool]]]]
) -> StructType:
    """
    Define a schema for a PySpark DataFrame based on a dictionary specification.

    This function creates a StructType schema for a PySpark DataFrame using a dictionary
    that specifies column names, data types, and optional attributes like nullability.

    Args:
        schema_dict (Dict[str, Union[str, Dict[str, Union[str, bool]]]]): A dictionary where keys are column names
            and values are either string data types or dictionaries with 'type' and optional 'nullable' keys.

    Returns:
        StructType: A PySpark StructType schema definition.

    Raises:
        ValueError: If an unsupported data type is provided.

    Example:
        >>> schema_dict = {
        ...     "id": "integer",
        ...     "name": {"type": "string", "nullable": False},
        ...     "age": "integer",
        ...     "height": "float",
        ...     "is_student": "boolean",
        ...     "enrollment_date": "date",
        ...     "last_login": "timestamp"
        ... }
        >>> schema = define_dataframe_schema(schema_dict)
        >>> print(schema)
        StructType([
            StructField('id', IntegerType(), True),
            StructField('name', StringType(), False),
            StructField('age', IntegerType(), True),
            StructField('height', FloatType(), True),
            StructField('is_student', BooleanType(), True),
            StructField('enrollment_date', DateType(), True),
            StructField('last_login', TimestampType(), True)
        ])
    """

    def _get_spark_type(
        type_str: str,
    ) -> Union[
        StringType, IntegerType, FloatType, BooleanType, DateType, TimestampType
    ]:
        type_mapping = {
            "string": StringType(),
            "integer": IntegerType(),
            "float": FloatType(),
            "boolean": BooleanType(),
            "date": DateType(),
            "timestamp": TimestampType(),
        }
        if type_str.lower() not in type_mapping:
            raise ValueError(f"Unsupported data type: {type_str}")
        return type_mapping[type_str.lower()]

    schema_fields: List[StructField] = []

    for column_name, column_spec in schema_dict.items():
        if isinstance(column_spec, str):
            field = StructField(column_name, _get_spark_type(column_spec), True)
        elif isinstance(column_spec, dict):
            data_type = _get_spark_type(column_spec["type"])
            nullable = column_spec.get("nullable", True)
            field = StructField(column_name, data_type, nullable)
        else:
            raise ValueError(f"Invalid schema specification for column: {column_name}")

        schema_fields.append(field)

    return StructType(schema_fields)


def enforce_schema(df: DataFrame, expected_schema: StructType) -> DataFrame:
    """
    Compare the DataFrame's schema with an expected schema and enforce it.
    Allows for None values in nullable columns.

    Args:
        df (DataFrame): Input DataFrame
        expected_schema (StructType): Expected schema to enforce

    Returns:
        DataFrame: DataFrame with enforced schema

    Raises:
        ValueError: If a non-nullable column cannot be cast to the expected data type

    Example:
        from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, BooleanType, DateType

        # Create a sample DataFrame
        data = [
            (1, "Alice", 30, 5000.50, True, "2023-01-01"),
            (2, "Bob", None, 6000.75, False, "2023-02-15"),
            (3, "Charlie", 35, None, True, "2023-03-30")
        ]
        df = spark.createDataFrame(data, ["id", "name", "age", "salary", "is_active", "join_date"])

        # Define the expected schema
        expected_schema = StructType([
            StructField("id", IntegerType(), nullable=False),
            StructField("name", StringType(), nullable=True),
            StructField("age", IntegerType(), nullable=True),
            StructField("salary", DoubleType(), nullable=True),
            StructField("is_active", BooleanType(), nullable=True),
            StructField("join_date", DateType(), nullable=True),
            StructField("department", StringType(), nullable=True)  # New column
        ])

        # Apply schema enforcement
        result_df = df.transform(enforce_schema, expected_schema)

        # Show the result
        result_df.show()
        result_df.printSchema()
    """
    current_columns = set(df.columns)
    expected_columns = set(field.name for field in expected_schema.fields)

    # Add missing columns
    missing_columns = expected_columns - current_columns
    if missing_columns:
        df = df.select(
            "*",
            *[
                lit(None).cast(expected_schema[col].dataType).alias(col)
                for col in missing_columns
            ],
        )

    # Remove extra columns
    df = df.select(*[col for col in df.columns if col in expected_columns])

    # Enforce data types and check for null values
    cast_exprs: List[Tuple[str, str]] = []
    for field in expected_schema.fields:
        cast_expr = (
            f"CAST(`{field.name}` AS {field.dataType.simpleString()}) AS `{field.name}`"
        )
        cast_exprs.append((field.name, cast_expr))

    try:
        df = df.selectExpr(*[expr for _, expr in cast_exprs])
    except AnalysisException as e:
        raise ValueError(f"Schema enforcement failed: {str(e)}")

    # Check for null values introduced by casting in non-nullable columns
    null_counts: Dict[str, int] = {}
    for field in expected_schema.fields:
        if not field.nullable:
            null_count = df.filter(col(field.name).isNull()).count()
            if null_count > 0:
                null_counts[field.name] = null_count

    if null_counts:
        error_msg = "Casting introduced null values in non-nullable columns:\n"
        for col_name, count in null_counts.items():
            error_msg += f"  - {col_name}: {count} null value(s)\n"
        raise ValueError(error_msg)

    # Reorder columns to match expected schema
    return df.select(*[col(field.name) for field in expected_schema.fields])


# Example usage:
# expected_schema = StructType([
#     StructField("id", IntegerType(), nullable=False),
#     StructField("name", StringType(), nullable=True),
#     StructField("age", IntegerType(), nullable=True)
# ])
# result_df = df.transform(enforce_schema, expected_schema)


def assert_unique_combination(df: DataFrame, columns: List[str]) -> None:
    """
    Assert that specified columns or their combination have unique values in the DataFrame.

    This function checks for uniqueness in the specified columns. If a single column
    is provided, it checks that column for uniqueness. If multiple columns are provided,
    it checks the combination of these columns for uniqueness.

    Args:
        df (DataFrame): Input DataFrame to check.
        columns (List[str]): List of column names to check for uniqueness.

    Raises:
        ValueError: If duplicate values are found in the specified column(s).

    Example:
        >>> df = spark.createDataFrame([(1, 'A'), (2, 'B'), (1, 'C')], ['id', 'value'])
        >>> assert_unique_combination(df, ['id'])  # Raises ValueError
        >>> assert_unique_combination(df, ['id', 'value'])  # No error
    """
    if not columns:
        raise ValueError("At least one column must be specified.")

    if len(columns) == 1:
        check_column = columns[0]
    else:
        check_column = "unique_combo_" + "_".join(columns)
        concat_expr = F.concat_ws("||", *columns)
        df = df.withColumn(check_column, concat_expr)

    duplicates = df.groupBy(check_column).count().filter(F.col("count") > 1)

    if duplicates.count() > 0:
        dup_values = duplicates.select(check_column).limit(3).collect()
        dup_list = [row[check_column] for row in dup_values]

        if len(columns) == 1:
            error_message = f"🚨 Column {columns[0]} has duplicate values."
        else:
            error_message = f"🚨 Combination of {', '.join(columns)} is not unique."

        error_message += f"\n💡 Examples: {', '.join(map(str, dup_list))}"
        raise ValueError(error_message)

    if len(columns) > 1:
        df = df.drop(check_column)

    print(f"✅ Uniqueness check passed for: {', '.join(columns)}")


def assert_dataframe_equality(df1: DataFrame, df2: DataFrame) -> None:
    """
    Assert that two PySpark DataFrames have equal content, optimized for speed.

    This function efficiently compares two DataFrames and asserts that they have the same content.
    It assumes that both DataFrames have the same column names.

    Args:
        df1 (DataFrame): The first DataFrame to compare
        df2 (DataFrame): The second DataFrame to compare

    Raises:
        AssertionError: If the DataFrames are not equal, with catchy details about the differences

    Example:
        >>> df1 = spark.createDataFrame([(1, 'a'), (2, 'b')], ['id', 'value'])
        >>> df2 = spark.createDataFrame([(1, 'a'), (2, 'b')], ['id', 'value'])
        >>> assert_dataframe_equality(df1, df2)  # This will pass
        >>>
        >>> df3 = spark.createDataFrame([(1, 'a'), (3, 'c')], ['id', 'value'])
        >>> assert_dataframe_equality(df1, df3)  # This will raise an AssertionError
    """

    # Check if both inputs are DataFrames
    if not isinstance(df1, DataFrame) or not isinstance(df2, DataFrame):
        raise AssertionError(
            "🚫 Oops! Both inputs must be PySpark DataFrames. Let's stick to the script!"
        )

    # Check if the column names are the same
    if df1.columns != df2.columns:
        raise AssertionError(
            f"🔤 Column name mismatch! We've got:\n{df1.columns}\nvs\n{df2.columns}"
        )

    try:
        # Efficiently compare DataFrames
        df_combined = df1.selectExpr("*", "1 as df1_marker").unionAll(
            df2.selectExpr("*", "2 as df1_marker")
        )

        diff_stats = (
            df_combined.groupBy(*df1.columns)
            .agg(
                count(when(col("df1_marker") == 1, True)).alias("count_df1"),
                count(when(col("df1_marker") == 2, True)).alias("count_df2"),
            )
            .where("count_df1 != count_df2")
        )

        diff_count = diff_stats.count()

        if diff_count > 0:
            sample_diff = diff_stats.limit(5).collect()

            error_message = f"📊 Found {diff_count} mismatched rows.\n"
            error_message += "🔍 Here's a sneak peek at the differences:\n"
            for row in sample_diff:
                error_message += f"   {row.asDict()}\n"
            error_message += "💡 Tip: Check your data sources or transformations!"

            raise AssertionError(error_message)

        # If we've made it this far, the DataFrames are equal
        print("🎉 Jackpot! The DataFrames are identical twins.")

    except Exception as e:
        if isinstance(e, AssertionError):
            raise e
        else:
            raise AssertionError(
                f"❌ An unexpected error occurred while comparing DataFrames: {str(e)}"
            )


def find_duplicates(df, subset=None):
    """
    Find duplicate rows in a PySpark DataFrame.

    Args:
    df (pyspark.sql.DataFrame): The input DataFrame.
    subset (list, optional): List of columns to consider for identifying duplicates.
                             If None, all columns are considered. Default is None.

    Returns:
    pyspark.sql.DataFrame: A DataFrame containing only the duplicate rows.
    """
    # If subset is not provided, use all columns
    if subset is None:
        subset = df.columns

    # Define a window specification that partitions by the specified columns
    window_spec = Window.partitionBy(subset)

    # Add a count column using the window function
    df_with_count = df.withColumn("count", count("*").over(window_spec))

    # Filter the rows where the count is greater than 1
    duplicates = df_with_count.filter(col("count") > 1).drop("count")

    return duplicates


# 🏗 helper function to compare_dataframe_columns
def get_unique_values(df: DataFrame, column: str) -> List:
    """
    Get unique values from a specific column in a DataFrame.

    Args:
        df (DataFrame): Input DataFrame
        column (str): Column name to get unique values from

    Returns:
        List: List of unique values
    """
    return [row[column] for row in df.select(column).distinct().collect()]


# 🏗 helper function to compare_dataframe_columns
def find_missing_values(df1_values: List, df2_values: List) -> List:
    """
    Find values that are in df2 but not in df1.

    Args:
        df1_values (List): List of values from first DataFrame
        df2_values (List): List of values from second DataFrame

    Returns:
        List: List of missing values
    """
    return list(set(df2_values) - set(df1_values))


# 🏗 helper function to compare_dataframe_columns
def format_output_message(missing_values: List) -> Tuple[bool, str]:
    """
    Format the output message based on missing values.

    Args:
        missing_values (List): List of missing values

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating test pass/fail and the formatted message
    """
    if not missing_values:
        return (
            True,
            "🎉 Great news! All values from the second DataFrame are present in the first. Test passed! 🎉",
        )
    else:
        examples = missing_values[:5]
        return False, (
            f"❌ Oops! We found some values in the second DataFrame that are missing from the first. "
            f"Here are up to 5 examples: {examples} ❌"
        )


def compare_dataframe_columns(
    df1: DataFrame, df2: DataFrame, col1: str, col2: str
) -> Tuple[bool, str]:
    """
    Compare column values between two DataFrames to check if all values from the second DataFrame's column
    are present in the first DataFrame's column.

    This function is useful for data validation, ensuring data completeness, or checking for discrepancies
    between two datasets. It compares unique values in the specified columns of both DataFrames.

    Args:
        df1 (DataFrame): The first PySpark DataFrame. This is considered the "master" or "complete" dataset.
        df2 (DataFrame): The second PySpark DataFrame. This is the dataset being checked against df1.
        col1 (str): The name of the column in df1 to compare. Must exist in df1.
        col2 (str): The name of the column in df2 to compare. Must exist in df2.

    Returns:
        Tuple[bool, str]: A tuple containing two elements:
            - bool: True if all values in df2[col2] are present in df1[col1], False otherwise.
            - str: A formatted message describing the result of the comparison.
              If successful, it returns a celebratory message.
              If there are missing values, it returns an error message with up to 5 examples of missing values.

    Raises:
        ValueError: If either col1 is not in df1 or col2 is not in df2.

    Example:
        >>> # Create sample DataFrames
        >>> df1 = spark.createDataFrame([(1, "A"), (2, "B"), (3, "C")], ["id", "value"])
        >>> df2 = spark.createDataFrame([(1, "A"), (2, "B"), (4, "D")], ["id", "value"])
        >>>
        >>> # Compare 'value' columns
        >>> result, message = compare_dataframe_columns(df1, df2, "value", "value")
        >>> print(f"Comparison result: {result}")
        >>> print(f"Message: {message}")
        Comparison result: False
        Message: ❌ Oops! We found some values in the second DataFrame that are missing from the first. Here are up to 5 examples: ['D'] ❌

    Note:
        - This function only compares the presence of values, not their frequency or order.
        - The comparison is case-sensitive for string values.
        - The function assumes that the columns contain comparable data types.
        - Large DataFrames may impact performance, as the function collects all unique values to the driver.

    See Also:
        get_unique_values: Helper function to extract unique values from a DataFrame column.
        find_missing_values: Helper function to identify values present in one list but not another.
        format_output_message: Helper function to create user-friendly output messages.
    """
    if col1 not in df1.columns:
        raise ValueError(f"Column '{col1}' not found in the first DataFrame.")
    if col2 not in df2.columns:
        raise ValueError(f"Column '{col2}' not found in the second DataFrame.")

    df1_values = get_unique_values(df1, col1)
    df2_values = get_unique_values(df2, col2)
    missing_values = find_missing_values(df1_values, df2_values)
    return format_output_message(missing_values)
