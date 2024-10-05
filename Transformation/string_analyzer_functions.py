from pyspark.ml.feature import StopWordsRemover, Tokenizer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    array_join,
    col,
    length,
    regexp_replace,
    size,
    split,
    trim,
    udf,
    when,
)
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession.builder.getOrCreate()


def remove_stopwords_numbers_special_chars(df, input_col: str, output_col: str):
    """
    Removes German stop words, numbers, and special characters from a specified string column in a PySpark DataFrame.

    Parameters:
    - df: PySpark DataFrame containing the data.
    - input_col: Name of the input column containing strings.
    - output_col: Name of the output column to store cleaned strings.

    Returns:
    - A new DataFrame with the cleaned strings in the specified output column.
    """

    # Step 1: Replace null values with empty strings and trim leading/trailing whitespace
    df = df.withColumn(
        input_col, when(col(input_col).isNull(), "").otherwise(trim(col(input_col)))
    )

    # Step 2: Remove special characters and numbers from the text, keeping only letters
    df = df.withColumn(
        input_col, regexp_replace(col(input_col), r"[^A-Za-zÄÖÜäöüß\s]", "")
    )

    # Step 3: Tokenize the text
    tokenizer = Tokenizer(inputCol=input_col, outputCol="tokens")
    df = tokenizer.transform(df)

    # Step 4: Load German stop words
    german_stop_words = StopWordsRemover.loadDefaultStopWords("german")
    # Create broadcast variable
    sc = df.sql_ctx._sc
    broadcast_stopwords = sc.broadcast(set(german_stop_words))

    # Step 5: Define UDF to remove stop words
    def remove_stopwords(tokens):
        return [
            token for token in tokens if token.lower() not in broadcast_stopwords.value
        ]

    remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))
    df = df.withColumn("filtered_tokens", remove_stopwords_udf(col("tokens")))

    # Step 6: Join the cleaned tokens back into a single string
    df = df.withColumn(output_col, array_join(col("filtered_tokens"), " "))

    # Step 7: Drop intermediate columns
    df = df.drop("tokens", "filtered_tokens")

    return df


def add_string_length_column(
    df: DataFrame, string_column: str, new_column: str
) -> DataFrame:
    """
    Adds a new column to the DataFrame containing the length of the strings
    in the specified string column.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.
    string_column : str
        The name of the column containing the strings.
    new_column : str
        The name of the new column that will contain the length of the strings.

    Returns
    -------
    DataFrame
        A new DataFrame with the added string length column.

    Raises
    ------
    ValueError
        If the specified string column does not exist in the DataFrame.
    """
    if string_column not in df.columns:
        raise ValueError(f"Column '{string_column}' does not exist in the DataFrame.")

    # Add the length column
    df_with_length = df.withColumn(new_column, length(col(string_column)))

    return df_with_length


def add_word_count_column(
    df: DataFrame, string_column: str, new_column: str
) -> DataFrame:
    """
    Adds a new column to the DataFrame containing the word count of the strings
    in the specified string column, accounting for empty strings and None values.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame.
    string_column : str
        The name of the column containing the strings.
    new_column : str
        The name of the new column that will contain the word count.

    Returns
    -------
    DataFrame
        A new DataFrame with the added word count column.

    Raises
    ------
    ValueError
        If the specified string column does not exist in the DataFrame.
    """
    if string_column not in df.columns:
        raise ValueError(f"Column '{string_column}' does not exist in the DataFrame.")

    # Handle None and empty strings by returning a word count of 0
    df_with_word_count = df.withColumn(
        new_column,
        when(
            col(string_column).isNull() | (col(string_column) == ""), 0
        ).otherwise(  # Handle None and ""
            size(split(col(string_column), " "))
        ),  # Count words for non-empty strings
    )

    return df_with_word_count
