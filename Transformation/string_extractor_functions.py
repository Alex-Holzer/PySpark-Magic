"""
string_extractor_functions.py
===========================

This module provides utility functions for extracting and processing 
string data within a PySpark DataFrame. It focuses on operations like 
extracting numerical values from string columns, replacing specific patterns, 
and other string manipulation tasks.

Key functionalities:
- Extraction of numeric values from mixed string columns.
- Customizable string manipulation using regular expressions.
- Integration with PySpark's DataFrame API.

Dependencies:
- PySpark: Provides the DataFrame structure and necessary functions.
- re: Regular expression library for pattern matching.
- UDF (User Defined Functions): Used to create custom transformations for DataFrame columns.
  
Author: Alex Holzer
"""

import re
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    regexp_extract,
    regexp_replace,
    trim,
    udf,
    when,
)
from pyspark.sql.types import StringType, StructField, StructType

spark = SparkSession.builder.getOrCreate()


def extract_contract_number(
    df: DataFrame,
    column: str,
    running_number_length: int = 2,
    policy_number_length: int = 7,
) -> DataFrame:
    """
    Cleans and extracts the contract number from a given column in a PySpark DataFrame.

    The contract number will be formatted based on the provided running_number_length and policy_number_length.

    Args:
        df (DataFrame): The input PySpark DataFrame.
        column (str): The column name that contains the uncleaned contract numbers.
        running_number_length (int): Length of the running number (digits before the insurance division).
                                     Default is 2.
        policy_number_length (int): Length of the policy number (digits after the insurance division).
                                    Default is 7.

    Returns:
        DataFrame: The original DataFrame with an additional 'contract_number_cleaned' column
                   that contains the cleaned and formatted contract numbers.
    """

    # Clean the input by trimming spaces
    df = df.withColumn("clean_input", trim(col(column)))

    patterns = [
        (
            rf"^(\d{{1,{running_number_length + 1}}})\s*([A-Za-z]{{1,3}})(\d{{{policy_number_length}}})$",
            lambda m: (
                f"{m.group(1)[-running_number_length:]}{m.group(2).strip()}-{m.group(3)}"
                if len(m.group(2).strip()) > 1
                else f"{m.group(1)[-running_number_length:]} {m.group(2).strip()}-{m.group(3)}"
            ),
        ),
        # Pattern 2: policy_number_length digits + 2 or 3 letters + running_number_length digits
        (
            rf"^(\d{{{policy_number_length}}})([A-Za-z]{{2,3}})(\d{{{running_number_length}}})$",
            lambda m: f"{m.group(3)}{m.group(2)}-{m.group(1)}",
        ),
        (
            rf"^(\d{{{policy_number_length}}})-?(\d{{{running_number_length}}})([A-Za-z]{{1,3}})$",
            lambda m: (
                f"{m.group(2)} {m.group(3).strip()}-{m.group(1)}"
                if len(m.group(3).strip()) == 1
                else f"{m.group(2)}{m.group(3)}-{m.group(1)}"
            ),
        ),
    ]

    def transform_contract_number(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        # Standardize by removing any non-alphanumeric characters except spaces
        value = re.sub(r"[^A-Za-z0-9 ]", "", value)

        # Loop through the patterns and apply the first one that matches
        for pattern, formatter in patterns:
            match = re.match(pattern, value)
            if match:
                # Apply the formatter and return the cleaned value
                return formatter(match)

        # Return None if no pattern matches
        return None

    # Register the UDF
    transform_udf = udf(transform_contract_number, StringType())

    # Apply the UDF to create the 'contract_number_cleaned' column
    df = df.withColumn("contract_number_cleaned", transform_udf(col("clean_input")))

    # Drop the intermediate 'clean_input' column
    df = df.drop("clean_input")

    return df


def extract_policy_number(
    df: DataFrame, column: str, policy_number_length: int = 7
) -> DataFrame:
    """
    Extracts the policy number from the uncleaned contract number in the specified column.

    The function extracts the sequence of digits based on the given policy_number_length
    and stores the result in a new column called 'policy_number'.

    Args:
        df (DataFrame): Input PySpark DataFrame.
        column (str): Name of the column containing the uncleaned contract numbers.
        policy_number_length (int): Length of the policy number to extract (default: 7).

    Returns:
        DataFrame: The original DataFrame with an additional 'policy_number' column.
    """

    def extract_policy(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        # Standardize by removing any non-alphanumeric characters except spaces
        value = re.sub(r"[^A-Za-z0-9 ]", "", value)

        # Use a regular expression to find the policy number (sequence of digits of specified length)
        pattern = rf"(\d{{{policy_number_length}}})"
        match = re.search(
            pattern, value
        )  # Extract the specified number of consecutive digits
        if match:
            return match.group(1)

        return None

    # Register the UDF for policy number extraction
    extract_policy_udf = udf(extract_policy, StringType())

    # Apply the UDF to extract the policy number
    df = df.withColumn("policy_number", extract_policy_udf(trim(col(column))))

    return df


def get_numbers_from_string(df: DataFrame, column: str) -> DataFrame:
    """
    Extract all numbers from a single string column containing a mix of strings and integers.

    Args:
        df: Input PySpark DataFrame.
        column: Name of the column to process.

    Returns:
        DataFrame with an additional column containing all extracted numbers.

    Example:
        >>> df = spark.createDataFrame([("123ABC456",), ("789DEF012",)], ["mixed_column"])
        >>> result = get_numbers_from_string(df, "mixed_column")
        >>> result.show()
        +------------+------------------------+
        |mixed_column|mixed_column_only_numbers|
        +------------+------------------------+
        |   123ABC456|                   123456|
        |   789DEF012|                   789012|
        +------------+------------------------+
    """
    return df.withColumn(
        f"{column}_only_numbers",
        regexp_replace(col(column).cast(StringType()), r"[^0-9]", ""),
    )


def get_text_from_string(df: DataFrame, column: str) -> DataFrame:
    """
    Extract all alphabetic characters from a single string column
    containing a mix of strings and other characters.

    Args:
        df: Input PySpark DataFrame.
        column: Name of the column to process.

    Returns:
        DataFrame with an additional column containing all extracted alphabetic characters.

    Example:
        >>> df = spark.createDataFrame([("123ABC456",), ("DEF789GHI",)], ["mixed_column"])
        >>> result = get_text_from_string(df, "mixed_column")
        >>> result.show()
        +------------+----------------------+
        |mixed_column|mixed_column_only_text|
        +------------+----------------------+
        |   123ABC456|                   ABC|
        |   DEF789GHI|                DEFGHI|
        +------------+----------------------+
    """
    return df.withColumn(
        f"{column}_only_text",
        regexp_replace(col(column).cast(StringType()), r"[^a-zA-Z]", ""),
    )


def extract_email_from_column(
    df: DataFrame, column_name: str, output_column: str = "extracted_email"
) -> DataFrame:
    """
    Extract email addresses from a given string column in a PySpark DataFrame.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame containing the column with email addresses.
    column_name : str
        The name of the column from which to extract the email address.
    output_column : str, optional
        The name of the output column where the extracted emails will be stored.

    Returns
    -------
    DataFrame
        A new PySpark DataFrame with an additional column containing the extracted emails.

    Example
    -------
    df = extract_email_from_column(input_df, 'text_column', 'email_column')
    """
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    return df.withColumn(
        output_column, regexp_extract(col(column_name), email_pattern, 0)
    )


def extract_street_and_house_number(df, address_column):
    """
    Extracts German street names and house numbers from an address column in a PySpark DataFrame.
    Adds two new columns: 'street_name' and 'house_number'.

    Parameters:
    df (DataFrame): Input PySpark DataFrame.
    address_column (str): Name of the column containing the address strings.

    Returns:
    DataFrame: PySpark DataFrame with additional 'street_name' and 'house_number' columns.
    """

    # Regular expression pattern to match house number at the end of the string
    house_number_pattern = (
        r"(\d+(\s*[a-zA-Z]+)?(\s*[-/]{1,2}\s*\d*(\s*[a-zA-Z]+)?)*\s*)$"
    )

    # Extract house_number using regex
    df = df.withColumn(
        "house_number",
        trim(regexp_extract(col(address_column), house_number_pattern, 0)),
    )

    # Remove house_number from address_column to get street_name
    df = df.withColumn(
        "street_name",
        trim(regexp_replace(col(address_column), house_number_pattern, "")),
    )

    return df


def split_full_name(df, full_name_column):
    """
    Splits the full_name column in a DataFrame into title, first_name, and last_name columns.
    """
    # List of possible German titles
    raw_titles = [
        "Dr.",
        "Prof.",
        "Dipl.-Ing.",
        "Dipl.-Kfm.",
        "Dipl.",
        "Mag.",
        "Ing.",
        "B.Sc.",
        "M.Sc.",
        "Ph.D.",
        "Univ.-Prof.",
        "Priv.-Doz.",
        "Univ.-Doz.",
        "Dr.-Ing.",
    ]
    # Normalize titles for case-insensitive matching and handle variations with or without periods
    normalized_titles = set()
    for t in raw_titles:
        t_norm = t.replace(".", "").lower()
        normalized_titles.add(t_norm)

    # List of German and common European prepositions or particles in surnames
    prepositions_list = [
        ["von", "der"],
        ["van", "der"],
        ["von", "dem"],
        ["van", "den"],
        ["de", "la"],
        ["de", "le"],
        ["de", "los"],
        ["de", "las"],
        ["von"],
        ["van"],
        ["zu"],
        ["zum"],
        ["zur"],
        ["vom"],
        ["de"],
        ["del"],
        ["da"],
        ["di"],
        ["der"],
        ["den"],
        ["du"],
        ["la"],
        ["le"],
    ]
    # Sort prepositions_list by length in decreasing order
    prepositions_list.sort(key=lambda x: -len(x))

    # Normalize prepositions to lowercase for comparison
    prepositions_list = [
        [word.lower() for word in preposition] for preposition in prepositions_list
    ]

    # Define the UDF to split the full name
    def split_name(full_name):
        if not full_name:
            return ("", "", "")
        words = full_name.strip().split()
        title_words = []
        i = 0
        # Extract titles from the beginning of the name
        while i < len(words):
            word = words[i]
            word_norm = word.replace(".", "").lower()
            if word_norm in normalized_titles:
                title_words.append(word)
                i += 1
            else:
                break
        title = " ".join(title_words)
        # Remaining words after extracting the title
        name_words = words[i:]
        if not name_words:
            return (title, "", "")
        # Build last_name_words starting from the last word
        last_name_words = [name_words[-1]]
        j = len(name_words) - 2
        while j >= 0:
            match = False
            for preposition in prepositions_list:
                preposition_length = len(preposition)
                if j - preposition_length + 1 >= 0:
                    candidate_words = name_words[j - preposition_length + 1 : j + 1]
                    candidate_words_lower = [w.lower() for w in candidate_words]
                    if candidate_words_lower == preposition:
                        last_name_words = candidate_words + last_name_words
                        j = j - preposition_length
                        match = True
                        break
            if not match:
                break
        first_name_words = name_words[: j + 1]
        first_name = " ".join(first_name_words)
        last_name = " ".join(last_name_words)
        return (title, first_name, last_name)

    # Define the UDF with the appropriate return type
    split_name_udf = udf(
        split_name,
        StructType(
            [
                StructField("title", StringType(), True),
                StructField("first_name", StringType(), True),
                StructField("last_name", StringType(), True),
            ]
        ),
    )

    # Apply the UDF to the DataFrame
    df = df.withColumn("name_struct", split_name_udf(col(full_name_column)))
    df = df.withColumn("title", col("name_struct.title"))
    df = df.withColumn("first_name", col("name_struct.first_name"))
    df = df.withColumn("last_name", col("name_struct.last_name"))
    df = df.drop("name_struct")

    return df


def extract_form_of_address(df: DataFrame, column_name: str) -> DataFrame:
    """
    Normalizes the form of address in a given column of a PySpark DataFrame to either 'Herr' or 'Frau'.

    Parameters
    ----------
    df : DataFrame
        The input PySpark DataFrame containing the form of address column.
    column_name : str
        The name of the column with form of address values.

    Returns
    -------
    DataFrame
        The transformed DataFrame with normalized forms of address.
    """
    return df.withColumn(
        column_name,
        when(col(column_name).startswith("H"), lit("Herr"))
        .when(col(column_name).startswith("F"), lit("Frau"))
        .otherwise(col(column_name)),
    )
