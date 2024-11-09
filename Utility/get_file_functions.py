import io
import logging
import os
from typing import Any, Dict, List, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def _list_csv_files(
    folder_path: str, recursive: bool, file_extension: str
) -> List[str]:
    """
    List all CSV files in a specified directory, with an option for recursive search.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing files.
    recursive : bool
        If True, search for files recursively in subfolders.
    file_extension : str
        The file extension to filter by (e.g., "csv").

    Returns
    -------
    List[str]
        A list of file paths with the specified extension.
    """
    try:
        file_infos = dbutils.fs.ls(folder_path)
        if recursive:
            file_infos = dbutils.fs.ls(folder_path, recursive=True)
        return [
            file_info.path
            for file_info in file_infos
            if file_info.path.endswith(f".{file_extension}") and not file_info.isDir()
        ]
    except Exception as e:
        logging.error(f"Error listing files in folder '{folder_path}': {e}")
        return []


def _read_csv_file(
    file_path: str, options: Dict[str, Any], columns: Optional[List[str]] = None
) -> DataFrame:
    """
    Read a single CSV file into a DataFrame, select specified columns, and add the file name as a column.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    options : Dict[str, Any]
        Options for reading CSV.
    columns : Optional[List[str]]
        List of columns to select. If None, all columns are selected.

    Returns
    -------
    DataFrame
        The read DataFrame with selected columns and an additional 'source_file' column.
    """
    df = spark.read.options(**options).csv(file_path)
    if columns:
        df = df.select(*columns)
    return df.withColumn("source_file", F.lit(os.path.basename(file_path)))


def _combine_dataframes(dataframes: List[DataFrame]) -> DataFrame:
    """
    Combine multiple DataFrames using unionByName.

    Parameters
    ----------
    dataframes : List[DataFrame]
        A list of DataFrames to combine.

    Returns
    -------
    DataFrame
        A single DataFrame resulting from the union of all input DataFrames.
    """
    if not dataframes:
        raise ValueError("No DataFrames to combine.")
    combined_df = dataframes[0]
    for df in dataframes[1:]:
        combined_df = combined_df.unionByName(df, allowMissingColumns=True)
    return combined_df


def get_combined_csv_dataframe(
    folder_path: str,
    columns: Optional[List[str]] = None,
    recursive: bool = False,
    file_extension: str = "csv",
    **kwargs,
) -> DataFrame:
    """
    Retrieve and combine CSV files from a specified folder into a single DataFrame.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing CSV files.
    columns : Optional[List[str]], optional
        List of columns to select from each file. If None, all columns are selected.
    recursive : bool, optional
        If True, searches for files recursively in subfolders. Defaults to False.
    file_extension : str, optional
        The file extension to filter by. Defaults to "csv".
    **kwargs
        Additional keyword arguments to pass to spark.read.csv().

    Returns
    -------
    DataFrame
        A DataFrame containing the combined data from all CSV files, with selected columns
        and an additional 'source_file' column.

    Raises
    ------
    ValueError
        If no files with the specified extension are found in the given path.
    """
    logging.info(f"Reading CSV files from: {folder_path}")

    # Set default options and update with additional kwargs
    options = {
        "sep": ";",
        "header": "true",
        "ignoreLeadingWhiteSpace": "true",
        "ignoreTrailingWhiteSpace": "true",
        "encoding": "UTF-8",
    }
    options.update(kwargs)

    # List and load CSV files
    csv_files = _list_csv_files(folder_path, recursive, file_extension)
    if not csv_files:
        raise ValueError(
            f"No {file_extension.upper()} files found in the specified path."
        )

    # Read and combine DataFrames
    dataframes = [_read_csv_file(file, options, columns) for file in csv_files]
    return _combine_dataframes(dataframes)


# Function to list Excel files
def _list_excel_files(
    folder_path: str, recursive: bool, file_extension: str
) -> List[str]:
    """
    List all Excel files in the specified folder using dbutils.
    """
    try:
        all_files = []
        paths = [folder_path]
        while paths:
            current_path = paths.pop()
            files = dbutils.fs.ls(current_path)
            for f in files:
                if f.isDir() and recursive:
                    paths.append(f.path)
                elif f.path.endswith(f".{file_extension}"):
                    all_files.append(f.path)
        if not all_files:
            raise ValueError(
                f"No .{file_extension} files found in the specified folder: {folder_path}"
            )
        return all_files
    except Exception as e:
        logging.error(f"Error listing Excel files: {str(e)}")
        raise


# Function to read Excel files and convert to Spark DataFrames
def _read_excel_to_spark_df(
    file_path: str, columns: Optional[List[str]] = None
) -> Optional[DataFrame]:
    """
    Read an Excel file from a given path and convert it to a Spark DataFrame.
    """
    try:
        # Read the binary content of the file using spark.read.format("binaryFile")
        binary_df = spark.read.format("binaryFile").load(file_path)
        # Collect the binary content
        binary_content = binary_df.select("content").collect()[0][0]
        # Convert binary content to BytesIO object
        file_like_obj = io.BytesIO(binary_content)
        # Read Excel file using pandas
        pandas_df = pd.read_excel(
            file_like_obj,
            engine="openpyxl",  # Ensure compatibility with .xlsx files
            sheet_name=0,  # Read the first worksheet
            dtype=str,  # Read all columns as strings
        )

        if pandas_df.empty:
            return None  # Skip empty workbooks

        # Convert Pandas DataFrame to Spark DataFrame
        spark_df = spark.createDataFrame(pandas_df)

        if columns:
            spark_df = spark_df.select(*columns)

        file_name = os.path.basename(file_path)
        return spark_df.withColumn("source_file", F.lit(file_name))
    except Exception as e:
        logging.error(f"Error reading Excel file {file_path}: {str(e)}")
        return None  # Treat exceptions as empty workbooks


# Main function to combine DataFrames and handle empty workbooks
def get_combined_excel_dataframe(
    folder_path: str,
    columns: Optional[List[str]] = None,
    recursive: bool = False,
    file_extension: str = "xlsx",
) -> Optional[DataFrame]:
    """
    Retrieve and combine Excel files from a specified folder into a single DataFrame in Databricks.
    """
    logging.info(f"Reading Excel files from: {folder_path}")
    empty_workbooks = []
    dataframes = []
    try:
        excel_files = _list_excel_files(folder_path, recursive, file_extension)

        # Read all Excel files individually, selecting specified columns
        for file in excel_files:
            df = _read_excel_to_spark_df(file, columns)
            if df is None or df.rdd.isEmpty():
                empty_workbooks.append(file)
                continue
            dataframes.append(df)

        if not dataframes:
            logging.warning("No dataframes to combine.")
            return None

        # Combine all DataFrames using unionByName
        combined_df = dataframes[0]
        for df in dataframes[1:]:
            combined_df = combined_df.unionByName(df, allowMissingColumns=True)

        if empty_workbooks:
            print("Empty workbooks skipped:")
            for wb in empty_workbooks:
                print(wb)

        return combined_df

    except Exception as e:
        logging.error(f"Error in get_combined_excel_dataframe: {str(e)}")
        raise
