import io
import logging
import os
from typing import List, Optional

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from pyspark.sql import functions as F


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
