import fnmatch
import logging
from typing import Any, Dict, List, Literal, Optional, Union


def validate_path(
    path: Optional[
        str
    ] = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW/example.csv",
) -> bool:
    """
    Validate if the provided path exists in the Azure Data Lake Storage.

    Parameters
    ----------
    path : str, optional
        The file path to validate. The default is the path to a specific CSV file in Azure Data Lake Storage.

    Returns
    -------
    bool
        True if the path exists, False otherwise.
    """
    try:
        # Check if the path exists using dbutils.fs.ls()
        dbutils.fs.ls(path)
        return True
    except Exception as e:
        # If the path does not exist or any other error occurs, return False
        return False


def list_files_in_directory(folder_path: str) -> list[str]:
    """
    List all files in a specified directory using Databricks' dbutils.

    Parameters
    ----------
    folder_path : str
        The full path to the directory in the data lake storage.

    Returns
    -------
    list of str
        A list of file paths in the specified directory. If the directory is empty, returns an empty list.

    Raises
    ------
    ValueError
        If the folder_path is empty or None.

    Example
    -------
    >>> folder_path = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW_UC1/Process_Mining/"
    >>> files = list_files_in_directory(folder_path)
    >>> for file in files:
    ...     print(file)
    """
    if not folder_path:
        raise ValueError("folder_path cannot be empty or None")

    try:
        # Use dbutils.fs.ls to list files and directories in the specified path
        file_list = dbutils.fs.ls(folder_path)

        # If directory is empty, return an empty list
        if not file_list:
            print(f"Directory '{folder_path}' is empty.")
            return []

        # Filter out directories and return only file paths
        return [file.path for file in file_list if not file.isDir()]
    except Exception as e:
        print(f"An error occurred while listing files: {str(e)}")
        return []


def list_directories_in_path(path):
    """
    List all directories in a specified path using Databricks' dbutils.

    Args:
        path (str): The full path to the directory in the data lake storage.

    Returns:
        list: A list of directory paths in the specified path.

    Raises:
        ValueError: If the path is empty or None.

    Example:
        >>> path = " "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW_UC1/Process_Mining/"
        >>> directories = list_directories_in_path(path)
        >>> for directory in directories:
        ...     print(directory)
    """
    if not path:
        raise ValueError("path cannot be empty or None")

    try:
        # Use dbutils.fs.ls to list files and directories in the specified path
        all_items = dbutils.fs.ls(path)

        # Filter out files and return only directory paths
        return [item.path for item in all_items if item.isDir()]
    except Exception as e:
        print(f"An error occurred while listing directories: {str(e)}")
        return []


def list_directory_recursive(path: str) -> List[Dict[str, str]]:
    """
    Recursively list all files and directories under a given path.

    Args:
        path (str): The full path to the directory in the data lake storage.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'path' and 'type'
        ('file' or 'directory') for each item found.

    Raises:
        ValueError: If the path is empty or None.

    Example:
        >>> path = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW_UC1/Process_Mining/Rohdaten/"
        >>> items = list_directory_recursive(path)
        >>> for item in items:
        ...     print(f"{item['type']}: {item['path']}")
    """
    if not path:
        raise ValueError("path cannot be empty or None")

    def _recursive_list(current_path: str) -> List[Dict[str, str]]:
        try:
            items = dbutils.fs.ls(current_path)
            result = []
            for item in items:
                if item.isDir():
                    result.append({"path": item.path, "type": "directory"})
                    result.extend(_recursive_list(item.path))
                else:
                    result.append({"path": item.path, "type": "file"})
            return result
        except Exception as e:
            print(f"An error occurred while listing {current_path}: {str(e)}")
            return []

    return _recursive_list(path)


def list_files_by_pattern(directory: str, pattern: str) -> List[Dict[str, str]]:
    """
    List all files in a specified directory that match a given pattern.

    Args:
        directory (str): The full path to the directory in the data lake storage.
        pattern (str): The pattern to match against file names. Supports wildcards (* and ?).

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'path' and 'name'
        for each file that matches the pattern.

    Raises:
        ValueError: If the directory or pattern is empty or None.

    Example:
        >>> directory = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW"
        >>> pattern = "*.csv"
        >>> files = list_files_by_pattern(directory, pattern)
        >>> for file in files:
        ...     print(f"Name: {file['name']}, Path: {file['path']}")
    """
    if not directory or not pattern:
        raise ValueError("directory and pattern cannot be empty or None")

    def _match_pattern(name: str, pattern: str) -> bool:
        return fnmatch.fnmatch(name.lower(), pattern.lower())

    try:
        all_files = dbutils.fs.ls(directory)
        matched_files = [
            {"name": file.name, "path": file.path}
            for file in all_files
            if not file.isDir() and _match_pattern(file.name, pattern)
        ]
        return matched_files
    except Exception as e:
        print(f"An error occurred while listing files: {str(e)}")
        return []


def list_csv_files(folder_path: str) -> list[str]:
    """
    List all CSV files in a specified directory using Databricks' dbutils.

    Parameters
    ----------
    folder_path : str
        The full path to the directory in the data lake storage.

    Returns
    -------
    list of str
        A list of CSV file paths in the specified directory. If there are no CSV files, returns an empty list.

    Raises
    ------
    ValueError
        If the folder_path is empty or None.

    Example
    -------
    >>> folder_path = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW_UC1/Process_Mining/"
    >>> csv_files = list_csv_files(folder_path)
    >>> for file in csv_files:
    ...     print(file)
    """
    if not folder_path:
        raise ValueError("folder_path cannot be empty or None")

    try:
        # Use dbutils.fs.ls to list files and directories in the specified path
        file_list = dbutils.fs.ls(folder_path)

        # Filter to only include CSV files and exclude directories
        csv_files = [
            file.path
            for file in file_list
            if file.path.endswith(".csv") and not file.isDir()
        ]

        # If no CSV files are found, optionally print a message and return an empty list
        if not csv_files:
            print(f"No CSV files found in directory '{folder_path}'.")
            return []

        return csv_files
    except Exception as e:
        print(f"An error occurred while listing CSV files: {str(e)}")
        return []


def list_excel_files(folder_path: str) -> list[str]:
    """
    List all Excel files in a specified directory using Databricks' dbutils.

    Parameters
    ----------
    folder_path : str
        The full path to the directory in the data lake storage.

    Returns
    -------
    list of str
        A list of Excel file paths (both .xls and .xlsx) in the specified directory. If there are no Excel files, returns an empty list.

    Raises
    ------
    ValueError
        If the folder_path is empty or None.

    Example
    -------
    >>> folder_path = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW_UC1/Process_Mining/"
    >>> excel_files = list_excel_files(folder_path)
    >>> for file in excel_files:
    ...     print(file)
    """
    if not folder_path:
        raise ValueError("folder_path cannot be empty or None")

    try:
        # Use dbutils.fs.ls to list files and directories in the specified path
        file_list = dbutils.fs.ls(folder_path)

        # Filter to only include Excel files (.xls, .xlsx) and exclude directories
        excel_files = [
            file.path
            for file in file_list
            if (file.path.endswith(".xls") or file.path.endswith(".xlsx"))
            and not file.isDir()
        ]

        # If no Excel files are found, optionally print a message and return an empty list
        if not excel_files:
            print(f"No Excel files found in directory '{folder_path}'.")
            return []

        return excel_files
    except Exception as e:
        print(f"An error occurred while listing Excel files: {str(e)}")
        return []


def list_text_files(folder_path: str) -> list[str]:
    """
    List all text files in a specified directory using Databricks' dbutils.

    Parameters
    ----------
    folder_path : str
        The full path to the directory in the data lake storage.

    Returns
    -------
    list of str
        A list of text file paths (files ending with .txt) in the specified directory. If there are no text files, returns an empty list.

    Raises
    ------
    ValueError
        If the folder_path is empty or None.

    Example
    -------
    >>> folder_path = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW_UC1/Process_Mining/"
    >>> text_files = list_text_files(folder_path)
    >>> for file in text_files:
    ...     print(file)
    """
    if not folder_path:
        raise ValueError("folder_path cannot be empty or None")

    try:
        # Use dbutils.fs.ls to list files and directories in the specified path
        file_list = dbutils.fs.ls(folder_path)

        # Filter to only include text files (.txt) and exclude directories
        text_files = [
            file.path
            for file in file_list
            if file.path.endswith(".txt") and not file.isDir()
        ]

        # If no text files are found, optionally print a message and return an empty list
        if not text_files:
            print(f"No text files found in directory '{folder_path}'.")
            return []

        return text_files
    except Exception as e:
        print(f"An error occurred while listing text files: {str(e)}")
        return []


def list_non_empty_csv_files(folder_path: str) -> list[str]:
    """
    List all non-empty CSV files in a specified directory using Databricks' dbutils.

    Parameters
    ----------
    folder_path : str
        The full path to the directory in the data lake storage.

    Returns
    -------
    list of str
        A list of non-empty CSV file paths in the specified directory. If there are no non-empty CSV files, returns an empty list.

    Raises
    ------
    ValueError
        If the folder_path is empty or None.

    Example
    -------
    >>> folder_path = "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW_UC1/Process_Mining/"
    >>> csv_files = list_non_empty_csv_files(folder_path)
    >>> for file in csv_files:
    ...     print(file)
    """
    if not folder_path:
        raise ValueError("folder_path cannot be empty or None")

    try:
        # Use dbutils.fs.ls to list files and directories in the specified path
        file_list = dbutils.fs.ls(folder_path)

        # Filter to only include non-empty CSV files and exclude directories
        non_empty_csv_files = [
            file.path
            for file in file_list
            if file.path.endswith(".csv") and not file.isDir() and file.size > 0
        ]

        # If no non-empty CSV files are found, optionally print a message and return an empty list
        if not non_empty_csv_files:
            print(f"No non-empty CSV files found in directory '{folder_path}'.")
            return []

        return non_empty_csv_files
    except Exception as e:
        print(f"An error occurred while listing non-empty CSV files: {str(e)}")
        return []


def _validate_input_get_delta_table(
    database: str, table_name: str, columns: Optional[List[str]] = None
) -> None:
    """
    Validate the input parameters for the Delta table extraction.

    Args:
        database (str): The name of the database.
        table_name (str): The name of the table.
        columns (Optional[List[str]]): List of column names to extract.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(database, str) or not database.strip():
        raise ValueError("Database name must be a non-empty string.")
    if not isinstance(table_name, str) or not table_name.strip():
        raise ValueError("Table name must be a non-empty string.")
    if columns is not None:
        if not isinstance(columns, list) or not all(
            isinstance(col, str) for col in columns
        ):
            raise ValueError("Columns must be a list of strings.")


def _construct_table_path(database: str, table_name: str) -> str:
    """
    Construct the full table path for the Delta table.

    Args:
        database (str): The name of the database.
        table_name (str): The name of the table.

    Returns:
        str: The full table path.
    """
    return f"{database}.{table_name}"


def get_delta_table(
    database: str, table_name: str, columns: Optional[List[str]] = None
) -> Optional[DataFrame]:
    """
    Get a Delta table or specific columns from the Hive metastore in Databricks.

    This function validates the input, constructs the table path,
    and reads the Delta table or specified columns from the Hive metastore
    using the built-in spark session in Databricks.

    Args:
        database (str): The name of the database containing the table.
        table_name (str): The name of the table to extract.
        columns (Optional[List[str]]): List of column names to extract.
                                       If None, all columns are extracted.

    Returns:
        Optional[DataFrame]: The extracted Delta table (or specified columns) as a DataFrame,
                             or None if an error occurs.

    Example:
        >>> # Get entire table
        >>> df = get_delta_table("my_database", "my_table")
        >>> # Get specific columns
        >>> df = get_delta_table("my_database", "my_table", ["column1", "column2"])
        >>> if df is not None:
        ...     df.show()
    """
    try:
        _validate_input_get_delta_table(database, table_name, columns)
        full_table_path = _construct_table_path(database, table_name)

        if columns:
            return spark.table(full_table_path).select(*columns)
        else:
            return spark.table(full_table_path)

    except ValueError as e:
        print(f"Input validation error: {e}")
        return None
    except AnalysisException as e:
        print(f"Error reading Delta table: {e}")
        return None


def _write_delta_table(
    df: DataFrame,
    path: str,
    partition_by: Optional[List[str]] = None,
    mode: str = "overwrite",
) -> None:
    """Helper function to write the DataFrame as a Delta table."""
    writer = df.write.format("delta").mode(mode)
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.save(path)


def _optimize_delta_table(path: str, z_order_by: Optional[List[str]] = None) -> None:
    """Helper function to optimize the Delta table and apply Z-Ordering if specified."""
    delta_table = DeltaTable.forPath(spark, path)
    if z_order_by:
        z_order_cols = ", ".join(z_order_by)
        delta_table.optimize().executeZOrderBy(z_order_cols)
    else:
        delta_table.optimize().executeCompaction()


def save_delta_table(
    df: DataFrame,
    path: str,
    partition_by: Optional[List[str]] = None,
    z_order_by: Optional[List[str]] = None,
    mode: str = "overwrite",
) -> None:
    """
    Saves a DataFrame as a Delta table, performs optimization, and applies Z-Ordering.

    Args:
        df (DataFrame): The DataFrame to be saved.
        path (str): The path where the Delta table should be saved.
        partition_by (Optional[List[str]]): List of columns for partitioning.
        z_order_by (Optional[List[str]]): List of columns for Z-Ordering.
        mode (str): Write mode ("overwrite", "append", etc.). Defaults to "overwrite".

    Returns:
        None

    Example:
        >>> df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "value"])
        >>> save_delta_table(df, "abfss://prod@eudldegikoproddl.dfs.core.windows.net/PROD/usecases/AnalyticsUW_UC1/prod_uc_analyticsuw_uc1/WEBLIFE, partition_by=["_CASE_KEY"], z_order_by=["_CASE_KEY"])
    """
    _write_delta_table(df, path, partition_by, mode)
    _optimize_delta_table(path, z_order_by)
    print(f"✅Delta table has been saved and optimized: {path}")


def _database_exists(database_name: str) -> bool:
    """
    Check if a database exists in Databricks.

    Args:
        database_name (str): The name of the database to check.

    Returns:
        bool: True if the database exists, False otherwise.
    """
    result = spark.sql(f"SHOW DATABASES LIKE '{database_name}'").collect()
    return len(result) > 0


def _create_database(
    database_name: str,
    location: str,
    database_properties: Dict[str, str],
    database_comment: str,
) -> None:
    """
    Create a new database with the given parameters.

    Args:
        database_name (str): The name of the database to create.
        location (str): The ABFSS path for the database.
        database_properties (Dict[str, str]): Additional properties for the database.
        database_comment (str): A comment describing the database.
    """
    properties_str = ", ".join(
        [f"'{k}' = '{v}'" for k, v in database_properties.items()]
    )
    create_db_sql = f"""
    CREATE DATABASE IF NOT EXISTS {database_name}
    LOCATION '{location}'
    WITH DBPROPERTIES ({properties_str})
    COMMENT '{database_comment}'
    """
    spark.sql(create_db_sql)


def _generate_message(database_name: str, exists: bool) -> str:
    """
    Generate an appropriate message based on whether the database exists.

    Args:
        database_name (str): The name of the database.
        exists (bool): Whether the database already exists.

    Returns:
        str: A message indicating the result of the operation.
    """
    if exists:
        return f"🚫 Database '{database_name}' already exists. No action taken."
    else:
        return f"✅ Database '{database_name}' has been successfully created!"


def create_database_if_not_exists(
    database_name: str,
    location: str,
    database_properties: Optional[Dict[str, str]] = None,
    database_comment: str = "",
) -> str:
    """
    Create a database if it doesn't exist in Databricks.

    This function checks if a database with the given name exists. If it doesn't,
    it creates the database with the specified parameters. If it already exists,
    it returns a message indicating so.

    Args:
        database_name (str): The name of the database to create.
        location (str): The ABFSS path for the database storage location.
            This should be in the format 'abfss://<container>@<storage-account-name>.dfs.core.windows.net/<path>'.
        database_properties (Optional[Dict[str, str]]): Additional properties for the database.
            Defaults to None.
        database_comment (str): A comment describing the database. Defaults to an empty string.

    Returns:
        str: A message indicating the result of the operation.

    Example:
        >>> result = create_database_if_not_exists(
        ...     "my_new_database",
        ...     "abfss://container@storageaccount.dfs.core.windows.net/databases/my_new_database",
        ...     {"creator": "data_team", "purpose": "analytics"},
        ...     "Database for analytics project"
        ... )
        >>> print(result)
    """
    if database_properties is None:
        database_properties = {}

    if _database_exists(database_name):
        return _generate_message(database_name, True)

    _create_database(database_name, location, database_properties, database_comment)
    return _generate_message(database_name, False)


def _validate_input(df: DataFrame, file_path: str) -> None:
    """
    Validate the input DataFrame and file path.

    Args:
        df (DataFrame): The Spark DataFrame to be validated.
        file_path (str): The file path to be validated.

    Raises:
        ValueError: If the DataFrame is empty or if the file_path is invalid.
    """
    if df.rdd.isEmpty():
        raise ValueError("DataFrame is empty. Cannot save an empty DataFrame.")
    if not file_path:
        raise ValueError("Invalid file path. Please provide a valid path.")


def _save_dataframe(
    df: DataFrame, file_path: str, header: bool, separator: str, overwrite: bool
) -> None:
    """
    Save the DataFrame as a CSV file.

    Args:
        df (DataFrame): The Spark DataFrame to be saved.
        file_path (str): The path where the CSV file will be saved.
        header (bool): Whether to include a header in the CSV file.
        separator (str): The separator to use in the CSV file.
        overwrite (bool): Whether to overwrite existing files.
    """
    write_mode = "overwrite" if overwrite else "error"
    df.repartition(1).write.format("csv").mode(write_mode).save(
        file_path, header=header, sep=separator
    )
