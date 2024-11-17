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
