import requests
import os
import json
import pandas as pd
from enum import Enum
from typing import Dict, Any, Union, Optional
from tools.utils.file_api_handler import download_task_file
from langchain.tools import Tool

class FileType(Enum):
    TEXT = "text"
    JSON = "json"
    PYTHON = "python"
    CSV = "csv"
    EXCEL = "excel"
    BINARY = "binary"
    PNG = "png"

def get_task_file(task_id: str, base_url: str = "https://agents-course-unit4-scoring.hf.space") -> Dict[str, Any]:
    """Download a file associated with a task and return its contents in a format based on file type.
    
    Args:
        task_id: The ID of the task to download the file for
        base_url: The base URL of the API
        
    Returns:
        A dictionary containing:
        - file_type: The type of the file (text, json, python, csv, excel, binary)
        - content: The content of the file, formatted according to its type
        - content_type: The content-type of the file
        - filename: The original filename of the downloaded file
    """
    
    try:
        # Get filename and file content from download_task_file
        filename, file_content = download_task_file(task_id)
        
        # Determine file type from filename extension
        file_extension = os.path.splitext(filename)[1].lower() if filename else ''
        
        if file_extension in ['.json']:
            file_type = FileType.JSON
        elif file_extension in ['.py', '.python']:
            file_type = FileType.PYTHON
        elif file_extension in ['.csv']:
            file_type = FileType.CSV
        elif file_extension in ['.xlsx', '.xls']:
            file_type = FileType.EXCEL
        elif file_extension in ['.txt', '.md', '.text']:
            file_type = FileType.TEXT
        elif file_extension in ['.png']:
            file_type = FileType.PNG
        else:
            # Default to binary for unknown extensions
            file_type = FileType.BINARY
            
        # For content type in the return value
        content_type = f"application/octet-stream"  # Default content type
        
        # Process content based on file type
        if file_type == FileType.JSON:
            try:
                content = json.loads(file_content.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # If decoding as JSON fails, treat as binary
                content = file_content
                file_type = FileType.BINARY
        elif file_type == FileType.EXCEL:
            # For Excel files, use pandas to read the data
            try:
                import io
                excel_file = io.BytesIO(file_content)
                content = pd.read_excel(excel_file)
            except Exception as e:
                print(f"Error parsing Excel file: {e}")
                # If parsing as Excel fails, treat as binary
                content = file_content
                file_type = FileType.BINARY
        elif file_type in [FileType.PYTHON, FileType.TEXT, FileType.CSV]:
            # For text-based files, return the string representation
            try:
                content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # If decoding as text fails, treat as binary
                content = file_content
                file_type = FileType.BINARY
        else:  # Binary
            content = file_content
            
        return {
            "file_type": file_type.value,
            "content": content,
            "filename": filename
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file for task {task_id}: {e}")
        return {
            "file_type": "error",
            "content": str(e),
            "content_type": ""
        }
    except json.JSONDecodeError as e:
        # If the file was supposed to be JSON but couldn't be parsed
        print(f"Error parsing JSON for task {task_id}: {e}")
        return {
            "file_type": "text",  # Fallback to text
            "content": str(e),
            "content_type": content_type
        }
    except Exception as e:
        print(f"Unexpected error processing file for task {task_id}: {e}")
        return {
            "file_type": "error",
            "content": str(e),
            "content_type": ""
        }

file_downloader_tool = Tool(
    name="file_downloader",
    func=get_task_file,
    description="Downloads a file associated with a task ID and returns its contents based on file type (supports text, JSON, CSV, Excel, Python, and binary files). Excel files are automatically converted to pandas DataFrames."
)
