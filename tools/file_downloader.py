import requests
import os
import json
from enum import Enum
from typing import Dict, Any, Union, Optional
from tools.utils.file_downloader import download_task_file

class FileType(Enum):
    TEXT = "text"
    JSON = "json"
    PYTHON = "python"
    CSV = "csv"
    BINARY = "binary"

def determine_file_type_from_content_type(content_type: str) -> FileType:
    """Determine the file type based on the content-type header."""
    if content_type.startswith('text/plain'):
        return FileType.TEXT
    elif content_type.startswith('application/json'):
        return FileType.JSON
    elif content_type.startswith('text/csv'):
        return FileType.CSV
    elif content_type.startswith('text/x-python') or content_type.startswith('application/x-python'):
        return FileType.PYTHON
    elif content_type.startswith('text/'):
        # Other text formats default to TEXT
        return FileType.TEXT
    else:
        # Non-text formats default to BINARY
        print(f"Unknown content type: {content_type}")
        return FileType.BINARY


def get_task_file(task_id: str, base_url: str = "https://agents-course-unit4-scoring.hf.space") -> Dict[str, Any]:
    """Download a file associated with a task and return its contents in a format based on file type.
    
    Args:
        task_id: The ID of the task to download the file for
        base_url: The base URL of the API
        
    Returns:
        A dictionary containing:
        - file_type: The type of the file (text, json, python, csv, binary)
        - content: The content of the file, formatted according to its type
        - content_type: The content-type of the file
    """
    
    try:
        # Get file content and content-type from download_task_file
        file_content, content_type = download_task_file(task_id)
        
        # Determine file type from content-type header
        file_type = determine_file_type_from_content_type(content_type)
        
        # Process content based on file type
        if file_type == FileType.JSON:
            try:
                content = json.loads(file_content.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # If decoding as JSON fails, treat as binary
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
            "content_type": content_type
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


# Create a tool that can be used with LangChain
from langchain.tools import Tool

file_downloader_tool = Tool(
    name="file_downloader",
    func=get_task_file,
    description="Downloads a file associated with a task ID and returns its contents based on file type."
)
