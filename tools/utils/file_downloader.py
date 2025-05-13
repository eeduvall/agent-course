import requests

def download_task_file(task_id: str):
        """
        Downloads the file for the given task ID.

        Args:
            task_id: The ID of the task whose file needs to be downloaded.

        Returns:
            bytes: The content of the file if the request is successful,
                   None otherwise.

        Raises:
            requests.exceptions.RequestException: If there's an issue with the HTTP request.
        """
        endpoint = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"
        
        try:
            response = requests.get(endpoint, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            content_type = response.headers.get('Content-Type', '').lower()

            # Return the raw content of the file as bytes
            return (content_type, response.content)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file for task ID '{task_id}': {e}")
            return None