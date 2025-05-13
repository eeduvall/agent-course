from board_to_fen.predict import get_fen_from_image
from langchain.tools import Tool
from tools.file_downloader import get_task_file

def board_to_fen(task_id: str):
    """
    Converts a chessboard image to a FEN string.
    
    Args:
        task_id: The task ID to download the chess image
        
    Returns:
        The FEN notation of the chess position
    """
    # Download the image using the task_id
    result = get_task_file(task_id)
    
    # Check if we got a valid image
    if result.get('file_type') == 'png' and 'content' in result:
        img = result['content']  # This should be a PIL Image object
        return get_fen_from_image(img)
    else:
        return f"Error: Could not process image for task {task_id}. File type: {result.get('file_type')}. Stop processing immediately and report an error."

board_to_fen_tool = Tool(
    name="board_to_fen",
    func=board_to_fen,
    description="Converts a chessboard image to a FEN string. Provide the task_id to analyze the chess position. Do NOT use the file_downloader tool before this one."
)