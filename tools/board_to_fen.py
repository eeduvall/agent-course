import io
from PIL import Image
from board_to_fen.predict import get_fen_from_image
from langchain.tools import Tool

def board_to_fen(image_binary):
    img = Image.open(io.BytesIO(image_binary))
    return get_fen_from_image(img)

board_to_fen_tool = Tool(
    name="board_to_fen",
    func=board_to_fen,
    description="Converts a chessboard image to a FEN string."
)






#####
# import io
# from PIL import Image
# from board_to_fen.predict import get_fen_from_image
# from langchain.tools import Tool
# from tools.file_downloader import get_task_file

# def board_to_fen(task_id: str) -> str:
#     print(f"Converting board to FEN for task ID: {task_id}")
#     file_content = get_task_file(task_id)
#     img = Image.open(io.BytesIO(file_content))
#     return get_fen_from_image(img)

# board_to_fen_tool = Tool(
#     name="board_to_fen",
#     func=board_to_fen,
#     description="Converts a chessboard image to a FEN string based on the task_id of the question."
# )
