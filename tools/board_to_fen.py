import io
from PIL import ImageFile
from board_to_fen.predict import get_fen_from_image
from langchain.tools import Tool

def board_to_fen(img: ImageFile):
    # img = Image.open(io.BytesIO(image_binary))
    return get_fen_from_image(img)

board_to_fen_tool = Tool(
    name="board_to_fen",
    func=board_to_fen,
    description="Converts a chessboard image to a FEN string. Use after downloading the image with file_downloader."
)