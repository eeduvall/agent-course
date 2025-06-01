from langchain.tools import Tool
import chess
import chess.engine

def split_on_first_space(text: str):
    parts = text.split(" ", 1)
    return (parts[0], parts[1] if len(parts) > 1 else text)

def chess_next_move(fen: str):
    """For a given fen string, calculate the next move."""
    fen_split = split_on_first_space(fen)

    final_fen = ""
    if isinstance(result, tuple):
        final_fen = f"{fen_split[0]} b KQkq - 0 1 {fen_split[1]}"
    else:
        final_fen = fen

    board = chess.Board(final_fen)
    with chess.engine.SimpleEngine.popen_uci("stockfish") as engine:
        result = engine.play(board, chess.engine.Limit(time=2))

chess_next_move_tool = Tool(
    name="chess_next_move",
    func=chess_next_move,
    description="For a given fen string, calculate the next move.",
)