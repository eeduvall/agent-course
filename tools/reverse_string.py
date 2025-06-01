from langchain.tools import Tool

def reverse_string(s):
    """Reverses a string"""
    return s[::-1]

reverse_string_tool = Tool(
    name="reverse_string",
    func=reverse_string,
    description="Reverses a string.",
)