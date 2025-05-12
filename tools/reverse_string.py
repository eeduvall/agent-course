from langchain.tools import Tool

def reverse_string(text: str) -> str:
    """Reverses a string for more easy parsing."""
    return text[::-1]

# Initialize the tool
reverse_string_tool = Tool(
    name="reverse_string",
    func=reverse_string,
    description="Reverses a given string."
)