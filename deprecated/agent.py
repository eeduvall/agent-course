import requests
import json
import os # For API keys potentially
from langchain_mistralai import ChatMistralAI # Keep for standard parameters if needed
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
)
# Import ToolCall specifically for constructing the message
from langchain_core.agents import AgentActionMessageLog, AgentFinish, AgentAction
from langchain_core.tools import BaseTool # For type hinting
from langchain_core.outputs import ChatGeneration, Generation # Needed for AIMessage structure
from langchain_core.load.dump import dumpd # To format messages for API

# ToolCall is needed for constructing the AIMessage with tool calls
from langchain_core.messages import ToolCall

from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, List, Dict, Any, Sequence

from chat_agent.mistral_agent import invoke_llm_manually

# --- Configuration ---
MISTRAL_API_BASE = "https://shd-mistral-small-3-1-24b-instruct-2503.apps.ocp-glue01.pg.wwtatc.ai/v1"
MISTRAL_MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

# # --- Tool Formatting Function ---
# # Helper to generate the JSON structure the API expects for tools
# def format_tool_for_api(tool: BaseTool) -> Dict[str, Any]:
#     tool_schema = tool.get_input_schema().schema()
#     return {
#         "type": "function",
#         "function": {
#             "name": tool.name,
#             "description": tool.description,
#             "parameters": tool_schema,
#         },
#     }

# # --- Helper Function to map LangChain message types to API roles ---
# def get_api_role(message: BaseMessage) -> str:
#     if isinstance(message, HumanMessage):
#         return "user"
#     elif isinstance(message, AIMessage):
#         return "assistant"
#     elif isinstance(message, SystemMessage):
#         return "system"
#     elif isinstance(message, ToolMessage) or isinstance(message, FunctionMessage) : # FunctionMessage might be used internally
#          return "tool" # Role for tool results
#     else:
#         # Fallback or raise error for unknown types
#         print(f"Warning: Unknown message type {type(message)}, using 'user'.")
#         return "user"

# # --- Custom LLM Invocation Function (Corrected Message Formatting) ---
# def invoke_llm_manually(
#     messages: Sequence[BaseMessage],
#     tools: Sequence[BaseTool],
#     api_base_url: str,
#     model_name: str,
#     auth_headers: Dict[str, str] = {},
#     temperature: float = 0.0,
#     max_tokens: int = 2048,
# ) -> AIMessage:
#     """
#     Manually calls the LLM API, parses the response, and constructs
#     an AIMessage, handling the 'content: null' case for tool calls.
#     Uses CORRECT message formatting for the API.
#     """
#     endpoint = f"{api_base_url.rstrip('/')}/chat/completions"

#     # --- Corrected Message Formatting ---
#     formatted_messages = []
#     for msg in messages:
#         role = get_api_role(msg)
#         # Default message structure
#         msg_dict = {"role": role, "content": msg.content}

#         # Special handling for AIMessages with tool calls (outgoing)
#         if isinstance(msg, AIMessage) and msg.tool_calls:
#             # Ensure content is null or omitted if empty/None
#             if not msg.content:
#                 msg_dict.pop("content", None)

#             # --- MODIFICATION START ---
#             # Handle tool_calls potentially being dicts from state serialization
#             api_tool_calls = []
#             for tc in msg.tool_calls:
#                 # Check if tc is a dict (likely from state) or a ToolCall object
#                 if isinstance(tc, dict):
#                     call_id = tc.get('id')
#                     call_name = tc.get('name')
#                     call_args = tc.get('args', {}) # Default to empty dict if missing
#                 elif hasattr(tc, 'id'): # Check if it looks like a ToolCall object
#                     call_id = tc.id
#                     call_name = tc.name
#                     call_args = tc.args
#                 else:
#                     print(f"Warning: Unknown tool call format: {tc}. Skipping.")
#                     continue # Skip this tool call if format is unexpected

#                 api_tool_calls.append({
#                     "id": call_id,
#                     "type": "function", # Assuming only function tools
#                     "function": {"name": call_name, "arguments": json.dumps(call_args)}
#                 })
#             # --- MODIFICATION END ---

#             if api_tool_calls: # Only add if we successfully parsed calls
#                 msg_dict["tool_calls"] = api_tool_calls
#             # Ensure role is 'assistant' for outgoing tool calls
#             msg_dict["role"] = "assistant"

#         # Special handling for ToolMessages (incoming results)
#         elif isinstance(msg, ToolMessage):
#             msg_dict = {
#                 "role": "tool",
#                 "content": msg.content,
#                 "tool_call_id": msg.tool_call_id
#             }

#         formatted_messages.append(msg_dict)
#     # ------------------------------------

#     # --- Corrected Tool Formatting (Simpler Schema based on successful test) ---
#     formatted_tools = []
#     for tool in tools:
#          # Use the schema structure that worked manually
#          formatted_tools.append({
#             "type": "function",
#             "function": {
#                 "name": tool.name, # Use the actual tool name (e.g., duckduckgo_search_run)
#                 "description": tool.description,
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                          # Assuming 'query' is the standard arg for DDG Search Run
#                          "query": {"type": "string", "description": "The search query"}
#                     },
#                     "required": ["query"] # Infer required args if possible, or hardcode if known
#                 }
#             }
#          })
#     # ------------------------------------


#     payload = {
#         "model": model_name,
#         "messages": formatted_messages,
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#     }
#     if formatted_tools:
#         payload["tools"] = formatted_tools
#         payload["tool_choice"] = "auto" # Keep this, usually correct

#     print(f"--- Sending Manual Request to: {endpoint} ---")
#     print(f"--- Sending Headers ---")
#     # Ensure auth_headers actually has your auth token in it correctly!
#     final_headers = {**{"Content-Type": "application/json"}, **auth_headers}
#     print(final_headers)
#     print(f"--- Sending Payload ---")
#     print(json.dumps(payload, indent=2)) # Print the corrected payload

#     try:
#         response = requests.post(
#             endpoint,
#             headers=final_headers, # Use the combined headers
#             json=payload,
#             timeout=90
#         )
#         response.raise_for_status()
#         response_json = response.json()
#         # (Rest of the parsing logic remains the same as before)
#         # ... see previous good version for parsing choices[0].message ...

#         # --- Start Copy from previous good version ---
#         if not response_json.get("choices"):
#              raise ValueError("Invalid response format: 'choices' field missing.")
#         message_data = response_json["choices"][0].get("message")
#         if not message_data:
#              raise ValueError("Invalid response format: 'message' field missing in choices.")

#         finish_reason = response_json["choices"][0].get("finish_reason")
#         usage_metadata = response_json.get("usage")

#         content = message_data.get("content")
#         tool_calls_data = message_data.get("tool_calls")
#         parsed_tool_calls = []

#         if finish_reason == "tool_calls" and tool_calls_data:
#             print("--- API indicated Tool Calls ---")
#             final_content = ""
#             for tool_call_data in tool_calls_data:
#                  if tool_call_data.get("type") == "function":
#                     function_data = tool_call_data.get("function")
#                     if function_data:
#                         try:
#                             arguments_str = function_data.get("arguments", "{}")
#                             arguments_dict = json.loads(arguments_str)
#                         except json.JSONDecodeError:
#                             print(f"Warning: Could not parse tool arguments: {arguments_str}")
#                             arguments_dict = {}
#                         parsed_tool_calls.append(
#                              ToolCall(
#                                  name=function_data.get("name"),
#                                  args=arguments_dict,
#                                  id=tool_call_data.get("id")
#                              )
#                         )
#             if not parsed_tool_calls:
#                  print("Warning: finish_reason was 'tool_calls' but no valid tool calls parsed.")
#                  final_content = content if isinstance(content, str) else ""
#         else:
#             print("--- API indicated Text Response (or unexpected finish_reason) ---")
#             if not isinstance(content, str):
#                  print(f"Warning: Expected string content, got {type(content)}. Using empty string.")
#                  final_content = ""
#             else:
#                  final_content = content
#             parsed_tool_calls = []

#         ai_message = AIMessage(
#             content=final_content,
#             tool_calls=parsed_tool_calls,
#             response_metadata={
#                 "finish_reason": finish_reason,
#                 "usage": usage_metadata
#             }
#         )
#         print(f"--- Constructed AIMessage ---\n{repr(ai_message)}")
#         return ai_message
#         # --- End Copy from previous good version ---

#     except requests.exceptions.Timeout:
#          print("Error: Request timed out.")
#          return AIMessage(content="Error: LLM call timed out.")
#     except requests.exceptions.RequestException as e:
#         print(f"Error during API call: {e}") # This will now print the 400 error again if it persists
#         return AIMessage(content=f"Error: LLM call failed: {e}")
#     except (ValueError, KeyError, json.JSONDecodeError) as e:
#          print(f"Error parsing LLM response: {e}")
#          return AIMessage(content=f"Error: Failed to parse LLM response: {e}")


# --- Agent Definition (Modified) ---
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [self.search_tool]
        self.system_prompt = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: [YOUR FINAL ANSWER GOES HERE].
        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
        If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
        If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""


    def __call__(self, question: str) -> dict:
        print(f"Agent received question: {question}")

        # AgentState definition remains the same
        class AgentState(TypedDict):
            messages: Annotated[List[AnyMessage], add_messages]

        # --- Modified Assistant Node ---
        def assistant_node(state: AgentState):
            print("\n--- Calling Assistant Node (using manual invoke) ---")
            print("Messages going IN:", state["messages"])

            # *** Use the custom invocation function ***
            result = invoke_llm_manually(
                messages=state["messages"],
                tools=self.tools, # Pass the tools list
                api_base_url=MISTRAL_API_BASE,
                model_name=MISTRAL_MODEL_NAME,
                temperature=0.0, # Pass other params
                max_tokens=2048
            )

            print("Assistant Node Manual Result Type:", type(result))
            print("Assistant Node Manual Result Content:", repr(result))
            # Add the manually constructed message to the state
            return {"messages": [result]}

        # ToolNode remains the same
        tool_node = ToolNode(self.tools)

        # --- Graph Definition (remains the same) ---
        builder = StateGraph(AgentState)
        builder.add_node("assistant", assistant_node) # Use the modified node
        builder.add_node("tools", tool_node)
        builder.set_entry_point("assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
            {"tools": "tools", END: END}
        )
        builder.add_edge("tools", "assistant")
        agent = builder.compile()

        # --- Invocation (remains the same) ---
        initial_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question)
        ]
        print("\n--- Invoking Agent ---")
        final_state = {}
        try:
            final_state = agent.invoke({"messages": initial_messages})
            print("\n--- Agent Invocation Finished ---")
            if "messages" in final_state and final_state["messages"]:
                 print("Final Agent State Messages:", final_state["messages"])
                 print("\nFinal Answer Message:", repr(final_state["messages"][-1]))
                 if final_state["messages"][-1].content:
                     print("\nFinal Answer Content:", final_state["messages"][-1].content)
                 else:
                     print("\nFinal Answer: (Tool call or empty content in last message)")
            else:
                print("Error: No messages found in the final state.")
        except Exception as e:
            print("\n--- Agent Invocation Error ---")
            import traceback
            print(f"An error occurred during agent execution: {e}")
            print(traceback.format_exc())

        return final_state["messages"][-1].content