from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
)
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, List, Dict, Any, Sequence

from chat_agent.meta_agent import invoke_llm_manually

# --- Configuration ---
AWS_REGION = "us-east-2"  # AWS region where Llama 405B is available
LLAMA_MODEL_ID = "us.meta.llama3-1-405b-instruct-v1:0"  # Llama 405B model ID

# --- Agent Definition (Modified) ---
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [self.search_tool]
        # self.system_prompt = """You are a helpful AI assistant using the AWS Bedrock Llama 405B model. You follow the ReAct (Reasoning and Acting) approach to solve problems step by step.
        
        # When you need information, you can use the available tools. For each step:
        # 1. Think about what you know and what you need to find out
        # 2. Decide which tool to use (if any)
        # 3. Use the tool and observe the result
        # 4. Update your understanding based on the result
        
        # When you have a final answer, provide just the answer in as few words as possible and no other text.
        # """
        self.system_prompt = "You are a general AI assistant. I will ask you a question. " \
                     "Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. " \
                     "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings. " \
                     "If you are asked for a number, don't use a comma to write your number neither use units such as $ or percent sign unless specified otherwise. " \
                     "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. " \
                     "If you are asked for a comma-separated list, apply the above rules depending on whether the element to be put in the list is a number or a string."

        print("\n--- Agent Initialized With PROMPT---")
        print(self.system_prompt)
        print("\n--- END Agent Initialized With PROMPT---")

    def __call__(self, question: str) -> dict:
        print(f"Agent received question: {question}")

        # AgentState definition remains the same
        class AgentState(TypedDict):
            messages: Annotated[List[AnyMessage], add_messages]

        # --- Modified Assistant Node ---
        def assistant_node(state: AgentState):
            print("\n--- Calling Assistant Node (using AWS Bedrock Llama 405B) ---")
            print("Messages going IN:", state["messages"])

            # Use the AWS Bedrock Llama 405B invocation function
            result = invoke_llm_manually(
                messages=state["messages"],
                tools=self.tools,
                model_name=LLAMA_MODEL_ID,
                temperature=0.2,
                max_tokens=5000,
                region_name=AWS_REGION
            )

            print("Assistant Node Result Type:", type(result))
            print("Assistant Node Result Content:", repr(result))
            # Add the message to the state
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