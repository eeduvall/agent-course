import os
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
)
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, List, Dict, Any, Sequence

from chat_agent.meta_agent import invoke_llm_manually

from tools.file_downloader import file_downloader_tool
from tools.board_to_fen import board_to_fen_tool
from tools.chess_next_move_calculator import chess_next_move_tool
from tools.transcribe_audio import transcribe_audio_tool
from tools.youtube import youtube_tool
from tools.reverse_string import reverse_string_tool

# --- Configuration ---
AWS_REGION = "us-east-2"
LLAMA_MODEL_ID = "us.meta.llama3-1-405b-instruct-v1:0"

# --- Agent Definition (Modified) ---
class BasicAgent:
    def __init__(self):
        load_dotenv()
        print("BasicAgent initialized.")
        self.search_tool = DuckDuckGoSearchRun()
        # self.system_prompt = """You are a helpful AI assistant using the AWS Bedrock Llama model. You follow the ReAct (Reasoning and Acting) approach to solve problems step by step.
        
        # When you need information, you can use the available tools. For each step:
        # 1. Think about what you know and what you need to find out
        # 2. Decide which tool to use (if any)
        # 3. Use the tool and observe the result
        # 4. Update your understanding based on the result
        
        # When you have a final answer, provide just the answer in as few words as possible and no other text.
        # """
        self.system_prompt = ("""
**Your Process - Follow these steps meticulously:**

1.  **Understand the Question:**
    * Read the question carefully. Identify the specific information being requested.
    * Note any constraints on the answer format (e.g., "algebraic notation," "comma separated list," "alphabetical order," "only the first name," "numeric output," "IOC country code," "surname of the equine veterinarian," "USD with two decimal places").
    * Identify if any external resources (files, URLs, images, audio, video) are mentioned or implied.

2.  **Strategize Tool Use:**
    * Determine which of your available tools is necessary to answer the question.
    * If a file (image, audio, Excel, python) is involved, your first step for that resource is likely `FileDownloader`.
    * If it's a chess image, you'll use `FileDownloader` then `board_to_fen` then `chess_next_move`.
    * If it's an audio file, you'll use `FileDownloader` then `transcribe_audio`.
    * If it's a YouTube video, use `youtube_processor`.
    * If you need to reverse a string, use `reverse_string`
    * If it's a general knowledge question or requires searching the web, use `duckduckgo_search`. Do NOT run more than 3 tool call searches at a time.
    * If the question is a direct instruction that doesn't require external data, you might not need a tool, but you still need to process the instruction.

3.  **Execute Tool(s):**
    * Execute the chosen tool(s) sequentially if needed.
    * For `duckduckgo_search`, formulate a precise search query. Try to anticipate what keywords will yield the most relevant results. You may need to iterate with different queries if the first one doesn't provide the answer. Do NOT run more than 3 searches at a time.
    * When using `FileDownloader`, ensure the input provided is the task_id.

4.  **Extract Information:**
    * From the output of your tool(s), or from your processing of downloaded data, carefully extract *only* the specific piece(s) of information needed to answer the original question.
    * Do not make assumptions or infer information beyond what the tool provides or the data contains.
    * Decide if you need to make additional tool calls for more information.

5.  **Formulate the Answer:**
    * Construct the answer based *solely* on the extracted information.
    * Crucially, ensure the answer *strictly adheres* to any formatting requirements specified in the question. If the question asks for a "comma separated list," provide only that. If it asks for a "numeric output," provide only the number. If it asks for "USD with two decimal places," provide the number in that format.
    * Do not add any conversational fluff, explanations, or apologies unless explicitly asked. Your output should be the direct answer.
    * Do not respond in the format of a sentence. Instead, give the answer directly.
    * The final answer should be the last response of the assistant.

6.  **Final Check:**
    * Reread the original question and your formulated answer. Does your answer directly and completely address the question? Is the formatting correct?

**Example Scenario (based on task_id: 8e867cd7-cff9-4e6c-867a-ff5ddc2550be):**

* **Question:** "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
* **Step 1 (Understand):** The question asks for the count of Mercedes Sosa's studio albums released from 2000 to 2009 inclusive. The information should be sourced from the latest 2022 English Wikipedia. The answer should be a number.
* **Step 2 (Strategize):** Use `duckduckgo_search` to search English Wikipedia for Mercedes Sosa's discography.
* **Step 3 (Execute):** `duckduckgo_search` with a query like "Mercedes Sosa studio albums 2000-2009 site:en.wikipedia.org" or "Mercedes Sosa discography site:en.wikipedia.org".
* **Step 4 (Extract):** Review the search results, navigate to the relevant Wikipedia page (discography). Identify studio albums released within the 2000-2009 timeframe. Count them. (Let's assume for this example the count is 3).
* **Step 5 (Formulate):** The answer is "3".
* **Output:** 3

**Example Scenario (based on task_id: 7bd855d8-463d-4ed5-93ca-5fe35145f733):**

* **Question:** "The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places." (Assume file_name is "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx")
* **Step 1 (Understand):** Calculate total sales from food items, excluding drinks, from data in an Excel file. The answer needs to be in USD format with two decimal places. The file name is "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx".
* **Step 2 (Strategize):** Use `FileDownloader` to get the Excel file. After downloading, the agent will need to process the data from the Excel file (e.g., identify columns for item type and sales, filter out drinks, sum sales for food).
* **Step 3 (Execute Tool(s)):**
    * `FileDownloader` with "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx".
    * (Internal Processing: Agent opens and reads the Excel data, filters for 'food' items, sums their sales. For example, if the sum is $5678.9012)
* **Step 4 (Extract Information):** The calculated total sales for food items is $5678.9012.
* **Step 5 (Formulate Answer):** Format the number as USD with two decimal places: "5678.90".
* **Output:** 5678.90

**Critical Reminders for the Agent:**

* **Stick to the Tools:** Do not use any capabilities beyond the ones listed for external data gathering. Data processing of downloaded files is an internal step. Do not make up information.
* **Precision is Key:** GAIA questions often have one very specific correct answer.
* **File Handling:** If a file is mentioned or implied by the question (e.g., 'the attached Python code', 'sent me an audio recording', 'the image provided', 'The attached Excel file'), use the `FileDownloader` first if the content isn't directly accessible, then the appropriate specialized tool or internal processing logic. For direct URLs that a tool can handle (like `YouTubeProcess`), `FileDownloader` is not be needed as the specialized tool handles fetching.
* **Iterative Search:** If `duckduckgo_search` is used, you might need to try a few queries to find the correct information. Be persistent but focused. Do NOT call Duck Duck Go with more than 3 searches at a time.
""")

    def __call__(self, question: str, task_id: str, file_name: str = '') -> dict:
        # print(f"Agent received question: {question}")

        if file_name != '':
            print(f"File is present '{file_name}'")
            self.tools = [self.search_tool, file_downloader_tool, board_to_fen_tool, transcribe_audio_tool, youtube_tool, reverse_string_tool, chess_next_move_tool]
        else:
            print("File is NOT present")
            self.tools = [self.search_tool, board_to_fen_tool, transcribe_audio_tool, youtube_tool, reverse_string_tool, chess_next_move_tool]

        # AgentState definition remains the same
        class AgentState(TypedDict):
            messages: Annotated[List[AnyMessage], add_messages]

        # --- Modified Assistant Node ---
        def assistant_node(state: AgentState):
            # print("\n--- Calling Assistant Node (using AWS Bedrock Llama) ---")
            # print("Messages going IN:", state["messages"])

            # Use the AWS Bedrock Llama invocation function
            result = invoke_llm_manually(
                messages=state["messages"],
                tools=self.tools,
                model_name=LLAMA_MODEL_ID,
                temperature=0.3,
                max_tokens=5000,
                region_name=AWS_REGION,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN")
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

        # --- Invocation with task_id if provided ---
        if task_id:
            # Add the task_id to the question to make it explicit
            enhanced_question = f"{question}\n\nIMPORTANT: The task_id for this question is '{task_id}'."# Use this exact task_id with the file_downloader tool if necessary.
            question = enhanced_question
            
        initial_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question)
        ]
        # print("\n--- Invoking Agent ---")
        final_state = {}
        answer = ""
        try:
            final_state = agent.invoke({"messages": initial_messages})
            # print("\n--- Agent Invocation Finished ---")
            if "messages" in final_state and final_state["messages"]:
                #  print("Final Agent State Messages:", final_state["messages"])
                #  print("\nFinal Answer Message:", repr(final_state["messages"][-1]))
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
#FINAL ANSWER: [YOUR FINAL ANSWER].
