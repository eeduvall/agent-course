from langchain_mistralai import ChatMistralAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated
from IPython.display import Image, display
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from BasicToolNode import BasicToolNode

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
    def __call__(self, question: str) -> str:
        # print(f"Agent received question (first 50 chars): {question[:50]}...")
        print(f"Agenct received question:  {question}")

        system_prompt = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: [YOUR FINAL ANSWER].
        YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
        If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
        If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""


        search_tool = DuckDuckGoSearchRun()

        chat = ChatMistralAI(
            model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            temperature=0,
            max_retries=2,
            max_tokens=2048,
            base_url="https://shd-mistral-small-3-1-24b-instruct-2503.apps.ocp-glue01.pg.wwtatc.ai/v1",
            # system_prompt=system_prompt
            # model_kwargs={"system_prompt": system_prompt}
        )
        tools=[search_tool]
        print("Tools are ", tools)
        chat_with_tools = chat.bind_tools(tools)


        # Generate the AgentState and Agent graph
        class AgentState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]

        def assistant(state: AgentState):
            #TODO seems to fail when it uses tools
            print("Messages being passed to chat_with_tools:", state["messages"])
            result = chat_with_tools.invoke(state["messages"])
            print("Chat response type:", type(result))
            # print("Chat response content:", result)
            print("Assistant Node Raw Result Content:", repr(result))
            return {
                "messages": [result],
            }

        builder = StateGraph(AgentState)

        builder.add_node("assistant", assistant)
        # tool_node = BasicToolNode(tools=tools)
        # builder.add_node("tools", tool_node)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message requires a tool, route to tools
            # Otherwise, provide a direct response
            tools_condition, # Checks the *last* message: if tool_calls, returns "tools", else returns END
            {
                "tools": "tools", # If tool_calls detected, go to tool_node
                END: END          # Otherwise, finish the graph execution
            }
        )
        builder.add_edge("tools", "assistant")
        agent = builder.compile()
        # try:
        #     display(Image(agent.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)))
        # except Exception as e:
        #     # This requires some extra dependencies and is optional
        #     print("failed to display", e)


        # ORIGINAL METHOD PER AGENT COURSE
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]

        print("\n--- Invoking Agent ---")
        # response = {}
        final_state = {}

        #GEMINI SUGGESTIONS
        try:
            # Invoke the agent. The state will accumulate messages.
            final_state = agent.invoke({"messages": messages}) # Pass the whole state dictionary
            print("\n--- Agent Invocation Finished ---")
            # Print the *final* messages list from the returned state
            if "messages" in final_state and final_state["messages"]:
                 print("Final Agent State Messages:", final_state["messages"])
                 # You usually want the content of the *last* message as the final answer
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
            print(traceback.format_exc()) # Print full traceback for debugging

        #ORIGINAL
        # try:
        #     # for event in agent.stream({"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]}):
        #     #     for value in event.values():
        #     #         print("Assistant:", value["messages"][-1].content)

        #     response = agent.invoke({"messages": messages})
        #     # response = agent.invoke({"messages": question})
        # except Exception as e:
        #     print(e)
        # if response.get("messages"):
        #     print("Agent response is:", response["messages"])
        # else:
        #     print("No valid messages returned.")

        # LANGGRAPH DOCUMENTATION STYLE
        # events = agent.stream(
        #     {"messages": [{"role": "user", "content": question}]},
        #     {"configurable": {"thread_id": "1"}},
        #     stream_mode="values",
        # )
        # for event in events:
        #     if "messages" in event:
        #         event["messages"][-1].pretty_print()

        # print("Agent response is:  ", response.messages[1])

        # print(f"Agent returning fixed answer: {response}")
        return final_state