import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from config import LLM_MODEL, SQLITE_DB_PATH, get_openai_api_key
from prompts import system_prompt
from rag import rag_tool
from tools import tools


all_tools = [*tools, rag_tool]
_llm_with_tools = None


def get_llm_with_tools():
    global _llm_with_tools

    if _llm_with_tools is None:
        api_key = get_openai_api_key()
        if not api_key:
            raise RuntimeError(
                "OpenAI credentials are missing. Set OPENAI_API_KEY in your environment or .env file."
            )

        llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
        _llm_with_tools = llm.bind_tools(all_tools)

    return _llm_with_tools


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    messages = [
        SystemMessage(content=system_prompt(thread_id)),
        *state["messages"],
    ]
    response = get_llm_with_tools().invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(all_tools)

conn = sqlite3.connect(database=SQLITE_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def invoke(messages, config):
    return chatbot.invoke({"messages": messages}, config=config)


def stream(messages, config):
    return chatbot.stream({"messages": messages}, config=config, stream_mode="messages")


def get_state(thread_id):
    return chatbot.get_state(config={"configurable": {"thread_id": thread_id}})


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
