import asyncio
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import Client
from retrieval_graph.graph import graph
import uuid
from dotenv import load_dotenv, find_dotenv

st.set_page_config(
    page_title="Oberlin Consulting AI Agent",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("Oberlin Consulting AI Agent")
button_css = """.stButton>button {
    color: #4F8BF9;
    border-radius: 50%;
    height: 2em;
    width: 2em;
    font-size: 4px;
}"""
st.markdown(f"<style>{button_css}</style>", unsafe_allow_html=True)

load_dotenv(find_dotenv())


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


client = Client()

set_state_if_absent(
    key="run_id",
    value=None,
)
set_state_if_absent(key="thread_id", value=str(uuid.uuid4()))


def set_thread_id():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state._thread_id = st.session_state.thread_id


with st.sidebar:
    st.button(label="Thread ID", help="Set new thread id", on_click=set_thread_id)

st.info(f"Current Thread ID: **{st.session_state.thread_id}**")


msgs = StreamlitChatMessageHistory(key="langchain_messages")

conversation_history = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
)

if len(msgs.messages) == 0 or st.sidebar.button(
    "Clear Chat",
    help="Clear the conversation",
):
    print("Clearing message history")
    set_thread_id()
    conversation_history.clear()

    msgs.add_ai_message("Hello and welcome to the Oberlin Consulting AI Agent!")


config = {
    "configurable": {
        "thread_id": str(st.session_state.thread_id),
    },
    "recursion_limit": 1000,
    "stream_mode": "updates",
}

messages = st.container(border=False, height=500)
for msg in msgs.messages:
    if msg.content and msg.type in ("ai", "assistant", "function", "human", "user"):
        messages.chat_message(
            msg.type,
            avatar="🤖" if msg.type in ("ai", "assistant") else "👤",
        ).write(msg.content)


def stream_response(text: str):
    for msg in text:
        yield msg


async def process_stream(input, config):
    response_generated = ""
    with st.status("Loading...", expanded=True) as status:
        async for output in graph.astream(input, config):
            for key, value in output.items():
                st.markdown(f"Node: **{key.upper()}**")
                if key.lower() in ["display_response", "display_not_useful_response"]:
                    messages = value.get("messages", [])
                    if isinstance(messages[-1], AIMessage):
                        if messages[-1].content:
                            response_generated += messages[0].content
                        else:
                            tool_call = messages[-1].tool_calls[0]
                            st.json(tool_call["args"])

        status.update(label="Completed", state="complete", expanded=False)
        return response_generated


if prompt := st.chat_input(placeholder="Ask me a question!"):
    messages.chat_message("human", avatar="👤").write(prompt)

    with messages.chat_message("ai", avatar="🤖"):
        response_generated = asyncio.run(
            process_stream(
                {
                    "messages": [HumanMessage(content=prompt)],
                },
                config,
            )
        )
        st.write_stream(stream_response(response_generated))
        msgs.add_user_message(prompt)
        msgs.add_ai_message(response_generated)
