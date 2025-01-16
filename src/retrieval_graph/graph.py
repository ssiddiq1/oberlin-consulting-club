from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
    filter_messages,
    trim_messages,
)
from langchain_core.prompts import PromptTemplate

from langchain_pinecone import PineconeVectorStore

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledGraph, StateGraph
from typing_extensions import List, Literal, Optional, Union, Sequence
from langgraph.types import Command
from langchain_core.prompts import format_document
from retrieval_graph.prompts import (
    answer_grader,
    hallucination_grader,
    retrieval_grader,
    answer_chain,
    condense_chain,
    no_answer_chain,
)

index_name = "consulting-research"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docsearch = PineconeVectorStore.from_existing_index(
    index_name, embeddings
).as_retriever()


def custom_doc_formatter(documents: List[str]) -> str:
    formatted_docs = []
    prompt = PromptTemplate.from_template("Title {title}: {page_content}")

    for doc in documents:
        formatted_docs.append(format_document(doc, prompt))
    return formatted_docs


def get_chat_history(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    chat_history = []
    for message in messages:
        if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(
            message, HumanMessage
        ):
            chat_history.append(message)

    trimmed_chat_history = trim_messages(
        chat_history,
        max_tokens=100_000,
        strategy="last",
        token_counter=ChatOpenAI(model="gpt-4o-mini"),
        start_on="human",
    )

    return trimmed_chat_history


def extract_latest_human_message(messages: List[Union[AIMessage, HumanMessage]]) -> str:
    """Extracts the content of the latest HumanMessage from a list of messages.

    Args:
        messages (List[Union[AIMessage, HumanMessage]]): List of AIMessage and HumanMessage objects.

    Returns:
        str: Content of the latest HumanMessage. Returns an empty string if no HumanMessage is found.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content


def extract_latest_ai_message(messages: List[Union[AIMessage, HumanMessage]]) -> str:
    """Extracts the content of the latest HumanMessage from a list of messages.

    Args:
        messages (List[Union[AIMessage, HumanMessage]]): List of AIMessage and HumanMessage objects.

    Returns:
        str: Content of the latest HumanMessage. Returns an empty string if no HumanMessage is found.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message.content


llm = ChatOpenAI(
    name="gpt-4o-mini",
    temperature=0.1,
)


class GraphState(MessagesState):
    rephrased_question: Optional[str]
    documents: Optional[List[str]]
    generated: Optional[str]


workflow = StateGraph(GraphState)

"""
HELPER FUNCTIONS
"""


def find_tool_calls(messages: list):
    """Find all tool calls in the messages returned"""
    tool_calls = [tc for m in messages[::-1] for tc in getattr(m, "tool_calls", [])]
    return tool_calls


"""
RETRIEVAL NODES
"""


@workflow.add_node
async def rephrase_question(state: GraphState) -> Command[Literal["retrieve"]]:
    messages = convert_to_messages(state["messages"])
    question = extract_latest_human_message(state["messages"])
    human_messages_list = filter_messages(messages, include_types=HumanMessage)

    trim_chat_history = get_chat_history(messages)

    if len(human_messages_list) >= 2:
        output = await condense_chain.ainvoke({"messages": trim_chat_history})
        rephrased_question = output.question

        return Command(
            update={"rephrased_question": rephrased_question}, goto="retrieve"
        )
    else:
        return Command(update={"rephrased_question": question}, goto="retrieve")


@workflow.add_node
async def retrieve(state: GraphState) -> Command[Literal["grade_documents"]]:
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["rephrased_question"]

    ret_documents = await docsearch.ainvoke(input=question)

    return Command(
        update={"documents": custom_doc_formatter(ret_documents)},
        goto="grade_documents",
    )


@workflow.add_node
async def grade_documents(
    state: GraphState,
) -> Command[Literal["generate", "display_not_useful_response"]]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with relevant documents
    """

    print("---CHECK RELEVANCE---")
    question = state["rephrased_question"]
    documents = state["documents"]

    batch_requests = [{"question": question, "document": doc} for doc in documents]

    # Invoke batch grading
    batch_scores = await retrieval_grader.abatch(batch_requests)

    filtered_docs = []
    for i, score in enumerate(batch_scores):
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(documents[i])
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    if len(filtered_docs) == 0:
        return Command(goto="display_not_useful_response")

    return Command(update={"documents": filtered_docs}, goto="generate")


@workflow.add_node
async def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    documents = state["documents"]
    question = state["rephrased_question"]

    generation = await answer_chain.ainvoke(
        {
            "context": "\n\n".join(documents),
            "question": question,
        }
    )

    return {"generated": generation.content}


@workflow.add_node
def display_response(state: GraphState) -> Command[Literal["__end__"]]:
    """
    Display response

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, messages, that contains the generated response
    """

    response = state["generated"]

    return Command(update={"messages": [AIMessage(response)]}, goto="__end__")


@workflow.add_node
def display_not_useful_response(state: GraphState) -> Command[Literal["__end__"]]:
    response = no_answer_chain.invoke({"question": state["rephrased_question"]})

    return Command(update={"messages": [response]}, goto="__end__")


"""
ROUTING LOGIC
"""


async def grade_generation_v_documents_and_question(
    state: GraphState,
) -> Literal["useful", "not useful", "not supported"]:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["rephrased_question"]
    documents = state["documents"]
    generation = state["generated"]

    # generation = extract_latest_ai_message(state["messages"])
    score = await hallucination_grader.ainvoke(
        {"documents": "\n\n".join(documents), "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = await answer_grader.ainvoke(
            {"question": question, "generation": generation}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def get_graph() -> CompiledGraph:
    workflow.add_edge(START, "rephrase_question")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "useful": "display_response",
            "not useful": "display_not_useful_response",
            "not supported": "display_not_useful_response",
        },
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


graph = get_graph()


async def main():
    import uuid

    async for chunk in graph.astream(
        {
            "messages": [
                {
                    "content": "Hi I would like to learn consumer habits in the United States.",
                    "role": "human",
                }
            ],
        },
        {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            },
        },
        stream_mode="updates",
        debug=True,
    ):
        print(chunk)
        for value in chunk.values():
            if "messages" in value:
                if isinstance(value["messages"], list):
                    last_message = value["messages"][0]
                    print(last_message.content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
