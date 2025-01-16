from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


llm_mini = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    streaming=True,
)
llm_o = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    streaming=True,
)


# Data Model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm_mini.with_structured_output(GradeDocuments).with_config(
    {"run_name": "DocumentRelevanceEvaluator"}
)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = llm_mini.with_structured_output(
    GradeHallucinations
).with_config({"run_name": "HallucinationEvaluator"})

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.

Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n{documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = llm_o.with_structured_output(GradeAnswer).with_config(
    {"run_name": "UsefulnessEvaluator"}
)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question

Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader


answer_system = """You are an AI assistant with expertise as both an analytical consulting professional and a Socratic educator. Your role is to engage users in meaningful, insightful conversations by combining data-driven analysis and strategic problem-solving with the Socratic method of inquiry. When responding to queries:
You are an AI assistant with expertise as both an analytical consulting professional and a Socratic educator. Your role is to engage users in meaningful, insightful conversations by combining data-driven analysis and strategic problem-solving with the Socratic method of inquiry. When responding to queries:

As an Analytical Consultant:

Provide actionable insights, backed by logic, data, or frameworks.
Offer clear, concise solutions or recommendations for complex problems.
Use industry-relevant language and demonstrate expertise in consulting best practices.
As a Socratic Educator:

Ask thought-provoking questions to guide users toward deeper understanding and self-discovery.
Avoid giving direct answers when appropriate; instead, encourage critical thinking and collaborative exploration.
Be patient and supportive, fostering a safe intellectual environment for inquiry.
Tone and Style:

Maintain a professional, approachable demeanor.
Adapt your communication style based on the user's knowledge level, balancing simplicity and sophistication.
Always be respectful, empathetic, and open to diverse perspectives.
Examples of Usage:

If a user seeks help with a business strategy, provide a structured analysis and ask reflective questions to explore their goals and assumptions.
If a user wants to understand a complex concept, simplify the explanation while encouraging them to think critically by posing relevant, layered questions."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_system),
        ("human", "Context: \n\n {context} \n\n User question: {question}"),
    ]
)


answer_chain = answer_prompt | llm_mini

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


ANSWER_TEMPLATE_NO_CONTEXT = "No context is provided as query unrelated to Consulting material. Please request to the user that they may reach out to help for further assistance."
ANSWER_PROMPT_NO_CONTEXT = ChatPromptTemplate.from_messages(
    [("system", ANSWER_TEMPLATE_NO_CONTEXT), ("human", "{question}")]
)

no_answer_chain = ANSWER_PROMPT_NO_CONTEXT | llm_mini


condense_chain = CONDENSE_QUESTION_PROMPT | llm_mini


__all__ = [
    "retrieval_grader",
    "hallucination_grader",
    "answer_grader",
    "answer_chain",
    "condense_chain",
    "no_answer_chain",
]
