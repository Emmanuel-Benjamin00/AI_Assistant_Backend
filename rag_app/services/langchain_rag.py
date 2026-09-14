"""
The same RAG answer flow built with LangChain (LCEL), for comparison with the hand-written one.

    question
      -> RunnableLambda(retrieve)          same hybrid pgvector + full-text search as /api/ask/
      -> RunnableBranch: no results  -> fixed "no content" answer (no model call)
                         otherwise   -> RunnablePassthrough.assign(answer = prompt | chat model | StrOutputParser)
      -> {"question", "results", "answer"}

Retrieval deliberately reuses services.retrieval, so a comparison measures the orchestration
layer only. LangChain replaces: the prompt template, the model client (retries, provider
switch) and output parsing.
"""
from __future__ import annotations

from . import llm, rag, retrieval


class LangChainUnavailable(llm.LLMConfigError):
    """langchain-openai is not installed."""


PROMPT_MESSAGES = [
    ("system", rag.SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
]


def build_chat_model():
    """ChatOpenAI or AzureChatOpenAI, configured from the same env vars as services.llm."""
    try:
        from langchain_openai import AzureChatOpenAI, ChatOpenAI
    except ImportError as e:
        raise LangChainUnavailable("langchain-openai is not installed") from e

    common = {"timeout": llm.REQUEST_TIMEOUT_SECONDS, "max_retries": llm.MAX_ATTEMPTS - 1}
    if llm.LLM_PROVIDER == "azure":
        if not (llm.AZURE_OPENAI_ENDPOINT and llm.AZURE_OPENAI_API_KEY and llm.AZURE_OPENAI_CHAT_DEPLOYMENT):
            raise llm.LLMConfigError("Azure OpenAI endpoint, key and chat deployment must be set.")
        return AzureChatOpenAI(
            azure_endpoint=llm.AZURE_OPENAI_ENDPOINT,
            api_key=llm.AZURE_OPENAI_API_KEY,
            api_version=llm.AZURE_OPENAI_API_VERSION,
            azure_deployment=llm.AZURE_OPENAI_CHAT_DEPLOYMENT,
            **common,
        )
    if not llm.OPENAI_API_KEY:
        raise llm.LLMConfigError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(
        model=llm.OPENAI_CHAT_MODEL,
        api_key=llm.OPENAI_API_KEY,
        base_url=llm.OPENAI_BASE_URL,
        **common,
    )


def build_chain(top_k: int, mode: str, rerank: bool, chat_model=None):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

    prompt = ChatPromptTemplate.from_messages(PROMPT_MESSAGES)
    model = chat_model if chat_model is not None else build_chat_model()

    retrieve = RunnableLambda(
        lambda question: {
            "question": question,
            "results": retrieval.retrieve(question, top_k, mode=mode, rerank=rerank),
        }
    )
    to_prompt_vars = RunnableLambda(
        lambda x: {"context": rag.build_context(x["results"]), "question": x["question"]}
    )
    generate = RunnablePassthrough.assign(answer=to_prompt_vars | prompt | model | StrOutputParser())
    # Empty index: skip the model call, like /api/ask/.
    no_content = RunnablePassthrough.assign(answer=RunnableLambda(lambda _: rag.NO_CONTENT_ANSWER))
    return retrieve | RunnableBranch((lambda x: not x["results"], no_content), generate)


def ask(question: str, top_k: int, mode: str, rerank: bool = False, chat_model=None) -> dict:
    """Returns {"question", "results", "answer"}; raises llm.LLMConfigError / LLMRequestError."""
    chain = build_chain(top_k, mode, rerank, chat_model)
    try:
        return chain.invoke(question)
    except (llm.LLMConfigError, llm.LLMRequestError):
        raise
    except Exception as e:  # map provider SDK errors to our error types
        mapped = _map_error(e)
        if mapped is None:
            raise
        raise mapped from e


def _map_error(error: Exception) -> Exception | None:
    try:
        import openai
    except ImportError:
        return None
    if isinstance(error, openai.APIStatusError):
        return llm.LLMRequestError("LLM provider request failed", error.status_code)
    if isinstance(error, openai.APIError):
        return llm.LLMRequestError(f"LLM provider request failed ({error.__class__.__name__})")
    return None
