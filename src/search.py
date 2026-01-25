import os
from dotenv import load_dotenv
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_chunks")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.0-flash")


def _get_embeddings():
    if OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    if GOOGLE_API_KEY:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    return None


def _get_llm():
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_LLM_MODEL, openai_api_key=OPENAI_API_KEY)
    if GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, google_api_key=GOOGLE_API_KEY)
    return None


def _get_vector_store(embeddings):
    url = DATABASE_URL
    if url.startswith("postgresql://") and "psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return PGVector(
        connection_string=url,
        embedding=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
    )


def search_prompt(question):
    """
    Vetoriza a pergunta, busca os 10 resultados mais relevantes no banco vetorial,
    monta o prompt, chama a LLM e retorna a resposta.
    """
    if not question or not question.strip():
        return "Não tenho informações necessárias para responder sua pergunta."

    embeddings = _get_embeddings()
    llm = _get_llm()
    if not embeddings or not llm:
        return "Erro: configure OPENAI_API_KEY ou GOOGLE_API_KEY no .env"

    vector_store = _get_vector_store(embeddings)

    # similarity_search_with_score(query, k=10)
    try:
        results = vector_store.similarity_search_with_score(question.strip(), k=10)
    except Exception as e:
        return f"Erro na busca: {e}. Verifique se o banco está no ar e se a ingestão foi executada."

    contexto = "\n\n".join(doc.page_content for doc, _ in results) if results else ""

    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question.strip())

    try:
        resposta = llm.invoke(prompt)
        return resposta.content if hasattr(resposta, "content") else str(resposta)
    except Exception as e:
        return f"Erro ao chamar a LLM: {e}"
