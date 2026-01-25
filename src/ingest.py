import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/rag")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_chunks")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")


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
    raise ValueError(
        "Configure OPENAI_API_KEY ou GOOGLE_API_KEY no .env"
    )


def ingest_pdf():
    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF não encontrado: {PDF_PATH}")

    embeddings = _get_embeddings()
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # Garantir formato postgresql+psycopg para psycopg3
    url = DATABASE_URL
    if url.startswith("postgresql://") and "psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    vector_store = PGVector(
        connection_string=url,
        embedding=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
    )
    vector_store.add_documents(chunks)
    print(f"Ingestão concluída: {len(chunks)} chunks de '{PDF_PATH}' em '{PG_VECTOR_COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_pdf()
