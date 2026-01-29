import os
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

load_dotenv()

# Tamanho do lote para inserção no vetor (progresso com tqdm)
ADD_DOCUMENTS_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "50"))

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
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
    print("=" * 60)
    print("Ingestão de PDF para vetor (RAG)")
    print("=" * 60)

    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF não encontrado: {PDF_PATH}")
    print(f"[1/5] PDF: {PDF_PATH}")

    print("[2/5] Inicializando modelo de embeddings...")
    embeddings = _get_embeddings()
    print("      Modelo de embeddings carregado.")

    print("[3/5] Carregando páginas do PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"      {len(documents)} página(s) carregada(s).")

    print("[4/5] Fragmentando texto (chunk_size=1000, overlap=150)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = []
    for doc in tqdm(documents, desc="      Fragmentando páginas", unit="pág"):
        chunks.extend(splitter.split_documents([doc]))
    print(f"      {len(chunks)} chunk(s) gerado(s).")

    url = DATABASE_URL
    if url.startswith("postgresql://") and "psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)

    print("[5/5] Conectando ao PostgreSQL e inserindo vetores...")
    vector_store = PGVector(
        connection=url,
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
    )
    batches = [
        chunks[i : i + ADD_DOCUMENTS_BATCH_SIZE]
        for i in range(0, len(chunks), ADD_DOCUMENTS_BATCH_SIZE)
    ]
    for batch in tqdm(batches, desc="      Inserindo lotes", unit="lote"):
        vector_store.add_documents(batch)

    print("=" * 60)
    print(f"Ingestão concluída: {len(chunks)} chunks de '{PDF_PATH}'")
    print(f"Coleção: '{PG_VECTOR_COLLECTION_NAME}'")
    print("=" * 60)


if __name__ == "__main__":
    ingest_pdf()
