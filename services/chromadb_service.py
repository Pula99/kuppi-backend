import shutil
import os
import re
import time
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.schema import Document
from services.document_service import split_text
from vertexai.generative_models import GenerativeModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"
vertex_embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

def safe_rmtree(path):
    """Remove a directory safely with retries on PermissionError."""
    retries = 5
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(1)
    logger.warning(f"Failed to delete {path} after {retries} attempts.")

def create_data_store(documents):
    """Create a data store from the provided documents."""
    chunks = split_text(documents)
    logger.info(f"Chunks generated: {len(chunks)}")
    if not chunks:
        logger.warning("No chunks to process!")
        return 
    save_to_chroma(chunks)

def save_to_chroma(chunks: list[Document], batch_size: int = 50):
    """Save document chunks to the Chroma database after generating embeddings."""
    if os.path.exists(CHROMA_PATH):
        safe_rmtree(CHROMA_PATH)

    if not chunks:
        logger.warning("No chunks to save!")
        return 

    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_texts = [chunk.page_content for chunk in batch_chunks]
        
        embeddings = vertex_embeddings.embed_documents(batch_texts)
        all_embeddings.extend(embeddings)
        logger.info(f"Embeddings generated for batch {i // batch_size + 1}: {len(embeddings)}")

    if any(len(embedding) == 0 for embedding in all_embeddings):
        logger.error("One or more embeddings are empty!")
        return

    db = Chroma.from_documents(chunks, vertex_embeddings, persist_directory=CHROMA_PATH)
    logger.info(f"Stored {len(chunks)} chunks in the vector store.")

def search_data_on_db(query: str):
    """Search the Chroma database for the query and return relevant results with a relevance score > 0.35."""
    if not os.path.exists(CHROMA_PATH):
        logger.error("No data store found!")
        return []

    try:
        # Load the persisted Chroma database
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=vertex_embeddings)

        # Generate embedding for the query
        query_embedding = vertex_embeddings.embed_documents([query])[0]
        if not query_embedding:
            logger.error("Query embedding is empty!")
            return []

        # Perform the similarity search with relevance scores
        search_results = db.similarity_search_with_relevance_scores(query, k=150)  # Retrieve top 150 results
        
        # Filter results based on relevance score > 0.35
        filtered_results = [(doc, score) for doc, score in search_results if score > 0.35]
        
        if not filtered_results:
            logger.info("No results found with relevance score > 0.5.")
            return []

        logger.info(f"Found {len(filtered_results)} results with relevance score > 0.5.")
        return filtered_results

    except Exception as e:
        logger.error(f"Error during database search: {e}")
        return []
