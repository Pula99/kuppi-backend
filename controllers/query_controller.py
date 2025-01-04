from services.chromadb_service import search_data_on_db
from services.vertexai_service import give_answer_english
from services.vertexai_service import give_answer_sinhala
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_query_sinhala(translated_query):
    logger.info("\n Received query: %s \n", translated_query)

    search_data = search_data_on_db(translated_query)
    
    if search_data:
        response = give_answer_sinhala(translated_query, search_data)
        return {"response": response}
    else:
        return {"response": "No relevant data found."}
    

def handle_query_english(query):
    logger.info("\n Received query: %s \n", query)

    search_data = search_data_on_db(query)

    if search_data:
        response = give_answer_english(query, search_data)
        return {"response" : response}
    else:
        return {"response": "No relevant data found"}