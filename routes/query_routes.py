from flask import Flask, request, jsonify
from controllers.query_controller import handle_query_sinhala
from controllers.query_controller import handle_query_english
from services.translation_service import translate_text_google
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/query/sinhala", methods=["POST"])
def ask_question_sinhala():
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    translated_query = translate_text_google(query, target_language="en")
    logger.info("\n translate_text query: %s \n", translated_query)
    if not translated_query:
        return jsonify({"error": "Translation failed for query"}), 500

    response = handle_query_sinhala(translated_query)

    return jsonify({"response": response})

@app.route("/query/english", methods=["POST"])
def ask_question_english():
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    logger.info("\n Received query: %s \n", query)

    response = handle_query_english(query)

    if not response:
        return jsonify({"error": "Failed to process query"}), 500

    return jsonify({"response": response})
    


if __name__ == "__main__":
    app.run(debug=True)
