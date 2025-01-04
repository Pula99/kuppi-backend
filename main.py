import logging
from flask_cors import CORS
from routes.query_routes import app
from configs.vertexai_config import init_vertex_ai
from services.document_service import load_data_from_s3
from services.chromadb_service import create_data_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORS(app, origins=["*"])

def main():
    init_vertex_ai()

    data = load_data_from_s3()
    create_data_store(data)

    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
