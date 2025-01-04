import os
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai

def init_vertex_ai():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vertexAIconfig.json'
    
    credentials = service_account.Credentials.from_service_account_file('vertexAIconfig.json')

    aiplatform.init(
        project="stories-434205",
        location="us-central1",
        credentials=credentials
    )   

    vertexai.init(project="stories-434205", location="us-central1")
