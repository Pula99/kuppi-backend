from google.cloud import translate_v2 as translate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def translate_text_google(text, target_language):
    """Translate text using Google Cloud Translation API."""
    client = translate.Client()

    try:
        result = client.translate(text, target_language=target_language)
        return result.get("translatedText")
    except Exception as e:
        logger.error(f"Error in Google Translate API: {e}")
        return None
