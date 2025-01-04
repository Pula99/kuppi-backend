import os
import logging
import boto3
from botocore.exceptions import NoCredentialsError
import fitz  
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_BUCKET_NAME = "kuppi-bucket"
S3_REGION = "ap-south-1"
s3_client = boto3.client('s3', region_name=S3_REGION)

def upload_file_to_s3(file_path, bucket_name, object_name=None):
    """Upload a file to S3."""
    if not object_name:
        object_name = os.path.basename(file_path)
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        logger.info(f"Uploaded {file_path} to S3 bucket {bucket_name} as {object_name}.")
        return True
    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        return False
    except Exception as e:
        logger.error(f"Error uploading file to S3: {e}")
        return False

def download_file_from_s3(bucket_name, object_name, download_path):
    """Download a file from S3."""
    try:
        s3_client.download_file(bucket_name, object_name, download_path)
        logger.info(f"Downloaded {object_name} from S3 bucket {bucket_name}.")
        return download_path
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        return None

def list_files_in_s3(bucket_name, prefix=""):
    """List files in an S3 bucket."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            files = [content['Key'] for content in response['Contents']]
            logger.info(f"Files in bucket {bucket_name}: {files}")
            return files
        else:
            logger.info(f"No files found in bucket {bucket_name} with prefix {prefix}.")
            return []
    except Exception as e:
        logger.error(f"Error listing files in S3: {e}")
        return []
    

def load_data_from_s3():
    """Load PDF files from S3 bucket, extract text, and return as Documents."""
    files = list_files_in_s3(S3_BUCKET_NAME)
    data = []
    for file_key in files:
        if file_key.endswith(".pdf"):
            # Download file locally
            local_path = f"temp_{os.path.basename(file_key)}"
            download_path = download_file_from_s3(S3_BUCKET_NAME, file_key, local_path)
            if download_path:
                text = extract_text_from_pdf(download_path)
                os.remove(download_path)  # Cleanup local temp file
                if text.strip():
                    document = Document(page_content=text, metadata={"filename": file_key})
                    data.append(document)
                else:
                    logger.warning(f"No text extracted from {file_key}")
    return data

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    if len(text) > 500:
        logger.info(f"Extracted {len(text)} characters from {file_path}")
    else:
        logger.warning(f"Extracted too little text from {file_path}")
    return text

def split_text(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Generated {len(chunks)} chunks.")
    return chunks

