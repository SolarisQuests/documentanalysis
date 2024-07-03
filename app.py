import os
import json
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from datetime import datetime
import pytz
import openai
from werkzeug.utils import secure_filename
from docx2pdf import convert

# Load environment variables
load_dotenv()

# Set your Form Recognizer and OpenAI API keys and endpoints
form_recognizer_endpoint = os.getenv("AZURE_OCR_ENDPOINT")
form_recognizer_key = os.getenv("AZURE_OCR_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_uri = os.getenv("MONGODB_URI")
database_name = os.getenv("DATABASE_NAME")
collection_name = os.getenv("COLLECTION_NAME")

document_analysis_client = DocumentAnalysisClient(
    endpoint=form_recognizer_endpoint, credential=AzureKeyCredential(form_recognizer_key)
)

# Initialize OpenAI client
openai.api_key = openai_api_key

# Initialize FastAPI application
app = FastAPI()

# MongoDB client setup
mongo_client = MongoClient(mongo_uri)
db = mongo_client[database_name]
collection = db[collection_name]

def analyze_document(file_path):
    try:
        with open(file_path, "rb") as f:
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-document", document=f.read()
            )
        result = poller.result()

        extracted_data = []
        for page in result.pages:
            page_content = " ".join([line.content for line in page.lines])
            extracted_data.append({str(page.page_number - 1): page_content})
        return extracted_data
    except Exception as e:
        raise

def get_openai_response(messages):
    try:
        chat_completion = openai.ChatCompletion.create(
            messages=messages,
            model="gpt-3.5-turbo",
            max_tokens=1500
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        raise

def process_ocr_output(ocr_output):
    try:
        corrected_output_parts = []
        
        for page in ocr_output:
            page_content = list(page.values())[0]
            messages = [
                {"role": "system", "content": "You are a helpful assistant that fixes errors in OCR outputs and provides correct data in the same JSON format."},
                {"role": "user", "content": f"Fix the errors and get correct data in same JSON format:\n{page_content}"}
            ]
            response = get_openai_response(messages)
            corrected_output_parts.append({list(page.keys())[0]: response})
        
        return corrected_output_parts
    except json.JSONDecodeError as e:
        raise

def process_document(file_path):
    try:
        extracted_data = analyze_document(file_path)
        if not extracted_data:
            return {"status": "failed", "message": "No data extracted"}
        corrected_output = process_ocr_output(extracted_data)
        processed_date = datetime.now(pytz.timezone('UTC')).isoformat()
        return {
            "status": "processed",
            "json_data": corrected_output,
            "ocr_output": extracted_data,
            "processed_date": processed_date
        }
    except Exception as e:
        return {"status": "failed", "message": str(e)}

def convert_docx_to_pdf(input_path, output_path):
    try:
        convert(input_path, output_path)
        if not os.path.exists(output_path):
            raise Exception(f"PDF conversion failed, output file not found: {output_path}")
    except Exception as e:
        raise

@app.get("/")
def read_root():
    return {"status": "success"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".pdf", ".doc", ".docx"]:
            raise HTTPException(status_code=400, detail="Unsupported document type. Only PDF, DOC, and DOCX files are allowed.")
        temp_file_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        if file_extension == ".docx":
            pdf_temp_file_path = os.path.splitext(temp_file_path)[0] + ".pdf"
            convert_docx_to_pdf(temp_file_path, pdf_temp_file_path)
            temp_file_path = pdf_temp_file_path
        processing_result = process_document(temp_file_path)

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        if file_extension == ".docx" and os.path.exists(pdf_temp_file_path):
            os.remove(pdf_temp_file_path)

        return JSONResponse(content=processing_result, status_code=200 if processing_result["status"] == "processed" else 500)

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
