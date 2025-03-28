import os
import time
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import speech_recognition as sr
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import required libraries
from pinecone import Pinecone, ServerlessSpec
from semantic_router.encoders import HuggingFaceEncoder
from groq import Groq
from deep_translator import GoogleTranslator

# Retrieve API Keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mommycareknowledgebase")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API keys
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set in environment variables.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment variables.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-west-2")

# Check and create Pinecone index if needed
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Index '{PINECONE_INDEX_NAME}' does not exist. Creating it...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=spec,
        deletion_protection=False
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready", False):
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)
time.sleep(1)

# Initialize encoder and Groq client
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")
groq_client = Groq(api_key=GROQ_API_KEY)

# Translation functions
def sinhalaToEnglish(query: str) -> str:
    """Translate Sinhala to English"""
    translator = GoogleTranslator(source="si", target="en")
    return translator.translate(query)

def englishToSinhala(text: str) -> str:
    """Translate English to Sinhala"""
    translator = GoogleTranslator(source="en", target="si")
    return translator.translate(text)

def get_docs(query: str, top_k: int = 5) -> List[dict]:
    """
    Encodes the query and retrieves top_k matching chunks from the Pinecone index.
    Returns a list of metadata dictionaries.
    """
    xq = encoder([query])
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])
    if not matches:
        print("No matching documents found.")
        return []
    return [match["metadata"] for match in matches]

def generate_answer(query: str, docs: List[dict]) -> str:
    """
    Generates an answer using Groq's chat API with context from retrieved documents.
    """
    if not docs:
        return "I'm sorry, I couldn't find any relevant information. Please consult your doctor for medical advice."

    # Prepare context text
    context_texts = [doc.get("text", "") for doc in docs]
    context = "\n---\n".join(context_texts)

    system_message = (
        "You are a compassionate and helpful medical chatbot designed for mothers. "
        "Answer questions in a friendly and supportive manner. "
        "CONTEXT:\n" + context
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    try:
        chat_response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages
        )
        answer = chat_response.choices[0].message.content
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return answer + "\n\n"

# FastAPI Application
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/get_answer/")
def get_answer(request: QueryRequest):
    """Get answer in English"""
    try:
        english_query = sinhalaToEnglish(request.query)
        docs = get_docs(english_query, top_k=5)
        answer = generate_answer(request.query, docs)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_answer_sinhala/")
def get_answer_sinhala(request: QueryRequest):
    """Get answer in Sinhala"""
    try:
        english_query = sinhalaToEnglish(request.query)
        docs = get_docs(english_query, top_k=5)
        english_answer = generate_answer(english_query, docs)
        sinhala_answer = englishToSinhala(english_answer)
        return {"answer": sinhala_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_answer_voice/")
async def get_answer_voice(audio_file: UploadFile = File(...)):
    """Get answer from voice input"""
    temp_audio_path = None
    try:
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        content = await audio_file.read()
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(content)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text_query = recognizer.recognize_google(audio_data)
        
        docs = get_docs(text_query, top_k=5)
        answer = generate_answer(text_query, docs)
        return {"query": text_query, "answer": answer}
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Speech could not be understood")
    except sr.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Speech recognition service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass

# Optional: Local development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)